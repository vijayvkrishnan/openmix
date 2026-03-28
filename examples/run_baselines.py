#!/usr/bin/env python3
"""
FormulaBench Baseline Runner — compares feature tiers on shampoo stability.

Tests the hypothesis: raw composition < molecular descriptors < domain knowledge
                      < physics observations.

Two evaluation protocols:
  1. Random split — standard train/test
  2. Leave-ingredients-out — tests generalization to unseen ingredient combinations

Usage:
    python examples/run_baselines.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

from openmix.benchmarks import ShampooStability
from openmix.benchmarks.features import (
    tier1_features, tier2_features, tier3_features, tier4_features,
    TIER1_NAMES, TIER2_NAMES, TIER3_NAMES, TIER4_NAMES,
)
from openmix.benchmarks.shampoo import INGREDIENT_COLS
from openmix.knowledge.loader import load_knowledge


def evaluate(X_train, y_train, X_test, y_test, seed=42):
    """Train XGBoost and return metrics."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    return {
        "auc": roc_auc_score(y_test, probs),
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
        "model": model,
    }


def leave_ingredients_out_split(ds, hold_out_ingredients, seed=42):
    """
    Split by ingredient presence: test set contains formulations
    that use held-out ingredients. Trains on formulations without them.

    This tests whether features generalize to unseen ingredient combinations.
    """
    rng = np.random.RandomState(seed)

    hold_out_cols = set(hold_out_ingredients)
    test_records = []
    train_pool = []

    for r in ds.records:
        uses_held_out = any(
            r.ingredients.get(col, 0) > 0
            for col in hold_out_cols
        )
        if uses_held_out:
            test_records.append(r)
        else:
            train_pool.append(r)

    # Subsample train to avoid extreme imbalance
    rng.shuffle(train_pool)
    return train_pool, test_records


def run():
    ds = ShampooStability()
    kb = load_knowledge()
    print(ds)
    print()

    # ===== Protocol 1: Random Split =====
    print("=" * 70)
    print("  PROTOCOL 1: Random Split (standard)")
    print("=" * 70)
    print()

    train, val, test = ds.split_random(test_size=0.15, val_size=0.15, seed=42)
    print(f"  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    print()

    tiers = {
        "Tier 1: Raw Composition": (tier1_features, TIER1_NAMES),
        "Tier 2: + Molecular Descriptors": (tier2_features, TIER2_NAMES),
        "Tier 3: + Domain Knowledge": (
            lambda r: tier3_features(r, kb), TIER3_NAMES
        ),
        "Tier 4: + Physics Observations": (
            lambda r: tier4_features(r, kb), TIER4_NAMES
        ),
    }

    random_results = {}

    for tier_name, (feat_fn, feat_names) in tiers.items():
        X_train = np.array([feat_fn(r) for r in train], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test], dtype=np.float32)
        y_train = np.array([1 if r.stable else 0 for r in train])
        y_test = np.array([1 if r.stable else 0 for r in test])

        result = evaluate(X_train, y_train, X_test, y_test)
        random_results[tier_name] = result

        # Top features
        importances = result["model"].feature_importances_
        top5 = np.argsort(importances)[::-1][:5]
        top_feats = [(feat_names[i], importances[i]) for i in top5]

        print(f"  {tier_name}")
        print(f"    Features: {X_train.shape[1]}")
        print(f"    AUROC:    {result['auc']:.3f}")
        print(f"    Accuracy: {result['acc']:.3f}")
        print(f"    F1 Macro: {result['f1']:.3f}")
        print(f"    Top features: {', '.join(f'{n} ({v:.3f})' for n, v in top_feats[:3])}")
        print()

    # ===== Protocol 2: Leave-Ingredients-Out =====
    print("=" * 70)
    print("  PROTOCOL 2: Leave-Ingredients-Out (generalization)")
    print("=" * 70)
    print()

    # Hold out 3 surfactants the model hasn't seen in training
    hold_out = ["Dehyton MC", "Plantacare 818", "Plantapon Amino KG-L"]
    train_lio, test_lio = leave_ingredients_out_split(ds, hold_out)
    print(f"  Held out: {', '.join(hold_out)}")
    print(f"  Train: {len(train_lio)}  |  Test: {len(test_lio)}")
    test_stable = sum(1 for r in test_lio if r.stable)
    print(f"  Test composition: {test_stable} stable, {len(test_lio) - test_stable} unstable")
    print()

    lio_results = {}

    for tier_name, (feat_fn, feat_names) in tiers.items():
        X_train = np.array([feat_fn(r) for r in train_lio], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test_lio], dtype=np.float32)
        y_train = np.array([1 if r.stable else 0 for r in train_lio])
        y_test = np.array([1 if r.stable else 0 for r in test_lio])

        if len(np.unique(y_test)) < 2:
            print(f"  {tier_name}: skipped (single class in test)")
            continue

        result = evaluate(X_train, y_train, X_test, y_test)
        lio_results[tier_name] = result

        print(f"  {tier_name}")
        print(f"    AUROC:    {result['auc']:.3f}")
        print(f"    Accuracy: {result['acc']:.3f}")
        print(f"    F1 Macro: {result['f1']:.3f}")
        print()

    # ===== Summary =====
    print("=" * 70)
    print(f"  {'Model':<40} {'Random':>8} {'LIO':>8}")
    print(f"  {'':40} {'AUROC':>8} {'AUROC':>8}")
    print("  " + "-" * 56)
    for tier_name in tiers:
        r_auc = random_results.get(tier_name, {}).get("auc", 0)
        l_auc = lio_results.get(tier_name, {}).get("auc", 0)
        print(f"  {tier_name:<40} {r_auc:>8.3f} {l_auc:>8.3f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
