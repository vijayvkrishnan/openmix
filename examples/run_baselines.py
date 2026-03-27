#!/usr/bin/env python3
"""
FormulaBench Baseline Runner — compares feature tiers on shampoo stability.

Tests the hypothesis: rules alone < ML alone < domain knowledge + ML.

Usage:
    python examples/run_baselines.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

from openmix.benchmarks import ShampooStability
from openmix.benchmarks.features import (
    tier1_features, tier2_features, tier3_features,
    TIER1_NAMES, TIER2_NAMES, TIER3_NAMES,
)
from openmix.knowledge.loader import load_knowledge


def run():
    ds = ShampooStability()
    kb = load_knowledge()
    print(ds)
    print()

    train, val, test = ds.split_random(test_size=0.15, val_size=0.15, seed=42)

    print(f"Train: {len(train)} ({sum(1 for r in train if r.stable)} stable)")
    print(f"Val:   {len(val)} ({sum(1 for r in val if r.stable)} stable)")
    print(f"Test:  {len(test)} ({sum(1 for r in test if r.stable)} stable)")
    print()

    # Build feature matrices for each tier
    tiers = {
        "Tier 1: Raw Composition": (tier1_features, TIER1_NAMES),
        "Tier 2: + Molecular Descriptors": (tier2_features, TIER2_NAMES),
        "Tier 3: + Domain Knowledge": (
            lambda r: tier3_features(r, kb), TIER3_NAMES
        ),
    }

    results = {}

    for tier_name, (feat_fn, feat_names) in tiers.items():
        X_train = np.array([feat_fn(r) for r in train], dtype=np.float32)
        X_val = np.array([feat_fn(r) for r in val], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test], dtype=np.float32)
        y_train = np.array([1 if r.stable else 0 for r in train])
        y_val = np.array([1 if r.stable else 0 for r in val])
        y_test = np.array([1 if r.stable else 0 for r in test])

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        results[tier_name] = {"auc": auc, "acc": acc, "f1": f1}

        # Top features
        importances = model.feature_importances_
        top5 = np.argsort(importances)[::-1][:5]
        top_feats = [(feat_names[i], importances[i]) for i in top5]

        print(f"{tier_name}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  AUROC:    {auc:.3f}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1 Macro: {f1:.3f}")
        print(f"  Top features:")
        for name, imp in top_feats:
            print(f"    {name}: {imp:.3f}")
        print()

    # Summary table
    print("=" * 60)
    print(f"{'Model':<40} {'AUROC':>7} {'Acc':>7} {'F1':>7}")
    print("-" * 60)
    print(f"{'OpenMix Heuristic (no ML)':<40} {'0.500':>7} {'---':>7} {'---':>7}")
    for name, r in results.items():
        print(f"{name:<40} {r['auc']:>7.3f} {r['acc']:>7.3f} {r['f1']:>7.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
