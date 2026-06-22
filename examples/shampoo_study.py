#!/usr/bin/env python3
"""
Rigorous shampoo stability study (FormulaBench Task 1).

Upgrades examples/run_baselines.py from single-seed point estimates to the
evaluation the FormulaBench spec actually calls for:

  A. Tier 1->4 feature ablation with repeated stratified splits + 95% CIs
  B. Robust leave-ingredients-out: rotated hold-out folds, not one cherry-pick
  C. Data-efficiency curve (AUROC vs N training samples) against the published
     LLM result (~0.70 AUROC at 20 samples; Bigan & Dufour, Cosmetics 2025)
  D. Calibration (expected calibration error + reliability)

The question under test is OpenMix's own hypothesis that formulation-aware
features beat raw composition (Tier1 < Tier2 < Tier3 < Tier4). The single-seed
baseline suggests they do not; this run measures whether that holds with
variance, OOD, and at low N.

Real data: Chitre/Goldsworthy et al. 2024 (Scientific Data), 812 formulations,
294 stable / 518 unstable, CC-BY-4.0.

Usage:
    python examples/shampoo_study.py            # full run
    python examples/shampoo_study.py --quick    # fewer seeds (smoke test)
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from openmix.benchmarks import ShampooStability
from openmix.benchmarks.features import (
    tier1_features,
    tier2_features,
    tier3_features,
    tier4_features,
)
from openmix.benchmarks.shampoo import INGREDIENT_COLS
from openmix.knowledge.loader import load_knowledge


REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "cache"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "experiments" / "figures"

TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
TIER_LABEL = {
    "Tier 1": "Tier 1: raw composition",
    "Tier 2": "Tier 2: + molecular descriptors",
    "Tier 3": "Tier 3: + domain knowledge",
    "Tier 4": "Tier 4: + physics observations",
}
# Published LLM reference on this exact dataset (Bigan & Dufour, Cosmetics 2025).
PUBLISHED_LLM_AUROC_AT_20 = 0.70


def make_model(seed: int) -> XGBClassifier:
    """The FormulaBench baseline classifier (matches run_baselines.py)."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Bin-wise |accuracy - confidence|, weighted by bin population."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        k = int(mask.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def mean_ci(values) -> dict:
    """Mean and 95% CI (normal approx across repeats) plus spread."""
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sem = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {
        "mean": mean,
        "ci_lo": mean - 1.96 * sem,
        "ci_hi": mean + 1.96 * sem,
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def featurize_all(ds: ShampooStability) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Compute every tier's feature matrix once over all 812 records (the
    expensive step), cache to disk, and return {tier -> (N, d)} plus labels.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "shampoo_features.npz"
    y = np.array([1 if r.stable else 0 for r in ds.records], dtype=np.int32)

    if cache.exists():
        data = np.load(cache)
        feats = {t: data[t] for t in ("tier1", "tier2", "tier3", "tier4")}
        print(f"  Loaded cached features from {cache.relative_to(REPO)}")
        return _rename(feats), y

    kb = load_knowledge()
    print("  Computing features for 812 records x 4 tiers (one-time)...", flush=True)
    t0 = time.time()
    tier1 = np.array([tier1_features(r) for r in ds.records], dtype=np.float32)
    tier2 = np.array([tier2_features(r) for r in ds.records], dtype=np.float32)
    tier3 = np.array([tier3_features(r, kb) for r in ds.records], dtype=np.float32)
    tier4 = np.array([tier4_features(r, kb) for r in ds.records], dtype=np.float32)
    print(f"  done ({time.time() - t0:.1f}s)")

    np.savez(cache, tier1=tier1, tier2=tier2, tier3=tier3, tier4=tier4)
    return _rename({"tier1": tier1, "tier2": tier2, "tier3": tier3, "tier4": tier4}), y


def _rename(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Tier 1": feats["tier1"],
        "Tier 2": feats["tier2"],
        "Tier 3": feats["tier3"],
        "Tier 4": feats["tier4"],
    }


def _fit_eval(X, y, train_idx, test_idx, seed: int) -> dict:
    model = make_model(seed)
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(int)
    yt = y[test_idx]
    return {
        "auc": roc_auc_score(yt, prob),
        "bal_acc": balanced_accuracy_score(yt, pred),
        "f1": f1_score(yt, pred, average="macro"),
        "ece": expected_calibration_error(yt, prob),
        "prob": prob,
        "y": yt,
    }


# ---------------------------------------------------------------------------
# A. Tier ablation, repeated random splits
# ---------------------------------------------------------------------------

def experiment_random(feats, y, seeds) -> dict:
    print("\n" + "=" * 70)
    print(f"  A. TIER ABLATION - repeated random splits (n={len(seeds)} seeds)")
    print("=" * 70)
    idx = np.arange(len(y))
    out: dict[str, dict] = {}
    pooled_prob: dict[str, list] = {t: [] for t in TIER_ORDER}
    pooled_y: dict[str, list] = {t: [] for t in TIER_ORDER}

    for tier in TIER_ORDER:
        X = feats[tier]
        runs = {"auc": [], "bal_acc": [], "f1": [], "ece": []}
        for s in seeds:
            tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=s)
            r = _fit_eval(X, y, tr, te, s)
            for k in runs:
                runs[k].append(r[k])
            pooled_prob[tier].append(r["prob"])
            pooled_y[tier].append(r["y"])
        out[tier] = {k: mean_ci(v) for k, v in runs.items()}
        a = out[tier]["auc"]
        e = out[tier]["ece"]
        print(
            f"  {TIER_LABEL[tier]:<34} AUROC {a['mean']:.3f} "
            f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]   ECE {e['mean']:.3f}"
        )

    # Reliability data (pooled across seeds) for the figure, per tier.
    reliability = {
        t: (np.concatenate(pooled_y[t]), np.concatenate(pooled_prob[t]))
        for t in TIER_ORDER
    }
    return out, reliability


# ---------------------------------------------------------------------------
# B. Robust leave-ingredients-out
# ---------------------------------------------------------------------------

def _common_ingredients(ds, top: int = 10, min_per_class: int = 8) -> list[str]:
    counts, stable, unstable = Counter(), Counter(), Counter()
    for r in ds.records:
        for col in INGREDIENT_COLS:
            if r.ingredients.get(col, 0) > 0:
                counts[col] += 1
                (stable if r.stable else unstable)[col] += 1
    eligible = [
        c for c, _ in counts.most_common()
        if stable[c] >= min_per_class and unstable[c] >= min_per_class
    ]
    return eligible[:top]


def experiment_lio(ds, feats, y, n_folds: int = 10, min_test: int = 40) -> dict:
    print("\n" + "=" * 70)
    print("  B. LEAVE-INGREDIENTS-OUT - rotated hold-out folds")
    print("=" * 70)
    common = _common_ingredients(ds)
    print(f"  Common ingredients (in both classes): {len(common)}")

    # Folds = distinct 3-ingredient hold-out sets drawn from the common pool.
    combos = list(itertools.combinations(common, 3))
    rng = np.random.RandomState(42)
    rng.shuffle(combos)

    col_idx = {c: i for i, c in enumerate(INGREDIENT_COLS)}
    comp = np.array([r.feature_vector for r in ds.records], dtype=np.float32)

    out: dict[str, dict] = {}
    per_tier_runs = {t: {"auc": [], "bal_acc": [], "f1": [], "ece": []} for t in TIER_ORDER}
    used_folds = []

    for combo in combos:
        held = [col_idx[c] for c in combo]
        uses_held = (comp[:, held] > 0).any(axis=1)
        test_idx = np.where(uses_held)[0]
        train_idx = np.where(~uses_held)[0]
        yt = y[test_idx]
        if len(test_idx) < min_test or len(np.unique(yt)) < 2 or len(train_idx) < 100:
            continue
        used_folds.append(combo)
        for tier in TIER_ORDER:
            r = _fit_eval(feats[tier], y, train_idx, test_idx, seed=42)
            for k in per_tier_runs[tier]:
                per_tier_runs[tier][k].append(r[k])
        if len(used_folds) >= n_folds:
            break

    print(f"  Usable folds: {len(used_folds)}")
    for tier in TIER_ORDER:
        out[tier] = {k: mean_ci(v) for k, v in per_tier_runs[tier].items()}
        a = out[tier]["auc"]
        print(
            f"  {TIER_LABEL[tier]:<34} AUROC {a['mean']:.3f} "
            f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]"
        )
    out["_folds"] = [list(c) for c in used_folds]
    return out


# ---------------------------------------------------------------------------
# C. Data-efficiency curve
# ---------------------------------------------------------------------------

def experiment_data_efficiency(feats, y, seeds, n_grid) -> dict:
    print("\n" + "=" * 70)
    print("  C. DATA EFFICIENCY - AUROC vs N training samples")
    print("=" * 70)
    idx = np.arange(len(y))
    curves: dict[str, dict] = {}

    for tier in TIER_ORDER:
        X = feats[tier]
        per_n = {}
        for n in n_grid:
            aucs = []
            for s in seeds:
                # Fixed stratified test set per seed; subsample N from the rest.
                pool, test = train_test_split(
                    idx, test_size=0.25, stratify=y, random_state=s
                )
                if n >= len(pool):
                    sub = pool
                else:
                    sub, _ = train_test_split(
                        pool, train_size=n, stratify=y[pool], random_state=s
                    )
                model = make_model(s)
                model.fit(X[sub], y[sub])
                prob = model.predict_proba(X[test])[:, 1]
                aucs.append(roc_auc_score(y[test], prob))
            per_n[n] = mean_ci(aucs)
        curves[tier] = per_n
        row = "  ".join(f"N={n}:{per_n[n]['mean']:.3f}" for n in n_grid)
        print(f"  {TIER_LABEL[tier]:<34} {row}")

    # Where does each tier first cross the published LLM's 20-sample AUROC?
    for tier in TIER_ORDER:
        crossing = next(
            (n for n in n_grid if curves[tier][n]["mean"] >= PUBLISHED_LLM_AUROC_AT_20),
            None,
        )
        if crossing is not None:
            print(
                f"    {tier} reaches {PUBLISHED_LLM_AUROC_AT_20:.2f} AUROC "
                f"by N={crossing} (LLM needs ~20)"
            )
    return curves


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_data_efficiency(curves, n_grid, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tier in TIER_ORDER:
        means = [curves[tier][n]["mean"] for n in n_grid]
        los = [curves[tier][n]["ci_lo"] for n in n_grid]
        his = [curves[tier][n]["ci_hi"] for n in n_grid]
        ax.plot(n_grid, means, marker="o", label=TIER_LABEL[tier])
        ax.fill_between(n_grid, los, his, alpha=0.12)
    ax.axhline(
        PUBLISHED_LLM_AUROC_AT_20, ls="--", color="gray",
        label="Published LLM ~0.70 @ 20 (Bigan & Dufour 2025)",
    )
    ax.axvline(20, ls=":", color="gray", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.set_xlabel("Training samples (N)")
    ax.set_ylabel("Test AUROC (mean, 95% CI)")
    ax.set_title("Shampoo stability: data efficiency by feature tier")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(reliability, path: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for tier in TIER_ORDER:
        yt, prob = reliability[tier]
        acc = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
            acc.append(yt[mask].mean() if mask.sum() else np.nan)
        ax.plot(centers, acc, marker="o", label=TIER_LABEL[tier])
    ax.set_xlabel("Predicted P(stable)")
    ax.set_ylabel("Observed fraction stable")
    ax.set_title("Reliability (pooled across random splits)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer seeds")
    args = parser.parse_args()

    seeds = list(range(8)) if args.quick else list(range(25))
    n_grid = [20, 40, 80, 160, 320, 570]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = ShampooStability()
    print(ds)
    feats, y = featurize_all(ds)

    random_res, reliability = experiment_random(feats, y, seeds)
    lio_res = experiment_lio(ds, feats, y)
    eff_res = experiment_data_efficiency(feats, y, seeds, n_grid)

    eff_fig = FIG_DIR / "shampoo_data_efficiency.png"
    rel_fig = FIG_DIR / "shampoo_reliability.png"
    plot_data_efficiency(eff_res, n_grid, eff_fig)
    plot_reliability(reliability, rel_fig)

    # ----- headline summary -----
    print("\n" + "=" * 70)
    print("  SUMMARY (mean AUROC, 95% CI)")
    print("=" * 70)
    print(f"  {'':34} {'Random':>18} {'Leave-Ingredients-Out':>24}")
    for tier in TIER_ORDER:
        r = random_res[tier]["auc"]
        lio = lio_res[tier]["auc"]
        print(
            f"  {TIER_LABEL[tier]:<34} "
            f"{r['mean']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]   "
            f"{lio['mean']:.3f} [{lio['ci_lo']:.3f},{lio['ci_hi']:.3f}]"
        )
    print(f"\n  Figures: {eff_fig.relative_to(REPO)} | {rel_fig.relative_to(REPO)}")

    payload = {
        "dataset": "shampoo_stability_812 (Chitre/Goldsworthy 2024, CC-BY-4.0)",
        "n_formulations": len(y),
        "n_stable": int(y.sum()),
        "seeds": seeds,
        "n_grid": n_grid,
        "published_llm_auroc_at_20": PUBLISHED_LLM_AUROC_AT_20,
        "random_split": random_res,
        "leave_ingredients_out": lio_res,
        "data_efficiency": eff_res,
    }
    out_json = RESULTS_DIR / "shampoo_study.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"  Results: {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    run()
