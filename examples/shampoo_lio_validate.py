#!/usr/bin/env python3
"""
Stress-test the leave-ingredients-out (OOD) finding from shampoo_study.py.

The random split showed feature tiers are a wash; a single LIO fold once showed
Tier 4 worse, the 10-fold run showed Tier 4 better. That fold-sensitivity means
the OOD claim must be tested as a PAIRED comparison across every usable fold,
averaged over model seeds, before we trust it.

Run shampoo_study.py first (it writes the feature cache).

Usage:
    python examples/shampoo_lio_validate.py
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from openmix.benchmarks import ShampooStability
from openmix.benchmarks.shampoo import INGREDIENT_COLS

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "cache" / "shampoo_features.npz"
MODEL_SEEDS = [0, 1, 2, 3, 4]
MIN_TEST = 40


def model(seed):
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=seed, eval_metric="logloss", verbosity=0,
    )


def fold_auc(X, y, tr, te):
    aucs = []
    for s in MODEL_SEEDS:
        m = model(s)
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    if not CACHE.exists():
        raise SystemExit("Feature cache missing - run examples/shampoo_study.py first.")

    ds = ShampooStability()
    y = np.array([1 if r.stable else 0 for r in ds.records], dtype=np.int32)
    data = np.load(CACHE)
    t1, t4 = data["tier1"], data["tier4"]
    comp = np.array([r.feature_vector for r in ds.records], dtype=np.float32)

    # Top-10 ingredients present in both classes.
    counts, stable, unstable = Counter(), Counter(), Counter()
    for r in ds.records:
        for c in INGREDIENT_COLS:
            if r.ingredients.get(c, 0) > 0:
                counts[c] += 1
                (stable if r.stable else unstable)[c] += 1
    common = [c for c, _ in counts.most_common()
              if stable[c] >= 8 and unstable[c] >= 8][:10]
    col_idx = {c: i for i, c in enumerate(INGREDIENT_COLS)}

    diffs, t1s, t4s, sizes = [], [], [], []
    for combo in itertools.combinations(common, 3):
        held = [col_idx[c] for c in combo]
        uses = (comp[:, held] > 0).any(axis=1)
        te = np.where(uses)[0]
        tr = np.where(~uses)[0]
        if len(te) < MIN_TEST or len(np.unique(y[te])) < 2 or len(tr) < 100:
            continue
        a1 = fold_auc(t1, y, tr, te)
        a4 = fold_auc(t4, y, tr, te)
        t1s.append(a1)
        t4s.append(a4)
        diffs.append(a4 - a1)
        sizes.append(len(te))

    diffs = np.array(diffs)
    n = len(diffs)
    sem = diffs.std(ddof=1) / np.sqrt(n)
    wins = int((diffs > 0).sum())
    try:
        _, p = wilcoxon(diffs)
    except ValueError:
        p = float("nan")

    print(f"Usable folds (all C(10,3)): {n}")
    print(f"  Tier 1 mean OOD AUROC: {np.mean(t1s):.3f}")
    print(f"  Tier 4 mean OOD AUROC: {np.mean(t4s):.3f}")
    print(f"  Paired delta (T4-T1):  {diffs.mean():+.3f}  "
          f"95% CI [{diffs.mean()-1.96*sem:+.3f}, {diffs.mean()+1.96*sem:+.3f}]")
    print(f"  Folds where T4 > T1:   {wins}/{n}")
    print(f"  Wilcoxon signed-rank p: {p:.4f}")
    print(f"  (each fold averaged over {len(MODEL_SEEDS)} model seeds)")


if __name__ == "__main__":
    main()
