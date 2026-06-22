#!/usr/bin/env python3
"""
Which experiments to run first: sample-efficient active learning on real labels.

The shampoo dataset stands in for a lab: we hide the labels, let an acquisition
strategy choose which formulations to "measure" (reveal labels for), retrain, and
track how fast it (a) improves out-of-distribution prediction and (b) surfaces the
failures the model did not see coming. Labels are real, so this tests the
acquisition LOGIC, not a simulator.

Framing (the experiment-selection scenario): the model starts from a small slice of "known"
formulations and must learn a NOVEL region (formulations using ingredients held out
of the seed) by choosing which novel formulations to measure. We compare strategies
against random sampling and report the acceleration factor.

Run shampoo_study.py first (writes the feature cache).

Usage:
    python examples/shampoo_acquisition.py            # full
    python examples/shampoo_acquisition.py --quick    # smoke test
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from openmix.benchmarks import ShampooStability
from openmix.benchmarks.shampoo import INGREDIENT_COLS

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "cache" / "shampoo_features.npz"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "experiments" / "figures"

SEED_N = 40          # known-space seed (cold start)
BATCH = 24           # measurements per round
ROUNDS = 8           # budget = BATCH * ROUNDS = 192
K_ENSEMBLE = 5       # committee size for QBC/BALD
STRATEGIES = ["random", "uncertainty", "qbc", "diversity", "hybrid"]
GRID = [BATCH * r for r in range(ROUNDS + 1)]


def model(seed=0):
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=seed, eval_metric="logloss", verbosity=0,
    )


def _proba(m, X, idx):
    return m.predict_proba(X[idx])[:, 1]


def select_batch(strategy, X, Xs, cur_model, train_idx, remaining, rng):
    """Return BATCH indices (subset of `remaining`) chosen by the strategy.

    `cur_model` is the campaign's model already fit on the current train set.
    """
    rem = np.array(remaining)
    b = min(BATCH, len(rem))

    if strategy == "random":
        return list(rng.choice(rem, size=b, replace=False))

    if strategy == "uncertainty":
        p = _proba(cur_model, X, rem)
        order = np.argsort(np.abs(p - 0.5))      # most uncertain first
        return list(rem[order[:b]])

    if strategy == "qbc":
        votes = np.zeros((K_ENSEMBLE, len(rem)))
        for k in range(K_ENSEMBLE):
            boot = rng.choice(train_idx, size=len(train_idx), replace=True)
            mk = model(k).fit(X[boot], y[boot])
            votes[k] = _proba(mk, X, rem)
        disagree = votes.var(axis=0)             # committee variance
        order = np.argsort(-disagree)
        return list(rem[order[:b]])

    if strategy == "diversity":
        return _kcenter(Xs, train_idx, rem, b)

    if strategy == "hybrid":
        p = _proba(cur_model, X, rem)
        unc_order = np.argsort(np.abs(p - 0.5))
        cand = rem[unc_order[: 3 * b]]           # uncertain candidate band
        return _kcenter(Xs, train_idx, list(cand), b)  # diversify within it

    raise ValueError(strategy)


def _kcenter(Xs, train_idx, remaining, b):
    """Greedy k-center: pick points farthest from the already-measured set."""
    rem = np.array(remaining)
    chosen = []
    min_d = np.linalg.norm(Xs[rem][:, None, :] - Xs[train_idx][None, :, :], axis=2).min(axis=1)
    for _ in range(min(b, len(rem))):
        j = int(np.argmax(min_d))
        chosen.append(int(rem[j]))
        d_new = np.linalg.norm(Xs[rem] - Xs[rem[j]], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[j] = -1.0                          # don't repick
    return chosen


def run_campaign(strategy, X, Xs, seed_idx, pool_idx, test_idx, rng):
    """One acquisition campaign; returns per-round (n, auroc, cum_fail, cum_surprise)."""
    train_idx = list(seed_idx)
    selected: set[int] = set()
    cum_fail = cum_surprise = 0
    rows = []

    m = model(0).fit(X[train_idx], y[train_idx])
    rows.append((0, roc_auc_score(y[test_idx], _proba(m, X, test_idx)), 0, 0))

    for r in range(1, ROUNDS + 1):
        remaining = [i for i in pool_idx if i not in selected]
        if not remaining:
            break
        chosen = select_batch(strategy, X, Xs, m, train_idx, remaining, rng)
        # "surprising" = model-of-the-moment predicted stable, truth unstable
        p_chosen = _proba(m, X, np.array(chosen))
        cum_fail += int(sum(y[i] == 0 for i in chosen))
        cum_surprise += int(sum((p_chosen[j] >= 0.5) and (y[i] == 0) for j, i in enumerate(chosen)))
        selected.update(chosen)
        train_idx = list(seed_idx) + list(selected)
        m = model(0).fit(X[train_idx], y[train_idx])
        rows.append((r * BATCH, roc_auc_score(y[test_idx], _proba(m, X, test_idx)),
                     cum_fail, cum_surprise))
    return rows


def novel_region_folds(n_folds):
    """Held-out-ingredient definitions of a 'novel' region with enough pool."""
    counts, st, un = Counter(), Counter(), Counter()
    for r in ds.records:
        for c in INGREDIENT_COLS:
            if r.ingredients.get(c, 0) > 0:
                counts[c] += 1
                (st if r.stable else un)[c] += 1
    common = [c for c, _ in counts.most_common() if st[c] >= 8 and un[c] >= 8][:10]
    col_idx = {c: i for i, c in enumerate(INGREDIENT_COLS)}
    comp = np.array([r.feature_vector for r in ds.records], dtype=np.float32)

    folds, combos = [], list(itertools.combinations(common, 3))
    np.random.RandomState(0).shuffle(combos)
    for combo in combos:
        held = [col_idx[c] for c in combo]
        novel = np.where((comp[:, held] > 0).any(axis=1))[0]
        known = np.where(~(comp[:, held] > 0).any(axis=1))[0]
        if len(novel) < (BATCH * ROUNDS + 60) or len(known) < SEED_N + 20:
            continue
        if len(np.unique(y[novel])) < 2:
            continue
        folds.append((combo, known, novel))
        if len(folds) >= n_folds:
            break
    return folds


def aggregate(all_rows):
    """all_rows[strategy] = list of campaigns; each campaign = list of (n,auc,fail,surp)."""
    out = {}
    for strat, campaigns in all_rows.items():
        by_n = {n: {"auc": [], "fail": [], "surp": []} for n in GRID}
        for camp in campaigns:
            for (n, auc, fail, surp) in camp:
                if n in by_n:
                    by_n[n]["auc"].append(auc)
                    by_n[n]["fail"].append(fail)
                    by_n[n]["surp"].append(surp)
        out[strat] = {
            n: {k: _mci(v[k]) for k in ("auc", "fail", "surp")}
            for n, v in by_n.items() if v["auc"]
        }
    return out


def _mci(vals):
    a = np.asarray(vals, float)
    m = a.mean()
    sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": float(m), "lo": float(m - 1.96 * sem), "hi": float(m + 1.96 * sem)}


def acceleration(agg):
    """Measurements each strategy needs to reach random's final AUROC."""
    target = agg["random"][GRID[-1]]["auc"]["mean"]
    rand_n = GRID[-1]
    res = {}
    for strat in STRATEGIES:
        hit = next((n for n in GRID if agg[strat][n]["auc"]["mean"] >= target), None)
        res[strat] = {"reaches_at": hit, "accel_vs_random": (rand_n / hit) if hit else None}
    return target, res


def plot(agg, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for strat in STRATEGIES:
        ns = [n for n in GRID if n in agg[strat]]
        auc = [agg[strat][n]["auc"]["mean"] for n in ns]
        lo = [agg[strat][n]["auc"]["lo"] for n in ns]
        hi = [agg[strat][n]["auc"]["hi"] for n in ns]
        ax[0].plot(ns, auc, marker="o", label=strat)
        ax[0].fill_between(ns, lo, hi, alpha=0.12)
        surp = [agg[strat][n]["surp"]["mean"] for n in ns]
        ax[1].plot(ns, surp, marker="o", label=strat)
    ax[0].set_xlabel("Measurements")
    ax[0].set_ylabel("OOD test AUROC (mean, 95% CI)")
    ax[0].set_title("Learning the novel region")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("Measurements")
    ax[1].set_ylabel("Cumulative surprising failures found")
    ax[1].set_title("Failures the model did not see coming")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_folds = 2 if args.quick else 6
    seeds = [0] if args.quick else [0, 1, 2]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats = np.load(CACHE)["tier4"]
    folds = novel_region_folds(n_folds)
    print(f"Folds: {len(folds)}  seeds: {len(seeds)}  budget: {BATCH*ROUNDS} measurements")

    all_rows = {s: [] for s in STRATEGIES}
    for (combo, known, novel) in folds:
        scaler = StandardScaler().fit(feats[np.concatenate([known, novel])])
        Xs = scaler.transform(feats)
        for sd in seeds:
            seed_idx, _ = train_test_split(known, train_size=SEED_N, stratify=y[known], random_state=sd)
            pool_idx, test_idx = train_test_split(novel, test_size=0.3, stratify=y[novel], random_state=sd)
            for strat in STRATEGIES:
                all_rows[strat].append(
                    run_campaign(strat, feats, Xs, seed_idx, list(pool_idx), test_idx,
                                 np.random.RandomState(sd * 100 + STRATEGIES.index(strat)))
                )
        print(f"  fold {combo} done")

    agg = aggregate(all_rows)
    target, accel = acceleration(agg)

    print("\n" + "=" * 66)
    print(f"  OOD AUROC by measurements (random's final = {target:.3f})")
    print("=" * 66)
    print(f"  {'strategy':<12} " + " ".join(f"{n:>5}" for n in GRID))
    for s in STRATEGIES:
        print(f"  {s:<12} " + " ".join(f"{agg[s][n]['auc']['mean']:.3f}"[1:] for n in GRID))
    print("\n  Acceleration vs random (to reach random's final AUROC):")
    for s in STRATEGIES:
        a = accel[s]
        tag = f"{a['accel_vs_random']:.2f}x at {a['reaches_at']} meas" if a["accel_vs_random"] else "not reached"
        print(f"    {s:<12} {tag}")
    print("\n  Surprising failures found by budget end (mean):")
    for s in STRATEGIES:
        print(f"    {s:<12} {agg[s][GRID[-1]]['surp']['mean']:.1f}")

    fig = FIG_DIR / "shampoo_acquisition.png"
    plot(agg, fig)
    (RESULTS_DIR / "shampoo_acquisition.json").write_text(
        json.dumps({"target_auroc": target, "acceleration": accel, "curves": agg,
                    "config": {"seed_n": SEED_N, "batch": BATCH, "rounds": ROUNDS,
                               "folds": [list(c) for c, _, _ in folds], "seeds": seeds}},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  Figure: {fig.relative_to(REPO)}")
    print("  Results: experiments/results/shampoo_acquisition.json")


if __name__ == "__main__":
    ds = ShampooStability()
    y = np.array([1 if r.stable else 0 for r in ds.records], dtype=np.int32)
    run()
