# Shampoo Stability — Benchmark Results (FormulaBench Task 1)

**Date:** 2026-06-19
**Dataset:** Liquid (shampoo) formulations, Chitre/Goldsworthy et al. 2024, *Scientific Data* (DOI 10.1038/s41597-024-03573-w), CC-BY-4.0. 812 formulations, 294 stable / 518 unstable, 18 ingredients (12 surfactants, 4 conditioning polymers, 2 thickeners).
**Task:** binary phase-stability classification from composition.
**Pipeline:** `examples/shampoo_study.py`, `examples/shampoo_lio_validate.py`. Reproducible from committed code + data.

---

## 1. What this measures

Whether formulation-aware features beat raw composition for predicting stability, evaluated the way the FormulaBench spec requires: repeated splits with confidence intervals, an out-of-distribution protocol, a data-efficiency curve, and calibration. This supersedes the single-seed point estimates in `examples/run_baselines.py`.

Four feature tiers (cumulative), all fed to the same XGBoost classifier (200 trees, depth 4, lr 0.1):

- **Tier 1** — raw ingredient percentages + derived charge ratios (28 features)
- **Tier 2** — + aggregated molecular descriptors (36)
- **Tier 3** — + OpenMix domain-knowledge scores and rule-violation counts (47)
- **Tier 4** — + pairwise interaction terms (coacervation potential, charge interaction, amphoteric buffering) and physics observations (63)

## 2. Results

### 2a. Random split (25 stratified seeds, mean [95% CI])

| Tier | AUROC | ECE |
|---|---|---|
| 1: raw composition | 0.855 [0.845, 0.865] | 0.091 |
| 2: + molecular descriptors | 0.853 [0.845, 0.862] | 0.094 |
| 3: + domain knowledge | 0.853 [0.845, 0.862] | 0.094 |
| 4: + physics observations | 0.856 [0.847, 0.865] | 0.092 |

**In-distribution, the tiers are statistically indistinguishable.** Raw composition is as good as the full physics-aware feature set. (A single seed had shown Tier 2 at 0.874; across 25 seeds that was noise. Single-seed benchmarking of this dataset is unreliable.)

### 2b. Leave-ingredients-out (out-of-distribution generalization)

Hold out all formulations containing a set of ingredients; train on the rest; test on the held-out set. This measures generalization to unseen ingredient combinations rather than memorization of ingredient-outcome correlations.

Validated as a **paired** comparison across **all 112 usable 3-ingredient hold-out folds** (drawn from the 10 ingredients present in both classes), each fold averaged over 5 model seeds:

| Tier | OOD AUROC |
|---|---|
| 1: raw composition | 0.658 |
| 4: + physics observations | 0.700 |

- Paired delta (Tier 4 − Tier 1): **+0.042, 95% CI [+0.034, +0.050]**
- Tier 4 beats Tier 1 in **97 of 112 folds**; Wilcoxon signed-rank **p < 0.0001**

**Out-of-distribution, formulation-aware features matter.** The physics/interaction features add nothing when the test resembles the train set, but deliver a consistent, significant gain when the model faces novel ingredient combinations — the regime that matters for real use.

### 2c. Data efficiency (AUROC vs N training samples)

Fixed stratified test set; training subsampled to N; mean over 25 seeds.

| N | 20 | 40 | 80 | 160 | 320 | 570 |
|---|---|---|---|---|---|---|
| AUROC (Tier 1) | 0.60 | 0.65 | 0.72 | 0.78 | 0.82 | 0.85 |

All tiers track together and cross **0.70 AUROC by ~N=80**. For reference, an LLM approach on this dataset (Bigan & Dufour, *Cosmetics* 2025) reports ~0.70 AUROC at ~20 samples — i.e., the LLM is more sample-efficient at very low N, while the classical model catches up by ~N=80 and is far cheaper to run. Figure: `experiments/figures/shampoo_data_efficiency.png`.

### 2d. Calibration

ECE ≈ 0.09 across tiers (mildly overconfident at the extremes). Reliability: `experiments/figures/shampoo_reliability.png`.

## 3. The finding

1. Feature engineering is a **wash in-distribution** and a **measurable win out-of-distribution** (+0.042 AUROC, p < 1e-4). Physics-informed features encode transferable structure; raw ingredient identities do not generalize to unseen ingredients.
2. Even the best feature set **plateaus around 0.70 AUROC out-of-distribution**. The remaining gap is a property of learning from a small, single-condition, survivorship-biased dataset — not of the model class. Breaking it requires more measured data, especially on novel formulations and failures.
3. **Report OOD, not just random split.** Random-split numbers (~0.85) overstate real-world performance by ~15 AUROC points relative to the leave-ingredients-out reality (~0.70).

## 4. Active learning: which formulations to measure first

Treating the dataset as a stand-in lab (hide the labels, let a strategy pick which formulations to "measure," reveal the labels, retrain), we compare acquisition strategies on the leave-ingredients-out task — starting from a small known-ingredient seed and acquiring novel-ingredient formulations (`examples/shampoo_acquisition.py`, 6 folds x 3 seeds).

- **Uncertainty sampling** reaches random sampling's full-budget OOD AUROC in ~1.3x fewer measurements. The effect is modest and the CIs overlap, as expected for a low-dimensional task where active learning's advantage is small.
- It also surfaces **~1.6x more "surprising failures"** (formulations the current model predicted stable but are actually unstable) than random sampling.
- Pure exploration (feature-space diversity) is no better than random; committee disagreement is strong on failures but slower on AUROC.

When measurements are scarce, sampling where the model is least certain both learns the novel region fastest and surfaces the model's blind-spot failures fastest.

## 5. Limitations

- Single category (rinse-off shampoo), 812 samples, binary label under one test condition. No failure-mode breakdown.
- The OOD protocol holds out ingredients; it does not test transfer to other product categories (no multi-category data yet).
- XGBoost baseline only; a learned permutation-invariant aggregator has not yet been benchmarked on this OOD protocol.

## 6. Reproduce

```bash
pip install -e ".[bench,rdkit]"
python examples/shampoo_study.py          # tier ablation, OOD, data efficiency, calibration
python examples/shampoo_lio_validate.py   # paired 112-fold OOD validation
python examples/shampoo_acquisition.py    # active learning: what to measure first
```
