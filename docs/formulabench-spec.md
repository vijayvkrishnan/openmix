# FormulaBench: A Benchmark for Formulation Stability and Compatibility Prediction

**Status:** Draft v0.1
**Last updated:** 2026-03-25

## 1. What FormulaBench Is

FormulaBench is a benchmark for evaluating models that predict **formulation stability** and **ingredient compatibility** in multi-component mixtures. It targets consumer products (cosmetics, personal care, food) and pharmaceutical formulations.

**Relationship to CheMixHub (arXiv 2506.12231):** CheMixHub covers thermophysical mixture properties — density, viscosity, excess enthalpy, solubility — across ~500K data points of binary/ternary mixtures. FormulaBench is complementary: it focuses on the *formulation-level* outcomes that CheMixHub does not address. A shampoo with 12 ingredients may have individually predictable viscosities, but whether it phase-separates after 3 months at 40C is a different question. FormulaBench benchmarks that question.

| Concern | CheMixHub | FormulaBench |
|---|---|---|
| Property type | Thermophysical (density, viscosity, enthalpy) | Stability, compatibility, shelf life |
| Mixture size | Binary/ternary | Full formulations (3-30+ ingredients) |
| Outcome | Continuous physical property | Binary stability, compatibility score, shelf life in months |
| Domain | General chemistry | Consumer products, pharma |
| Data source | NIST TDE, Dortmund | Published stability studies, industry datasets |

## 2. Prediction Tasks

### Task 1: Stability Binary Classification
Given a complete formulation (ingredient list + percentages + processing conditions), predict whether it is **stable** or **unstable** after accelerated aging.

- **Label:** `stable` / `unstable` (binary)
- **Definition:** A formulation is stable if it passes accelerated stability testing (typically 3 months at 40C/75% RH) with no phase separation, discoloration, viscosity drift >15%, or microbial growth.

### Task 2: Pairwise Compatibility Scoring
Given two ingredients, their concentrations, and a pH, predict a **compatibility score** (0-100).

- **Label:** Continuous 0-100 (or categorical: compatible / conditional / incompatible)
- **Sources:** Known interaction rules (e.g., Vitamin C + Niacinamide at pH <3.5), published incompatibility tables, expert annotations.

### Task 3: Shelf Life Regression
Given a formulation and storage conditions, predict **shelf life in months** before first failure criterion is breached.

- **Label:** Continuous (months), right-censored for formulations that passed the full study duration.
- **Failure criteria:** Phase separation, pH drift >0.5 units, viscosity change >15%, color delta-E >3, preservative efficacy failure.

### Task 4: Failure Mode Classification
Given an unstable formulation, predict the **primary failure mode**.

- **Labels:** `phase_separation`, `pH_drift`, `viscosity_change`, `discoloration`, `microbial`, `precipitation`, `oxidation`
- **Multi-label allowed** (a formulation can exhibit multiple failure modes).

## 3. Data Schema

Each record in FormulaBench follows this structure:

```yaml
formulation:
  id: "SHAM-0042"
  name: "Sulfate-free volumizing shampoo"
  category: "personal-care"             # personal-care | skincare | pharma | food | home-care
  product_type: "shampoo"
  ingredients:
    - inci_name: "WATER"
      percentage: 62.0
      phase: "A"
      function: "solvent"
      cas_number: "7732-18-5"           # optional
      smiles: "O"                       # optional
    - inci_name: "SODIUM LAUROYL METHYL ISETHIONATE"
      percentage: 12.0
      phase: "A"
      function: "surfactant"
    # ... remaining ingredients
  target_ph: 5.5                        # optional
  total_percentage: 100.0

conditions:
  temperature_c: 40                     # storage/test temperature
  humidity_rh: 75                       # relative humidity (%)
  duration_months: 3                    # study duration
  container: "HDPE bottle"              # optional
  light_exposure: "ambient"             # optional: ambient | dark | UV-accelerated

outcome:
  stable: true                          # Task 1 label
  shelf_life_months: 18                 # Task 3 label (null if unknown)
  failure_modes: []                     # Task 4 labels (empty if stable)
  notes: "Passed 3-month accelerated"   # free text

metadata:
  source: "mdpi-cosmetics-2024"         # dataset origin
  doi: "10.3390/cosmetics11010001"      # publication DOI if available
  license: "CC-BY-4.0"
  contributed_by: "original-authors"
```

Pairwise compatibility records use a simpler schema:

```yaml
pair:
  id: "COMPAT-0107"
  ingredient_a: { inci_name: "ASCORBIC ACID", percentage: 10.0, cas_number: "50-81-7" }
  ingredient_b: { inci_name: "NIACINAMIDE", percentage: 5.0, cas_number: "98-92-0" }
  ph: 3.0
  category: "skincare"

outcome:
  compatible: false
  score: 25                             # 0-100
  mechanism: "pH_conflict"              # oxidation | pH_conflict | charge | chelation | ...
  notes: "Niacin flushing at low pH; Vitamin C degrades above pH 3.5"
  source: "cosmetic-chemistry-literature"
```

All datasets are distributed as JSONL files. One line per record, UTF-8 encoded.

## 4. Planned Datasets

### 4.1 Shampoo Stability (v0.1 -- launch dataset)
- **Source:** MDPI Cosmetics 2024 (doi pending extraction)
- **Size:** 812 formulations, 294 stable / 518 unstable
- **Features:** Full ingredient lists with percentages, surfactant systems, conditioning agents, preservatives
- **Labels:** Binary stability after standard accelerated testing
- **Status:** Available. Needs schema normalization.

### 4.2 Skincare Compatibility (v0.2)
- **Source:** OpenMix interaction rules (curated from literature) + expert chemist annotations
- **Size:** Target 500+ pairwise interactions with conditions
- **Features:** Ingredient pairs, pH, concentration, category
- **Labels:** Compatible/conditional/incompatible + mechanism

### 4.3 Pharmaceutical Emulsions (v0.3)
- **Source:** Published stability studies (USP/ICH accelerated conditions)
- **Size:** Target 200-500 formulations
- **Features:** API + excipient lists, processing parameters, storage conditions
- **Labels:** Stability, shelf life, failure modes

### 4.4 Food & Beverage (v0.4)
- **Source:** Published formulation studies, industry partnerships
- **Size:** Target 300+ formulations
- **Features:** Ingredient lists, pH, water activity, processing temperatures
- **Labels:** Stability (emulsion separation, sedimentation, microbial), shelf life

## 5. Evaluation Metrics

### Stability Classification (Tasks 1, 4)
| Metric | Description |
|---|---|
| **AUROC** | Primary metric. Threshold-independent discrimination. |
| **F1 (macro)** | Harmonic mean of precision and recall, macro-averaged across classes. |
| **Calibration ECE** | Expected calibration error. A model that says "80% stable" should be right 80% of the time. |
| **Balanced accuracy** | Accounts for class imbalance (shampoo dataset is 36% stable / 64% unstable). |

### Shelf Life Regression (Task 3)
| Metric | Description |
|---|---|
| **MAE** | Mean absolute error in months. Primary metric. |
| **RMSE** | Root mean squared error. Penalizes large misses. |
| **Concordance index** | Ranking accuracy for censored data. |

### Compatibility Scoring (Task 2)
| Metric | Description |
|---|---|
| **AUROC** | For the binary compatible/incompatible classification. |
| **MAE** | For the 0-100 score prediction. |
| **Mechanism accuracy** | For predicting the interaction mechanism (if labels available). |

All metrics are computed on the held-out test split. Confidence intervals via bootstrap (n=1000).

## 6. Data Splitting Strategies

Each dataset ships with three pre-computed splits:

### Split A: Random (baseline)
- 70% train / 15% validation / 15% test
- Stratified by stability label
- Seed: 42

### Split B: Leave-Ingredients-Out (generalization)
- Hold out all formulations containing specific ingredients not seen during training
- Tests whether the model generalizes to novel ingredients vs. memorizing ingredient-outcome correlations
- Implementation: identify the 10 most common ingredients that appear in both stable and unstable formulations; hold out 3 of them for test

### Split C: Leave-Category-Out (domain transfer)
- Train on N-1 product categories, test on the held-out category
- Example: train on skincare + pharma, test on personal-care
- Only applicable when multiple category datasets are available (v0.3+)

Leaderboard submissions must report results on all applicable splits.

## 7. Baseline Models

### Baseline 1: OpenMix Heuristic Score
The `openmix.score.score()` function computes a deterministic 0-100 stability score decomposed into five sub-scores:
- Compatibility (0-35): Pairwise interaction rules (hard/soft) from curated knowledge base
- pH Suitability (0-25): Ingredient optimal pH ranges vs. target pH
- Emulsion Balance (0-20): Oil phase HLB requirements vs. emulsifier system
- Formula Integrity (0-10): Percentage math, duplicates
- System Completeness (0-10): Preservative presence, ingredient count, solvent

This is a rule-based system with no learned parameters. It serves as the interpretability baseline: any ML model should beat it on accuracy while ideally matching its transparency.

**Expected performance:** Moderate on skincare/personal-care (where rules are densest), weak on pharma/food (thin rule coverage). The system itself warns when coverage is low.

### Baseline 2: XGBoost + Molecular Descriptors
- **Features:** For each ingredient, compute RDKit 2D descriptors (MolLogP, TPSA, MW, HBD, HBA, rotatable bonds). Aggregate across the formulation using weighted mean/min/max by percentage. Append formula-level features (ingredient count, total surfactant %, total oil %, target pH).
- **Model:** XGBoost classifier (stability) or regressor (shelf life). Hyperparameters via 5-fold CV on train split.
- **Expected performance:** Strong on random split, weaker on leave-ingredients-out.

### Baseline 3: LLM Zero-Shot
- Prompt an LLM (e.g., Claude, GPT-4) with the full formulation and ask: "Will this formulation be stable after 3 months at 40C/75% RH? Answer stable or unstable with confidence 0-100."
- No fine-tuning, no in-context examples beyond task description.
- **Expected performance:** Competitive on well-known ingredient interactions, poor calibration, likely overconfident.

### Baseline 4: LLM Few-Shot
- Same as Baseline 3 but with 5 labeled examples in context (balanced: 2-3 stable, 2-3 unstable).
- Measures whether in-context learning improves over zero-shot.

## 8. How to Contribute Data

FormulaBench is designed for community contribution. We need formulation-outcome pairs from:
- Published stability studies (with DOI)
- Industry datasets (anonymized if needed -- ingredient names + percentages + outcome is sufficient)
- Academic research groups with unpublished data willing to release under CC-BY-4.0

### Contribution process:
1. Format your data as JSONL following the schema in Section 3
2. Include a `metadata.source` and `metadata.license` for each record
3. Run the validation script: `python -m formulabench.validate your_data.jsonl`
4. Submit a pull request to the `data/` directory with a brief description of the dataset

### Minimum requirements per record:
- At least 3 ingredients with INCI names and percentages
- Percentages that sum to 95-105% (water balance is acceptable)
- A binary stability label with description of test conditions
- Source attribution (DOI, textbook citation, or "expert annotation")

### What we will NOT accept:
- Proprietary data without explicit release permission
- Records with only trade names (must have INCI)
- Synthetic/generated labels (LLM predictions do not count as ground truth)

## 9. Relationship to CheMixHub

FormulaBench and CheMixHub address different layers of the same problem:

**CheMixHub** answers: "What are the physical properties of this mixture?"
**FormulaBench** answers: "Will this formulation be stable on a shelf?"

They are complementary. A model could use CheMixHub-trained property predictors as *features* for FormulaBench tasks. For example: predicted mixture viscosity, predicted phase behavior, and predicted solubility parameters could all feed into a stability classifier.

We reference CheMixHub (arXiv 2506.12231) as prior art and encourage researchers to explore transfer learning between the two benchmarks. The key difference is that FormulaBench operates at formulation scale (5-30 ingredients, real products) with practical outcomes (stable/unstable, shelf life), while CheMixHub operates at mixture scale (2-3 components) with physical properties.

## License

FormulaBench metadata and tooling: Apache-2.0 (matching OpenMix).
Individual datasets carry their own licenses (minimum CC-BY-4.0).
