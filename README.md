<div align="center">

# OpenMix

### Autonomous Formulation Science

*Define an experiment. Run it. Wake up to results.*

**The autoresearch pattern, applied to chemistry.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CI](https://github.com/vijayvkrishnan/openmix/actions/workflows/ci.yml/badge.svg)](https://github.com/vijayvkrishnan/openmix/actions)
[![Tests](https://img.shields.io/badge/Tests-60%20passed-brightgreen.svg)](tests/)
[![Rules](https://img.shields.io/badge/Knowledge-85%20rules-orange.svg)](src/openmix/knowledge/data/)

[How It Works](#how-it-works) · [Run an Experiment](#run-an-experiment) · [Plug In Your Own Scorer](#pluggable-evaluation) · [Architecture](#architecture) · [Roadmap](#roadmap) · [Citation](#citation)

</div>

---

## The Problem

Chemistry has open-source tools for individual molecules (RDKit, DeepChem, ChemProp). But formulation science — combining ingredients into stable, effective mixtures — has no computational infrastructure. Every formulation scientist relies on institutional memory, expensive trial-and-error, and proprietary databases.

Meanwhile, autonomous research frameworks (AI Scientist, autoresearch) have transformed ML experimentation. Nobody has applied that pattern to chemistry.

**OpenMix is the autoresearch framework for formulation science.**

---

<a id="how-it-works"></a>

## Quick Start

### 1. Install

```bash
pip install openmix
```

### 2. Try it (no API key needed)

```bash
openmix demo
```

```
  1. Validate: Catches ingredient interactions
  [X] BENZOYL PEROXIDE + RETINOL — oxidizes and deactivates on contact
  [X] COPPER TRIPEPTIDE-1 + ASCORBIC ACID — Fenton reaction, free radicals
  [!] RETINOL + GLYCOLIC ACID — pH conflict, irritation synergy

  2. Score: Quantitative stability prediction
  Simple Moisturizer: 94.0/100
    Compatibility: 35.0/35  pH: 25.0/25  HLB: 14.0/20  ...
```

### 3. Run an autonomous experiment (needs API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
openmix run "Design a gentle sulfate-free baby shampoo that won't irritate eyes"
```

OpenMix plans the experiment from your brief, selects appropriate ingredients, and iterates autonomously:

```
  Planning experiment from brief...
  Plan: gentle-sulfate-free-baby-shampoo | 4 required + 25 available ingredients

  [ 1] REJECTED — Glycerin at 0.2% below minimum 2.0%
  [ 2] REJECTED — Total is 96.3%, target is 100%
  [ 3] REJECTED — Glycerin at 1.7% below minimum 2.0%
  [ 4] 100.0/100  *CONVERGED*

  BEST FORMULATION (12 ingredients, pH 6.0)
  Water                               65.0%  solvent
  Cocamidopropyl Betaine              12.0%  surfactant
  Coco-Glucoside                       8.0%  surfactant
  Glycerin                             3.5%  humectant
  Sodium Cocoyl Glutamate              3.0%  surfactant
  ...
```

No YAML files to write. Describe your research question in plain English.

For more control, use a YAML experiment file: `openmix experiment experiment.yaml`

---

## How It Works

```
Natural Language Brief    "Design a stable vitamin C serum..."
     |
Experiment Planner        LLM generates ingredient pool + constraints
     |
LLM (pluggable)           Anthropic, OpenAI, Ollama, any provider
     |
Constraint Enforcement    Rejects non-compliant formulas before scoring
     |
Scorer (pluggable)        Heuristic | Trained Model | Lab Feedback | Manual
     |
Iteration Loop            Propose -> Score -> Analyze -> Improve -> Repeat
     |
Post-Experiment Analysis  Which ingredients, patterns, sub-scores matter
     |
Experiment Log (JSON)     Every trial recorded. Reproducible. Shareable.
```

Every component is pluggable. The framework handles the loop.

---

<a id="run-an-experiment"></a>

## Run an Experiment

```bash
pip install openmix[agent]
export ANTHROPIC_API_KEY=sk-ant-...
```

### 1. Define the experiment

```yaml
# experiments/multivitamin_gummy.yaml

name: multivitamin-gummy
brief: |
  Design a comprehensive multivitamin gummy containing Vitamin C,
  B-vitamins, calcium, iron, and zinc. The challenge: many of these
  nutrients actively degrade or inhibit each other. Find the combination
  that actually works together.

ingredient_pool:
  required:
    - name: Ascorbic Acid
      min_pct: 5.0
      max_pct: 20.0
    - name: Calcium Carbonate
      min_pct: 10.0
      max_pct: 30.0
  available:
    - Ferrous Sulfate
    - Iron Bisglycinate
    - Zinc Oxide
    - Cyanocobalamin
    - Methylcobalamin
    - Thiamine Hydrochloride
    - Folic Acid
    - Vitamin D3
    # ... full ingredient pool

constraints:
  max_ingredients: 18
  total_percentage: 100
  category: supplement

llm:
  provider: anthropic              # or: openai, ollama, together, groq
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY

settings:
  max_iterations: 30
  target_score: 90
  mode: formulation
```

### 2. Run it

```bash
openmix experiment experiments/multivitamin_gummy.yaml --save results.json
```

### 3. Results

```
======================================================================
  OPENMIX EXPERIMENT: multivitamin-gummy
======================================================================
  Pool: 26 ingredients  |  Target: 90/100  |  Max: 30 iterations

  [ 1]  62.8/100  (c:10.8 pH:21.0 H:14.0 i:10.0 s: 7.0)
  [ 2]  54.0/100  (c: 0.0 pH:23.0 H:14.0 i:10.0 s: 7.0)
  [ 3]  66.8/100  (c:18.8 pH:17.0 H:14.0 i:10.0 s: 7.0)
  ...
  [ 9]  80.2/100  (c:24.2 pH:25.0 H:14.0 i:10.0 s: 7.0)  *NEW BEST*
  [10]  58.8/100
  [11]  87.5/100  (c:31.5 pH:25.0 H:14.0 i:10.0 s: 7.0)  *NEW BEST*

  Converged at iteration 11.

  BEST FORMULATION
  Score: 87.5/100  |  pH: 3.2  |  16 ingredients

  INCI Name                                     %  Function
  ---------------------------------------- ------  --------------------
  Glucose Syrup                             28.0%  sweetener/texture
  Gelatin                                   18.0%  gelling agent
  Sucrose                                   15.0%  sweetener
  Calcium Carbonate                         12.0%  mineral supplement
  Ascorbic Acid                              8.0%  vitamin C
  Maltodextrin                               8.0%  bulking agent
  Tocopherol                                 3.0%  vitamin E
  Sodium Citrate                             1.7%  buffer
  Vitamin D3                                 1.5%  fat-soluble vitamin
  Zinc Oxide                                 1.2%  zinc source
  Potassium Sorbate                          1.0%  preservative
  Pantothenic Acid                           0.9%  vitamin B5
  Pyridoxine Hydrochloride                   0.8%  vitamin B6
  Riboflavin                                 0.5%  vitamin B2
  Methylcobalamin                            0.3%  vitamin B12
  Biotin                                     0.1%  vitamin B7
                                           ------
                                    Total  100.0%

  EXPERIMENT ANALYSIS
    Insights:
    1. [strong] Thiamine in 83% of low-scoring formulas vs 0% of top
       — degrades with ascorbic acid in solution
    2. [strong] Methylcobalamin in 100% of top vs 50% of bottom
       — more stable than Cyanocobalamin with Vitamin C
    3. [strong] Compatibility is the key differentiator:
       21.9/35 in top formulas vs 8.8/35 in bottom
```

The agent navigated real nutrient interactions — calcium blocks iron absorption, ascorbic acid degrades B12, folic acid reduces zinc bioavailability. It learned which vitamin forms are compatible (Methylcobalamin over Cyanocobalamin), which ingredients to drop (iron, thiamine, folic acid), and produced a complete, manufacturable formula.

Every trial is recorded in a JSON experiment log for reproducibility.

---

<a id="pluggable-evaluation"></a>

## Pluggable Evaluation

The experiment framework doesn't care where the score comes from. Swap the evaluation function based on what you have:

```python
from openmix import Experiment

# Built-in heuristic (no data needed — works out of the box)
exp = Experiment.from_file("experiment.yaml")

# Trained ML model (domain-specific, learned from data)
from openmix.scorers import ModelScorer
scorer = ModelScorer.load("models/stability.pkl", feature_fn=my_features)
exp = Experiment.from_file("experiment.yaml", evaluate=scorer)

# Real lab feedback (cloud lab, robotic platform, or manual entry)
from openmix.scorers import LabScorer, ManualScorer
exp = Experiment.from_file("experiment.yaml", evaluate=ManualScorer())

# Your own cloud lab
class MyLabScorer(LabScorer):
    def run_experiment(self, formula):
        result = my_lab_api.run_stability_screen(formula)
        return {"stable": result.passed, "shelf_life_months": result.shelf_life}

exp = Experiment.from_file("experiment.yaml", evaluate=MyLabScorer())

# Composite (model where confident, heuristic as fallback)
from openmix.scorers import CompositeScorer
exp = Experiment.from_file("experiment.yaml",
    evaluate=CompositeScorer(primary=model_scorer, fallback=heuristic))
```

This is how the framework scales from "try it on your laptop" to "run it in a cloud lab."

---

## Validation & Scoring

OpenMix also works as a standalone validation and scoring tool:

```python
from openmix import Formula, validate, score

formula = Formula(
    ingredients=[
        ("Sodium Hypochlorite", 5.0),
        ("Ammonia", 3.0),
        ("Water", 92.0),
    ],
    category="home_care",
)

# Qualitative: what's wrong and why
report = validate(formula, mode="safety")
# -> [X] Produces toxic Chloramine gas. Never combine.

# Quantitative: decomposed stability score
stability = score(formula)
# -> 0/100 (compatibility: 0/35 — hard rule violation)
```

### Three Validation Modes

| Mode | Hard Rules | Soft Rules | Use Case |
|------|:---:|:---:|----------|
| **`safety`** | Error | Warning | Consumer products |
| **`formulation`** | Error | Info + mitigations | Professional formulators |
| **`discovery`** | Error | Ignored | Research, novel combinations |

### Knowledge Base

85 interaction rules (32 hard + 53 soft) across 6 domains, each with confidence scores, literature sources, conditions, and mitigations.

| Domain | Rules |
|--------|-------|
| Skincare & Cosmetics | 21 |
| Supplements | 13 |
| Pharma (Maillard, MgSt hydrolysis, PVP peroxide) | 15 |
| Food Science (sulfite-thiamine, phytate-iron) | 10 |
| Beverages | 5 |
| Home Care + Cross-category safety | 21 |

### Contributing Rules

Rules are YAML files — **no code required**. Two types:

```yaml
# HARD — unconditional, dangerous. Always fires in all modes.
- a: SODIUM HYPOCHLORITE
  b: AMMONIA
  type: hard
  severity: error
  mechanism: chemical_reaction
  confidence: 1.0
  source: "CDC NIOSH Pocket Guide"
  category: all
  message: Produces toxic Chloramine gas. Never combine.

# SOFT — conditional, context-dependent. Mode affects severity.
- a: NIACINAMIDE
  b: ASCORBIC ACID
  type: soft
  mechanism: pH_conflict
  confidence: 0.5
  source: "Cosmetics & Toiletries 2005; challenged by Fu et al. 2010"
  conditions:
    ph_below: 3.5
    min_concentration_either: 5.0
  severity_by_mode:
    safety: warning
    formulation: info
    discovery: ignore
  mitigation: "Buffer pH above 5.0, or use SAP instead of L-AA"
  category: skincare
  message: >
    At low pH and high concentrations, niacinamide may convert to
    nicotinic acid. Widely debated. Many products combine these.
```

Every rule has a confidence score, literature source, and optional conditions and mitigations. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

**We especially need:** pharmaceutical excipient compatibility, food science interactions, and regional regulatory constraints.

### More Examples

<details>
<summary><b>Beverage: Catches carcinogen formation</b></summary>

```python
validate(Formula(
    ingredients=[("Ascorbic Acid", 0.5), ("Sodium Benzoate", 0.1),
                 ("Citric Acid", 2.0), ("Water", 97.4)],
    category="beverage",
))
# [X] ERROR: Ascorbic Acid + Sodium Benzoate can form benzene
#     (a Group 1 carcinogen) in acidic conditions with heat or UV.
#     Source: FDA beverage benzene survey 2006
```

</details>

<details>
<summary><b>Pharma: Catches Maillard degradation</b></summary>

```python
validate(Formula(
    ingredients=[("Lactose Monohydrate", 60.0), ("Amlodipine", 5.0),
                 ("Microcrystalline Cellulose", 30.0),
                 ("Magnesium Stearate", 1.0), ("Croscarmellose Sodium", 4.0)],
    category="pharma",
))
# [X] ERROR: Lactose + Amlodipine — Maillard reaction degrades drug.
#     Source: Narang et al. 2012, J Pharm Biomed Anal
#
# [!] COVERAGE WARNING: Category 'pharma' has 15 dedicated rules.
#     Consider additional domain-specific review.
```

</details>

<details>
<summary><b>Validation modes: Same formula, different context</b></summary>

```python
formula = Formula(
    ingredients=[("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0)],
    category="skincare",
)

validate(formula, mode="safety")
# [!] WARNING: Retinol + Glycolic Acid increases irritation at high
#     concentrations. Use in separate products.

validate(formula, mode="formulation")
# [-] INFO: Retinol + Glycolic Acid — irritation risk at high concentrations.
#     Mitigation: Reduce to <0.5% retinol, <5% glycolic.

validate(formula, mode="discovery")
# (no issue — only dangerous reactions flagged in discovery mode)
```

</details>

---

## What Makes OpenMix Different

| Capability | RDKit | DeepChem | AI Scientist | Proprietary | **OpenMix** |
|:-----------|:-----:|:--------:|:------------:|:-----------:|:-----------:|
| Single-molecule properties | Yes | Yes | N/A | Yes | Via RDKit |
| **Mixture/formulation analysis** | No | No | No | Closed | **Open** |
| **Autonomous experiment loop** | No | No | ML only | No | **Chemistry** |
| Pluggable evaluation (model/lab) | N/A | N/A | No | No | **Yes** |
| Ingredient interaction rules | No | No | No | Partial | **85 rules, 6 domains** |
| Validation modes (safety/discovery) | N/A | N/A | N/A | No | **3 modes** |
| Community-contributable knowledge | N/A | N/A | N/A | No | **YAML, no code** |
| Bring your own LLM | N/A | N/A | Partial | No | **Any provider** |

---

<a id="architecture"></a>

## Architecture

```
+---------------------------------------------------------------------+
|                                                                     |
|   EXPERIMENT RUNNER (the autoresearch loop)                         |
|   +-------------------------------------------------------------+  |
|   |  YAML brief -> LLM proposes -> Score -> Constrain -> Iterate |  |
|   |  Post-experiment analysis . Experiment log (JSON)            |  |
|   +-------------------------------------------------------------+  |
|        |               |                |                           |
|   LLM Provider    Scorer (pluggable)    Constraint Enforcer         |
|   - Anthropic     - Heuristic           - Required ingredients      |
|   - OpenAI        - Trained model       - Percentage ranges         |
|   - Ollama        - Lab feedback        - pH, max count             |
|   - Any OAI-      - Manual entry        - Ingredient pool           |
|     compatible    - Composite                                       |
|        |               |                                            |
|   +-------------------------------------------------------------+  |
|   |  VALIDATION + SCORING ENGINE                                 |  |
|   |  85 rules (hard/soft) . 3 modes . Coverage honesty           |  |
|   |  Decomposed stability score (5 sub-scores, 0-100)            |  |
|   +-------------------------------------------------------------+  |
|        |                                                            |
|   +-------------------------------------------------------------+  |
|   |  KNOWLEDGE BASE (YAML — community-contributable)             |  |
|   |  Interaction rules . Oil HLB . Aliases . pH ranges           |  |
|   +-------------------------------------------------------------+  |
|                                                                     |
+---------------------------------------------------------------------+
```

---

## Bring Your Own LLM

Configure in the experiment YAML:

```yaml
# Anthropic
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY

# OpenAI
llm:
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

# Local model via Ollama (free, no API key)
llm:
  provider: ollama
  model: llama3.1

# Any OpenAI-compatible endpoint (Together, Groq, vLLM, LM Studio)
llm:
  provider: custom
  model: meta-llama/Llama-3.1-70B
  base_url: https://api.together.xyz/v1
  api_key_env: TOGETHER_API_KEY
```

---

## FormulaBench

A benchmark for formulation property prediction, complementary to [CheMixHub](https://arxiv.org/html/2506.12231v1) (which covers thermophysical mixture properties). CheMixHub answers "what are the physical properties of this mixture?" — FormulaBench answers "will this formulation work?"

| Dataset | Domain | Records | Task | Metric | Baseline |
|---------|--------|---------|------|--------|----------|
| **Shampoo Stability** | Personal care | 812 | Binary classification | AUROC | XGBoost: 0.85 |
| **Pharma Solubility** | Drug delivery | 251 | Regression (mg/mL) | MAE | XGBoost: 2.30 |

- **Leave-ingredients-out evaluation**: domain features improve generalization to unseen ingredients
- **4 planned tasks**: stability classification, compatibility scoring, shelf life regression, failure mode prediction
- Data sources: [Velho et al. 2024 (Nature Sci Data)](https://doi.org/10.1038/s41597-024-03573-w), [CheMixHub (Rajaonson et al. 2025)](https://arxiv.org/abs/2506.12231)

See [docs/formulabench-spec.md](docs/formulabench-spec.md) for the full specification.

---

<a id="roadmap"></a>

## Roadmap

| Milestone | Target | Status |
|-----------|--------|--------|
| **v0.1** — Experiment framework, validation, scoring | **Now** | Shipped |
| **v0.2** — FormulaBench baselines, more rules, CI | Q2 2026 | In progress |
| **v0.3** — INCI-to-SMILES bridge, molecular scoring | Q3 2026 | |
| **v0.5** — Trained stability models, HuggingFace | Q4 2026 | |
| **v1.0** — Mixture property prediction | H1 2027 | |
| **v2.0** — Multi-objective optimization, Bayesian search | 2027 | |
| **v3.0** — Cloud lab integration, active learning | 2028+ | |

---

## Project Structure

```
openmix/
  src/openmix/
    experiment.py         # Autonomous experiment runner (YAML or natural language)
    llm.py                # Multi-provider LLM abstraction
    constraints.py        # Programmatic constraint enforcement
    analysis.py           # Post-experiment insight extraction
    score.py              # Heuristic stability scoring
    validate.py           # Rule-based validation (3 modes)
    discover.py           # Hypothesis-driven rule discovery (experimental)
    matching.py           # Ingredient name matching
    schema.py             # Formula, Ingredient, ValidationReport
    molecular.py          # RDKit integration (optional)
    scorers/              # Pluggable evaluation functions
      model.py            #   Trained ML model scorer
      lab.py              #   Lab feedback (cloud lab, manual entry)
      base.py             #   Scorer interface, composite
    benchmarks/           # FormulaBench datasets + features
    knowledge/data/       # YAML rules (85 interactions, 42 HLB values)
    cli/                  # CLI: run, demo, experiment, validate, score, info
  experiments/            # YAML experiment definitions
  tests/                  # 60 tests
  docs/                   # FormulaBench spec
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The highest-impact contribution is **domain knowledge** — interaction rules for pharma, food, materials science. No code needed, just YAML.

---

<a id="citation"></a>

## Citation

```bibtex
@software{krishnan2026openmix,
  author = {Krishnan, Vijay},
  title = {OpenMix: Autonomous Formulation Science},
  year = {2026},
  url = {https://github.com/vijayvkrishnan/openmix},
  version = {0.1.0},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0

---

<div align="center">

*The autoresearch pattern, applied to chemistry.*

*Define an experiment. Run it. Wake up to results.*

</div>
