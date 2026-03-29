<div align="center">

# OpenMix

### The Open-Source Framework for Computational Formulation Science

*From physics observation to autonomous mixture design — the missing infrastructure between single-molecule tools and real-world formulation.*

**RDKit is for molecules. OpenMix is for mixtures.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CI](https://github.com/vijayvkrishnan/openmix/actions/workflows/ci.yml/badge.svg)](https://github.com/vijayvkrishnan/openmix/actions)
[![Tests](https://img.shields.io/badge/Tests-133%20passed-brightgreen.svg)](tests/)
[![Rules](https://img.shields.io/badge/Knowledge-85%20rules%2C%206%20domains-orange.svg)](src/openmix/knowledge/data/)

[Observe](#observe-a-formulation) · [Two Modes](#two-modes) · [Autonomous Experiments](#autonomous-experiments) · [Architecture](#architecture) · [Knowledge Base](#contributing-knowledge) · [Roadmap](#roadmap) · [Citation](#citation)

</div>

---

## The Problem

Chemistry has excellent open-source tools for individual molecules — [RDKit](https://www.rdkit.org/), [DeepChem](https://deepchem.io/), [ChemProp](https://github.com/chemprop/chemprop). But the moment you ask *"what happens when I mix these ingredients together?"* — the tooling disappears.

Every formulation scientist — cosmetics, pharma, food, supplements — relies on institutional memory, expensive trial-and-error, and proprietary databases locked inside large corporations. There is no open-source framework for computational mixture science.

**OpenMix changes that.**

OpenMix resolves any ingredient to its molecular identity and observes formulations through physics, out of the box, no training data required. The knowledge base catches dangerous interactions. The experiment runner automates exploration. And because evaluation is pluggable, proprietary stability data becomes a force multiplier. The more diverse your ingredient space, the more the molecular features differentiate.

---

## Quick Start

```bash
pip install openmix
```

<a id="observe-a-formulation"></a>

### Observe a formulation

The physics observation engine resolves each ingredient to its molecular identity (INCI → SMILES → LogP, MW, charge, HLB), then reports what it **sees**, what it **expected**, and where they **disagree**. No arbitrary scores — structured physics observations.

```python
from openmix import Formula, observe

cream = Formula(
    name="Retinol Night Cream",
    ingredients=[
        ("Water", 60.0),
        ("Retinol", 2.0),
        ("Squalane", 15.0),
        ("Cetyl Alcohol", 5.0),
        ("Glycerin", 8.0),
        ("Niacinamide", 5.0),
        ("Ascorbic Acid", 5.0),
    ],
    target_ph=5.5,
    category="skincare",
)

print(observe(cream))
```

```
Physics Observation (engineering): Retinol Night Cream
Resolved: 100% of ingredients

Violations (0 hard, 2 soft):
  [SOFT (conf 0.5)] NIACINAMIDE + ASCORBIC ACID
    At low pH and high concentrations, niacinamide may convert to nicotinic acid.
    Widely debated. Many commercial products combine these successfully.
  [SOFT (conf 0.7)] RETINOL + ASCORBIC ACID
    Retinol is unstable in the acidic conditions required for L-Ascorbic Acid.

MOLECULAR:
  [!] Squalane: LogP 14.7 at 15.0% — hydrophobic
      Expected: Hydrophobic ingredients in aqueous systems need solubilization
      LogP 14.7 suggests poor water solubility. At 15.0%, ensure adequate emulsifier.
  [?] Cetyl Alcohol: LogP 7.3 at 5.0% — hydrophobic

STRUCTURAL:
  [ ] formula: Total: 100.0%
  [!] formula: Water-based formula without detected preservative

PHASE:
  [?] formula: Hydrophobic phase: 22.0% — Retinol (2.0%), Squalane (15.0%),
      Cetyl Alcohol (5.0%)

Concern count: 3.2 (lower = better, 0 = no concerns)
```

The engine caught: two ingredient interactions with confidence scores and literature context, a hydrophobic solubility concern from molecular LogP, a missing preservative system, and a 22% oil phase needing emulsification. Each observation reports what was seen, what was expected, and whether they agree.

### Validate interactions

```python
from openmix import Formula, validate

report = validate(Formula(
    ingredients=[
        ("Sodium Hypochlorite", 5.0),
        ("Ammonia", 3.0),
        ("Water", 92.0),
    ],
    category="home_care",
))
print(report)
```

```
OpenMix Validation Report
Score: 75/100  |  1 errors, 0 warnings, 0 info

  [X] SODIUM HYPOCHLORITE + AMMONIA
      Produces toxic Chloramine gas (NH2Cl). Leading cause of household
      chemical poisoning. Never combine.
```

### CLI

```bash
openmix observe formula.yaml                   # Physics observations
openmix observe formula.yaml --mode discovery  # Discovery mode
openmix validate formula.yaml                  # Rule-based validation
openmix run "Design a stable vitamin C serum"  # Autonomous experiment (needs API key)
openmix demo                                   # Try it now (no API key)
```

---

<a id="two-modes"></a>

## Two Modes: One Engine

The same physics observation engine serves two fundamentally different goals. Same observations, different interpretation.

### Engineering Mode (default)

*"Build me a stable formula."*

Discrepancies are problems to fix. Soft violations are risks to mitigate. The optimization target is **zero concerns**.

```python
obs = observe(formula, mode="engineering")
# Concern count: 3.4 (lower = better, 0 = no concerns)
```

The autonomous experiment runner uses engineering mode to iterate toward stable, manufacturable formulations. Every concern the physics engine flags is something the LLM resolves on the next iteration.

### Discovery Mode

*"Why does this work despite the rules saying it shouldn't?"*

Hard violations still count — safety is non-negotiable. But soft violations become **signals** and low-confidence discrepancies become **knowledge gaps** worth investigating.

```python
obs = observe(formula, mode="discovery")
# Hard violations: 0  |  Signals: 2  |  Knowledge gaps: 1

obs.signals        # Soft violations — interesting interactions to explore
obs.discoveries    # Low-confidence discrepancies — where expectations may be wrong
```

Every major scientific breakthrough follows the same pattern: expectation existed → something violated it → someone **noticed** → they **investigated** instead of dismissing. Penicillin. CRISPR. GLP-1. Discovery mode is the noticing.

| | Engineering | Discovery |
|---|---|---|
| **Goal** | Zero concerns | Investigate surprises |
| **Hard violations** | Block (safety) | Block (safety) |
| **Soft violations** | Penalize | Surface as `signals` |
| **Physics discrepancies** | Fix them | Investigate them |
| **Low-confidence expectations** | Flag as uncertain | Highlight as `discoveries` |
| **Best for** | Product development | Research, novel combinations |

---

## Ingredient Resolution

OpenMix resolves any INCI ingredient name to its molecular identity and physicochemical properties through a three-tier lookup:

```
INCI Name → Seed Cache (2,400+ ingredients, ships with package)
          → User Cache (~/.openmix/, grows over time)
          → PubChem API (runtime fallback)
          → RDKit enrichment (optional: LogP, MW, HLB, charge from SMILES)
```

```python
from openmix.resolver import resolve

props = resolve("Niacinamide")
# ResolvedIngredient(smiles='c1ccc(c(c1)C(=O)N)N', log_p=-0.35,
#                    molecular_weight=122.12, charge_type='nonionic', ...)
```

This is how the observation engine knows that Retinol (LogP 5.7) is hydrophobic and needs solubilization, or that mixing an anionic surfactant with a cationic one will cause precipitation. Not a lookup table — molecular physics.

---

<a id="autonomous-experiments"></a>

## Autonomous Experiments

An LLM agent that explores the formulation space through iterative optimization, guided by physics observations. Not "generate and check" — the agent reads structured observations, analyzes what the physics shows, and converges.

### From natural language

```bash
pip install openmix[agent]
export ANTHROPIC_API_KEY=sk-ant-...

openmix run "Design a stable vitamin C serum under $30/kg"
```

OpenMix plans the experiment from your brief, selects ingredients, and iterates:

```
======================================================================
  OPENMIX EXPERIMENT: vitamin-c-stability
======================================================================
  Pool: 29 ingredients  |  Target: zero concerns  |  Max: 30 iterations

  [ 1] REJECTED — Total is 90.0%, target is 100%
  [ 2] concerns:  0.0  violations:0H/0S  *NEW BEST*

  Converged at iteration 2 — zero concerns.

  BEST FORMULATION
  Concerns: 0.0  |  pH: 3.0  |  12 ingredients
  Violations: 0 hard, 0 soft  |  Resolved: 100%

  Water                                     70.0%  solvent
  Ascorbic Acid                             15.0%  active
  Propanediol                                8.0%  humectant
  Glycerin                                   5.0%  humectant
  Ferulic Acid                               0.5%  antioxidant
  Phenoxyethanol                             0.5%  preservative
  ...
```

### From YAML (for reproducibility)

```yaml
# experiments/vitamin_c_stability.yaml
name: vitamin-c-stability
brief: |
  Find the most stable vitamin C serum formulation. Maximize ascorbic acid
  concentration while maintaining pH 2.5-3.5 and total COGS under $30/kg.

ingredient_pool:
  required:
    - name: Ascorbic Acid
      min_pct: 10.0
      max_pct: 20.0
      function: active
  available:
    - Water
    - Glycerin
    - Niacinamide
    - Ferulic Acid
    - Tocopherol
    - Phenoxyethanol
    # ... full ingredient pool

constraints:
  target_ph: [2.5, 3.5]
  max_ingredients: 12
  total_percentage: 100
  category: skincare

llm:
  provider: anthropic           # or: openai, ollama, together, groq, custom
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY

settings:
  max_iterations: 30
  mode: formulation             # or: discovery
```

```bash
openmix experiment experiments/vitamin_c_stability.yaml --save results.json
```

### The loop

```
Natural Language Brief       "Design a stable vitamin C serum..."
     |
Experiment Planner           LLM generates ingredient pool + constraints
     |
LLM (pluggable)             Anthropic, OpenAI, Ollama, any provider
     |
Constraint Enforcement       Rejects non-compliant formulas before observation
     |
Physics Observation Engine   Resolve ingredients → observe → report discrepancies
     |
Iteration Loop               Propose → Observe → Analyze → Improve → Repeat
     |
Post-Experiment Analysis     Which ingredients, patterns, violations matter
     |
Experiment Log (JSON)        Every trial recorded. Reproducible. Shareable.
```

Every component is pluggable. The framework handles the loop.

### Pluggable evaluation

The experiment runner accepts custom evaluation functions for when you have real data:

```python
from openmix import Experiment
from openmix.scorers import ModelScorer, ManualScorer

# Built-in physics observations (default — works out of the box)
exp = Experiment.from_file("experiment.yaml")

# Trained ML model (domain-specific)
scorer = ModelScorer.load("models/stability.pkl", feature_fn=my_features)
exp = Experiment.from_file("experiment.yaml", evaluate=scorer)

# Real lab feedback (cloud lab, robotic platform, or manual entry)
exp = Experiment.from_file("experiment.yaml", evaluate=ManualScorer())
```

This is how the framework scales from "try it on your laptop" to "run it in a cloud lab."

---

## Validation Modes

Not all formulation is the same. A consumer brand needs strict guardrails. A drug discovery researcher needs freedom to explore.

```python
report = validate(formula, mode="safety")       # Flag everything
report = validate(formula, mode="formulation")   # Real issues only, with mitigations
report = validate(formula, mode="discovery")     # Only block genuinely dangerous reactions
```

| Mode | Hard Rules (toxic gas, carcinogens) | Soft Rules (pH conflicts, absorption) | Use Case |
|------|:---:|:---:|----------|
| **`safety`** | Error | Warning | Consumer products, home care, OTC |
| **`formulation`** | Error | Info (with mitigations) | Professional formulators, trained chemists |
| **`discovery`** | Error | Ignored | Drug discovery, research, novel combinations |

Hard rules **always fire in every mode**. Bleach + ammonia produces toxic gas regardless of intent. But debated interactions like Niacinamide + Vitamin C? In discovery mode, you're free to explore.

### Coverage honesty

When OpenMix validates a formula in a domain with thin rule coverage, it says so:

```
  [!] COVERAGE WARNING: Category 'pharma' has 15 dedicated rules.
      Consider additional domain-specific review. Contributions welcome.
```

A 100/100 in a domain with limited rules is misleading. We'd rather be honest about what we know and don't know.

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
```

</details>

<details>
<summary><b>Same formula, three modes</b></summary>

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

## Heuristic Scoring

In addition to the observation engine, OpenMix includes a deterministic stability score — a decomposed objective function that tells you exactly what to improve.

| Sub-Score | Weight | What It Measures |
|-----------|--------|------------------|
| **Compatibility** | /35 | No dangerous interactions (hard rule = instant zero) |
| **pH Suitability** | /25 | All pH-sensitive ingredients in their optimal range |
| **Emulsion Balance** | /20 | Oil phase HLB matched by emulsifier system |
| **Formula Integrity** | /10 | Percentages sum to 100%, no duplicates |
| **System Completeness** | /10 | Preservative present, reasonable ingredient count |

```python
from openmix import Formula, score

s = score(my_formula)
print(f"Total: {s.total}/100")
print(f"Weakest area: pH ({s.ph_suitability}/25)")
```

This is a heuristic model. The observation engine provides richer physics-based feedback. Both are available; the experiment runner uses observations by default.

---

<a id="architecture"></a>

## Architecture

OpenMix is built in layers. Each is independently useful. Together, they form the infrastructure for autonomous formulation science.

```
+---------------------------------------------------------------------+
|                                                                     |
|   Layer 4: EXPERIMENT          Autonomous Formulation Agent         |
|   +-------------------------------------------------------------+  |
|   |  LLM proposes -> Observe -> Constrain -> Analyze -> Iterate |  |
|   |  Active learning loop . Cloud lab integration . Safety       |  |
|   +-------------------------------------------------------------+  |
|                              ^                                      |
|   Layer 3: OPTIMIZE          |  Multi-Objective Design              |
|   +--------------------------+----------------------------------+   |
|   |  Bayesian optimization . Ingredient substitution            |   |
|   |  Pareto frontier (cost vs stability vs efficacy)            |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 2: PREDICT           |  ML Mixture Properties               |
|   +--------------------------+----------------------------------+   |
|   |  Stability prediction . Phase behavior . Shelf life         |   |
|   |  FormulaBench benchmark . Mixture fingerprints              |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 1.5: OBSERVE    <====+====  CURRENT (v0.2)                  |
|   +--------------------------+----------------------------------+   |
|   |  Physics observation engine . Molecular resolution          |   |
|   |  Dual modes (engineering / discovery) . Concern tracking    |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 1: VALIDATE          |  Rule-Based Intelligence             |
|   +--------------------------+----------------------------------+   |
|   |  85 rules (32 hard + 53 soft) . 3 validation modes          |   |
|   |  Conditional (pH, concentration) . Coverage honesty         |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 0: FOUNDATION        |  Schema & Bridges                    |
|   +--------------------------+----------------------------------+   |
|   |  Formula representation . INCI->SMILES resolver . RDKit     |   |
|   |  Community knowledge base (YAML -- no code to contribute)   |   |
|   +-------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

---

## Domains

| Domain | Example Checks | Rules |
|--------|---------------|-------|
| **Skincare & Cosmetics** | Retinol + AHA pH conflict, BPO + antioxidant oxidation, copper peptide Fenton reaction, emulsion HLB | 21 |
| **Pharma** | Lactose-amine Maillard, MgSt ester hydrolysis, PVP peroxide, gelatin crosslinking | 15 |
| **Supplements** | Calcium/Iron absorption competition, probiotics + preservatives, B12 degradation | 13 |
| **Food Science** | Sulfite-thiamine destruction, sorbic acid + nitrite mutagenicity, phytate-mineral chelation | 10 |
| **Home Care** | Bleach + acid/ammonia toxic gas, cationic + anionic surfactant precipitation | 21 |
| **Beverages** | Protein precipitation at low pH, benzene formation, tannin-iron complexes | 5 |

---

## What Makes OpenMix Different

| Capability | RDKit | DeepChem | AI Scientist | Proprietary | **OpenMix** |
|:-----------|:-----:|:--------:|:------------:|:-----------:|:-----------:|
| Single-molecule properties | Yes | Yes | N/A | Yes | Via RDKit |
| **Mixture/formulation analysis** | No | No | No | Closed | **Open** |
| **Physics observation engine** | N/A | N/A | N/A | No | **Dual-mode** |
| **Molecular resolution (INCI→SMILES)** | N/A | N/A | N/A | Closed | **Open (2,400+ seed + PubChem)** |
| **Autonomous experiment loop** | No | No | ML only | No | **Chemistry** |
| Pluggable evaluation (model/lab) | N/A | N/A | No | No | **Yes** |
| Ingredient interaction rules | No | No | No | Partial | **85 rules, 6 domains** |
| Validation modes (safety/discovery) | N/A | N/A | N/A | No | **3 modes** |
| Coverage honesty | N/A | N/A | N/A | No | **Warns on thin domains** |
| Community-contributable knowledge | N/A | N/A | N/A | No | **YAML, no code** |
| Bring your own LLM | N/A | N/A | Partial | No | **Any provider** |

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

## Bring Your Own LLM

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

<a id="contributing-knowledge"></a>

## Contributing Knowledge

OpenMix knowledge lives in YAML. **Chemists contribute domain expertise without writing code.**

Two rule types:

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

Every rule has a **confidence score**, a **source citation**, and optional **conditions** and **mitigations**. This isn't a binary lookup table — it's a nuanced knowledge base.

**We especially need:** pharmaceutical excipient compatibility, food science interactions, drug delivery, and regional regulatory constraints.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

<a id="roadmap"></a>

## Roadmap

| Milestone | Target | What Ships |
|-----------|--------|------------|
| **v0.1** | March 2026 | Formula schema, 85 rules, heuristic scoring, autonomous experiment loop, 3 validation modes, FormulaBench baselines, CLI |
| **v0.2** | March 2026 | **Physics observation engine, ingredient resolver (INCI→SMILES), dual modes (engineering/discovery), molecular scorer** |
| **v0.3** | Q2 2026 | Expanded knowledge base (200+ rules), CI hardening, more benchmark datasets |
| **v0.5** | Q3 2026 | Trained stability models, pre-trained models on HuggingFace |
| **v1.0** | Q4 2026 | Mixture property prediction (stability, phase, shelf life), MCP server for AI agent integration |
| **v2.0** | 2027 | Multi-objective optimization, ingredient substitution, Bayesian search |
| **v3.0** | 2028+ | Autonomous formulation agent with cloud lab integration and active learning |

---

## Project Structure

```
openmix/
  src/openmix/
    observe.py              # Physics observation engine (engineering / discovery)
    resolver/               # INCI → SMILES → molecular properties
      resolve.py            #   Three-tier resolution (seed → cache → PubChem)
      cache.py              #   Local cache management
      pubchem.py            #   PubChem API integration
      seed_ingredients.json #   Bundled ingredient data (2,400+ ingredients)
    experiment.py           # Autonomous experiment runner (YAML or natural language)
    llm.py                  # Multi-provider LLM abstraction
    constraints.py          # Programmatic constraint enforcement
    analysis.py             # Post-experiment insight extraction
    validate.py             # Rule-based validation (3 modes)
    score.py                # Heuristic stability scoring (5 sub-scores)
    discover.py             # Hypothesis-driven rule discovery
    matching.py             # Ingredient name matching
    schema.py               # Formula, Ingredient, ValidationReport
    molecular.py            # RDKit integration (optional)
    scorers/                # Pluggable evaluation functions
      molecular.py          #   Physics-informed scorer (uses resolver)
      model.py              #   Trained ML model scorer
      lab.py                #   Lab feedback (cloud lab, manual entry)
      base.py               #   Scorer interface, composite
    benchmarks/             # FormulaBench datasets + features
    knowledge/data/         # YAML rules (85 interactions, 42 HLB values)
    cli/                    # CLI: observe, run, validate, score, demo, info
  experiments/              # YAML experiment definitions
  tests/                    # 133 tests
  docs/                     # FormulaBench spec
```

---

<a id="citation"></a>

## Citation

If you use OpenMix in research, please cite:

```bibtex
@software{krishnan2026openmix,
  author = {Krishnan, Vijay},
  title = {OpenMix: An Open-Source Framework for Computational Formulation Science},
  year = {2026},
  url = {https://github.com/vijayvkrishnan/openmix},
  version = {0.2.0},
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
