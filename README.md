<div align="center">

# OpenMix

### The Open-Source Framework for Autonomous Formulation Science

*From ingredient validation to autonomous mixture optimization — the missing infrastructure between single-molecule tools and real-world formulation.*

**RDKit is for molecules. OpenMix is for mixtures.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-31%20passed-brightgreen.svg)](tests/)
[![Rules](https://img.shields.io/badge/Rules-21%20hard%20%2B%2039%20soft-orange.svg)](src/openmix/knowledge/data/)
[![Domains](https://img.shields.io/badge/Domains-Skincare%20%7C%20Pharma%20%7C%20Food%20%7C%20Supplements%20%7C%20Home%20Care-purple.svg)](#domains)

[Agent Demo](#agent-demo) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Validation Modes](#validation-modes) · [Roadmap](#roadmap) · [Contributing](#contributing-knowledge) · [Citation](#citation)

</div>

---

## The Problem

Chemistry has excellent open-source tools for individual molecules — [RDKit](https://www.rdkit.org/), [DeepChem](https://deepchem.io/), [ChemProp](https://github.com/chemprop/chemprop). But the moment you ask *"what happens when I mix these ingredients together?"* — the tooling disappears.

Every formulation scientist — cosmetics, pharma, food, supplements — relies on institutional memory, expensive trial-and-error, and proprietary databases locked inside large corporations. There is no open-source framework for computational mixture science.

**OpenMix changes that.**

---

<a id="agent-demo"></a>

## Agent Demo

An AI agent that explores the formulation space through iterative optimization. Not "generate and check" — the agent has a quantitative objective (stability score), analyzes what's working, and converges.

```bash
pip install openmix[agent]
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo_agent.py
```

The default brief deliberately asks for **Retinol + Vitamin C + Copper Peptides** — three ingredients that are chemically incompatible. The agent has to discover this and find alternatives.

```
======================================================================
  OPENMIX FORMULATION AGENT
======================================================================
  Brief: Design a high-performance anti-aging serum with Retinol and
  Vitamin C (L-Ascorbic Acid). Include copper peptides for collagen
  synthesis. Target pH 3.5. Complete preservative system.
  Target: 90/100  |  Max iterations: 5

  [Iter 1] Proposing formulation...
           Score: 50.0/100  (compat:0 pH:17 HLB:14 integ:10 complete:9)
           - HARD: COPPER TRIPEPTIDE-1 + L-ASCORBIC ACID (Fenton reaction)
           - HARD: COPPER TRIPEPTIDE-1 + L-ASCORBIC ACID (GHK-Cu disruption)
           - SOFT: RETINOL + L-ASCORBIC ACID (pH conflict)
           - SOFT: EDTA + COPPER TRIPEPTIDE-1 (chelation)

  [Iter 2] Proposing formulation...
           Strategy: Removed copper tripeptide, swapped to Palmitoyl
           Tripeptide-1. Used Retinyl Palmitate (more pH-stable).
           Score: 85.5/100  (compat:32 pH:23 HLB:14 integ:7 complete:9)
           - Percentages sum to 102%

  [Iter 3] Proposing formulation...
           Strategy: Fixed percentages, improved emulsifier system.
           Score: 88.5/100  (compat:32 pH:23 HLB:14 integ:10 complete:9)

  [Iter 4] Proposing formulation...
           Strategy: Replaced sodium hyaluronate with betaine to
           eliminate soft pH penalty while keeping humectant properties.
           Score: 93.0/100  (compat:35 pH:25 HLB:14 integ:10 complete:9)

  Converged at iteration 4 (score 93.0 >= 90)

  Score Trajectory:
    Iter 1:  50.0 |#########################
    Iter 2:  85.5 |##########################################
    Iter 3:  88.5 |############################################
    Iter 4:  93.0 |############################################## <-- best
```

Four iterations. The agent caught Fenton reaction free radical generation, pH incompatibilities, and percentage errors — then systematically improved each sub-score until converging at 93/100.

<details>
<summary><b>Full output with final formula (click to expand)</b></summary>

```
======================================================================
  BEST FORMULA
======================================================================
  Stabilized Anti-Aging Complex Serum
  Category: skincare  |  pH: 3.5  |  Score: 93.0/100

  INCI Name                                     %  Phase   Function
  ---------------------------------------- ------  ------  --------------------
  Aqua                                      62.5%  A       solvent
  L-Ascorbic Acid                           15.0%  A       antioxidant
  Glycerin                                   5.0%  A       humectant
  Propanediol                                3.0%  A       solvent
  Betaine                                    2.0%  A       humectant
  Squalane                                   2.0%  B       emollient
  PEG-40 Hydrogenated Castor Oil             2.0%  C       emulsifier
  Citric Acid                                2.0%  C       pH adjuster
  Poloxamer 188                              1.5%  C       emulsifier
  Retinyl Palmitate                          1.0%  B       anti-aging
  Caprylic/Capric Triglyceride               1.0%  B       emollient
  Sodium Citrate                             1.0%  C       pH buffer
  Phenoxyethanol                             1.0%  C       preservative
  Xanthan Gum                                0.7%  C       thickener
  Palmitoyl Tripeptide-1                     0.5%  C       peptide
  Tocopherol                                 0.3%  B       antioxidant
  Ethylhexylglycerin                         0.3%  C       preservative booster
  Sodium Phytate                             0.2%  C       chelating agent
                                           ------
                                     Total 100.0%

  Validation Score: 93.0/100  |  CONVERGED
======================================================================
```

</details>

---

<a id="quick-start"></a>

## Quick Start

```bash
pip install openmix
```

### Score a formulation

The stability score is a quantitative objective — decomposed into sub-scores so you can see exactly what to improve.

```python
from openmix import Formula, score

result = score(Formula(
    name="Vitamin C Serum",
    ingredients=[
        ("Ascorbic Acid", 15.0),
        ("Niacinamide", 5.0),
        ("Water", 65.0),
        ("Glycerin", 8.0),
        ("Phenoxyethanol", 1.0),
    ],
    target_ph=3.2,
))
print(result)
```

```
Stability Score: 72.5/100
  Compatibility:      32.5/35
  pH Suitability:     17.0/25
  Emulsion Balance:   20.0/20
  Formula Integrity:    7.0/10
  System Completeness:  7.0/10
  Penalties:
    - SOFT (0.5): NIACINAMIDE + ASCORBIC ACID
    - pH: Niacinamide needs 5.0-7.0, formula is 3.2
    - Percentages sum to 94.0%
```

### Validate a formulation

Validation gives qualitative feedback — what's wrong, why, and how to fix it.

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
OpenMix Validation Report: None
Score: 75/100  |  1 errors, 0 warnings, 0 info

  [X] SODIUM HYPOCHLORITE + AMMONIA
      Produces toxic Chloramine gas (NH2Cl). Leading cause of household
      chemical poisoning. Never combine.
```

### CLI

```bash
openmix validate formula.yaml        # Validate from file
openmix info                          # Knowledge base stats
```

---

<a id="architecture"></a>

## Architecture

OpenMix is built in layers. Each is independently useful. Together, they form the infrastructure for autonomous formulation.

```
+---------------------------------------------------------------------+
|                                                                     |
|   Layer 4: EXPERIMENT          Autonomous Formulation Agent         |
|   +-------------------------------------------------------------+  |
|   |  LLM Reasoning -> Validate -> Predict -> Lab API -> Analyze |  |
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
|   Layer 1.5: SCORE      <====+====  CURRENT (v0.1)                  |
|   +--------------------------+----------------------------------+   |
|   |  Quantitative stability scoring . Decomposed sub-scores     |   |
|   |  Deterministic objective function for agent optimization    |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 1: VALIDATE          |  Rule-Based Intelligence             |
|   +--------------------------+----------------------------------+   |
|   |  21 hard + 39 soft rules . Conditional (pH, concentration)  |   |
|   |  3 validation modes (safety/formulation/discovery)          |   |
|   +-------------------------------------------------------------+   |
|                              ^                                      |
|   Layer 0: FOUNDATION        |  Schema & Bridges                    |
|   +--------------------------+----------------------------------+   |
|   |  Formula representation . INCI->SMILES bridge . RDKit       |   |
|   |  Community knowledge base (YAML -- no code to contribute)   |   |
|   +-------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

---

<a id="validation-modes"></a>

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

### Coverage Honesty

When OpenMix validates a formula in a domain with thin rule coverage, it says so:

```
  [!] COVERAGE WARNING: Category 'pharma' has 0 dedicated rules
      (21 total including cross-category). This score reflects only
      available checks. Do not rely solely on this score for pharma
      product development. Contributions welcome.
```

A 100/100 in a domain with zero rules is meaningless. We'd rather be honest.

---

## Stability Scoring

The scoring function is what makes the agent work. It's a **deterministic, decomposed objective function** — the same formula always gets the same score, and the sub-scores tell you exactly what to improve.

| Sub-Score | Weight | What It Measures |
|-----------|--------|------------------|
| **Compatibility** | /35 | No dangerous interactions (hard rules = instant zero) |
| **pH Suitability** | /25 | All pH-sensitive ingredients in their optimal range |
| **Emulsion Balance** | /20 | Oil phase HLB matched by emulsifier system |
| **Formula Integrity** | /10 | Percentages sum to 100%, no duplicates |
| **System Completeness** | /10 | Preservative present, reasonable ingredient count |

```python
from openmix import Formula, score

s = score(my_formula)
print(f"Total: {s.total}/100")
print(f"Weakest area: pH ({s.ph_suitability}/25)")  # Agent focuses here next
```

This is a heuristic model (Layer 1.5). Layer 2 will add ML-based prediction. But even heuristics give the agent a gradient to optimize against — that's the difference between "checking" and "discovering."

---

<a id="domains"></a>

## Domains

| Domain | Example Checks | Rules |
|--------|---------------|-------|
| **Skincare & Cosmetics** | Retinol + AHA pH conflict, BPO + antioxidant oxidation, copper peptide + Fenton reaction, emulsion HLB | 29 |
| **Supplements** | Calcium/Iron absorption competition, probiotics + preservatives, B12 degradation | 15 |
| **Beverages** | Protein precipitation at low pH, benzene formation, tannin-iron complexes | 7 |
| **Home Care** | Bleach + acid/ammonia toxic gas, cationic + anionic surfactant precipitation | 8 |
| **Pharmaceutics** | Cross-category safety checks | *Accepting contributions* |
| **Food Science** | Cross-category safety checks | *Accepting contributions* |

---

## What Makes OpenMix Different

| Capability | RDKit | DeepChem | ChemProp | Proprietary | **OpenMix** |
|:-----------|:-----:|:--------:|:--------:|:-----------:|:-----------:|
| Single-molecule properties | Yes | Yes | Yes | Yes | Via RDKit |
| **Mixture analysis** | No | No | No | Closed | **Open** |
| **Quantitative scoring** | No | No | No | Some | **Decomposed, deterministic** |
| **Iterative agent optimization** | No | No | No | No | **Yes** |
| Conditional rules (pH, concentration) | No | No | No | Some | **Yes, with confidence** |
| Validation modes (safety/discovery) | No | No | No | No | **3 modes** |
| Coverage honesty | N/A | N/A | N/A | No | **Warns on thin domains** |
| Community-contributable | N/A | N/A | N/A | No | **YAML, no code** |
| Mixture property prediction (ML) | No | No | No | Proprietary | Planned (Layer 2) |
| Autonomous experimentation | No | No | No | No | Planned (Layer 4) |

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
  source: "CDC NIOSH documentation"
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

**We especially need:** pharmaceutical excipient compatibility, food science interactions, drug delivery, regional regulatory constraints.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Optional: RDKit

```bash
pip install openmix[rdkit]
```

```python
from openmix.molecular import compute_properties, compute_hlb_griffin

props = compute_properties("OCC(O)CO")  # Glycerol
# {'molecular_weight': 92.05, 'log_p': -1.76, 'hbd': 3, 'hba': 3, ...}
```

RDKit is optional. All validation, scoring, and the agent work without it.

---

<a id="roadmap"></a>

## Roadmap

| Milestone | Target | What Ships |
|-----------|--------|------------|
| **v0.1** | **Now** | Formula schema, 60 rules (hard/soft), stability scoring, agent optimization loop, 3 validation modes, CLI |
| **v0.2** | Q2 2026 | 200+ rules, pharma/food domains, regulatory constraints, `openmix score` CLI |
| **v0.3** | Q3 2026 | INCI-to-SMILES bridge (10K+ ingredients), RDKit-powered solubility and charge checks |
| **v0.5** | Q4 2026 | **FormulaBench**: benchmark dataset of formulations with known outcomes, baseline ML models, leaderboard |
| **v1.0** | H1 2027 | Mixture property prediction (stability, phase, shelf life), pre-trained models on HuggingFace |
| **v2.0** | 2027 | Multi-objective optimization, ingredient substitution, Bayesian search |
| **v3.0** | 2028+ | Autonomous formulation agent with cloud lab integration and active learning |

---

## Project Structure

```
openmix/
├── src/openmix/
│   ├── __init__.py              # Public API: Formula, validate, score
│   ├── schema.py                # Formula, Ingredient, ValidationReport
│   ├── validate.py              # Rule engine with 3 validation modes
│   ├── score.py                 # Quantitative stability scoring
│   ├── agent.py                 # Iterative optimization agent
│   ├── molecular.py             # RDKit integration (optional)
│   ├── cli/
│   │   └── main.py              # CLI: openmix validate, openmix info
│   └── knowledge/
│       ├── loader.py            # YAML knowledge loader (hard/soft rules)
│       └── data/
│           ├── incompatible_pairs.yaml   # 21 hard + 39 soft rules
│           ├── oil_hlb.yaml              # 42 oil HLB values
│           └── aliases.yaml              # Ingredient alias mappings
├── tests/
│   ├── test_validate.py         # 23 validation tests
│   └── test_score.py            # 8 scoring tests
├── examples/
│   ├── vitamin_c_serum.yaml     # Example formula file
│   └── demo_agent.py            # Agent optimization demo
├── CONTRIBUTING.md
├── pyproject.toml               # pip install openmix
└── LICENSE                      # Apache 2.0
```

---

<a id="citation"></a>

## Citation

If you use OpenMix in your research, please cite:

```bibtex
@software{krishnan2026openmix,
  author = {Krishnan, Vijay},
  title = {OpenMix: An Open-Source Framework for Computational Formulation Science},
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

**OpenMix is built for the scientists who've been doing formulation on spreadsheets and institutional memory.** There's a better way.

*Star this repo if you believe formulation science deserves open-source infrastructure.*

</div>
