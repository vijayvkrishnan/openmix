# Contributing to OpenMix

OpenMix is built to grow through community contributions. Whether you're a chemist adding interaction rules, an ML researcher contributing models, or a developer improving the framework — there's a place for you.

## Ways to Contribute

### 1. Add Ingredient Interaction Rules (No Code Required)

This is the highest-impact, lowest-barrier contribution. If you know that two ingredients shouldn't be combined — or require special handling — add a rule.

**File:** `src/openmix/knowledge/data/incompatible_pairs.yaml`

```yaml
- a: INGREDIENT A          # INCI name, UPPERCASED
  b: INGREDIENT B
  severity: warning         # error | warning | info
  mechanism: pH_conflict    # see mechanism types below
  category: skincare        # see categories below
  message: >
    Clear explanation of what happens, why it matters,
    and what to do about it. Include concentration thresholds
    and alternative ingredients when possible.
```

**Severity levels:**
- `error` — Must never combine. Safety risk or complete deactivation.
- `warning` — Significantly reduces efficacy or stability. Should be avoided.
- `info` — Suboptimal but manageable. Worth knowing about.

**Mechanism types:**
`oxidation`, `pH_conflict`, `chelation`, `precipitation`, `irritation`, `absorption_competition`, `degradation`, `chemical_reaction`, `safety`

**Categories:**
`skincare`, `supplement`, `beverage`, `home_care`, `pharma`, `food`, `all`

### 2. Add Oil HLB Values

**File:** `src/openmix/knowledge/data/oil_hlb.yaml`

```yaml
OIL INCI NAME: 7.5   # Required HLB for o/w emulsification
```

### 3. Add Ingredient Aliases

If a rule uses a generic name (e.g., "CALCIUM") that should match specific forms (e.g., "CALCIUM CARBONATE", "DICALCIUM PHOSPHATE"):

**File:** `src/openmix/knowledge/data/aliases.yaml`

```yaml
GENERIC NAME:
  - SPECIFIC FORM 1
  - SPECIFIC FORM 2
```

### 4. Improve the Validation Engine

See open issues tagged `good first issue` or `help wanted`. The validation engine is in `src/openmix/validate.py`.

### 5. Add Tests

Tests live in `tests/`. We use pytest. Every new rule category should have at least one test proving it works.

## Development Setup

```bash
git clone https://github.com/vijayvkrishnan/openmix.git
cd openmix
pip install -e ".[dev]"
pytest
```

## PR Guidelines

1. One logical change per PR
2. All tests must pass (`pytest`)
3. Knowledge contributions: include a source or explanation in your PR description
4. Code contributions: follow existing patterns, type hints required

## Priority Areas

We especially need contributions in:
- **Pharmaceutical excipient compatibility** — drug-excipient interactions, stability concerns
- **Food science interactions** — emulsifier systems, flavor stability, preservation
- **Regional regulatory constraints** — EU, US, Japan, ASEAN, Brazil concentration limits
- **Hair care** — color chemistry interactions, protein-surfactant compatibility
- **Fragrance** — IFRA restrictions, allergen interactions

## Questions?

Open an issue or start a discussion. We're friendly.
