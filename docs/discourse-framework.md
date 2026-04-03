# Multi-Perspective Evaluation with Discourse Classification

*A domain-agnostic architecture pattern for complex system assessment.*

---

## The Problem

Complex systems are evaluated by multiple frameworks, each with different tools, different knowledge, and different blind spots. A pharmaceutical formulation is simultaneously:

- A **molecular system** (LogP, charge, functional groups)
- A **chemical system** (reaction mechanisms, degradation pathways)
- A **statistical artifact** (historical stability data, benchmark performance)
- A **manufacturing process** (thermal limits, mixing order, equipment)

No single framework captures all the relevant information. A physics model knows about solubility but not about Maillard reactions. A knowledge base knows about documented interactions but can't predict novel ones. A machine learning model knows what's worked before but can't explain why.

The standard approach is to run each evaluation independently and present the results. The problem: when evaluations disagree, the user has to figure out why. Is it because one evaluation is wrong? Because they're looking at different aspects of the problem? Because the evidence is genuinely ambiguous?

## The Architecture

The discourse engine answers this question computationally. It:

1. **Runs multiple evaluations** on the same system
2. **Groups their outputs by topic** (what they're about)
3. **Classifies each topic** based on how the evaluations relate

### The Four Classifications

**Agreement**: Multiple perspectives concur. The assessment is reliable. Move on.

**Correction**: One perspective has significantly stronger evidence than another. The stronger side overrides. This is not a disagreement -- it's error correction. Apply the correction and move on.

**True Disagreement**: Perspectives have comparable evidence but reach different conclusions. Neither can disprove the other. These are the interesting cases -- where our understanding is incomplete, where investigation might reveal something new.

**Knowledge Gap**: No perspective has enough information. The system explicitly says "we don't know" instead of silently producing a partial assessment.

### The Evidence Hierarchy

The classification depends on an evidence hierarchy that ranks claim strength:

```
Level 6: Empirical direct    (measured data for this exact system)
Level 5: Empirical proxy     (data from similar systems, commercial products)
Level 4: Computational       (physics from molecular structure -- RDKit, H-H)
Level 3: Rule-based          (curated knowledge base with cited sources)
Level 2: Heuristic           (rules of thumb, common practice)
Level 1: LLM reasoning       (model said so, no computational backing)
```

When two claims disagree and their evidence levels differ by 2+ levels, the stronger claim corrects the weaker one. When the gap is smaller (0-1 levels), both sides have defensible positions -- it's a true disagreement worth investigating.

This threshold (2 levels) is a design choice, not an empirical constant. It encodes the principle: a computational prediction can override an LLM's reasoning, but it can't override a curated rule from the literature. Both have enough standing to disagree.

### The Key Insight

**Corrections are error prevention. True disagreements are discovery opportunities.**

A system that only reports corrections is a validator. A system that identifies and classifies genuine disagreements is a research tool. The discourse engine does both.

## Generalizability

This pattern is not specific to formulation chemistry. It applies to any domain where:

1. Multiple evaluation frameworks exist with different knowledge
2. No single framework is authoritative on all aspects
3. The cost of missing a problem is high
4. Disagreements between frameworks contain information

**Examples:**

- **Drug safety**: Clinical data vs mechanistic models vs patient demographics vs drug interaction databases. When they disagree about a drug-drug interaction, is one framework wrong or are they capturing different risks?

- **Materials engineering**: Finite element analysis vs empirical testing data vs manufacturing constraints vs cost models. When FEA says a design is safe but testing data from similar designs shows failure, that's a true disagreement worth investigating.

- **Software architecture**: Security analysis vs performance profiling vs maintainability metrics vs cost estimates. When the security framework says "encrypt everything" but the performance framework says "latency is critical," the discourse engine classifies this as a true disagreement (both have strong evidence), not a correction.

- **Climate modeling**: Multiple climate models evaluating the same scenario. Where they agree, confidence is high. Where they disagree, the disagreement tells you something about model uncertainty.

The evidence hierarchy adapts to each domain. In software, a measured benchmark (empirical direct) overrides an architectural heuristic. In medicine, a randomized controlled trial (empirical direct) overrides a mechanistic hypothesis (computational). The structure is constant; the content is domain-specific.

## Implementation

The minimal implementation requires:

1. **Perspectives**: Functions that evaluate a system and produce structured claims (subject, position, evidence, evidence_level, confidence)
2. **Topic matching**: Group claims about the same subject
3. **Classification logic**: Compare evidence levels within each topic
4. **Output structure**: Agreements, corrections, true disagreements, knowledge gaps

In OpenMix, this is ~250 lines of Python with no external dependencies beyond the perspective functions themselves. The architecture is deliberately simple -- the complexity lives in the perspectives, not in the discourse engine.

```python
# The core classification -- 15 lines that do the work
def classify_topic(claims):
    if all claims agree:
        return "agreement"

    safe_claims = [c for c in claims if c.position == "safe"]
    concern_claims = [c for c in claims if c.position in ("concern", "violation")]

    if safe_claims and concern_claims:
        gap = abs(max_evidence(safe_claims) - max_evidence(concern_claims))
        if gap >= CORRECTION_THRESHOLD:
            return "correction"  # stronger side wins
        return "true_disagreement"  # both sides have standing

    return "knowledge_gap"
```

The value is not in the algorithm. It's in the framing: explicitly classifying disagreements by evidence strength, rather than averaging scores or picking the most conservative answer.
