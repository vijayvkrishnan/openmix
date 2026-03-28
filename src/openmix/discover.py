"""
Formulation rule discovery from experimental data. (EXPERIMENTAL)

Given formulations + outcomes, proposes hypotheses, tests them against
the data, and iterates. Discovers interpretable rules ranked by
statistical significance.

Status: experimental — the hypothesis generation can be noisy.
The statistical testing framework is solid; the LLM prompt
for generating hypotheses is the area for improvement.

Output: discovered rules with evidence, ready to add to the knowledge base.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from openmix.benchmarks.shampoo import (
    ShampooStability, ShampooRecord, INGREDIENT_COLS,
    TRADE_TO_INCI, TRADE_TO_TYPE,
)

from openmix.llm import LLMProvider, AnthropicProvider


@dataclass
class DiscoveredRule:
    """A formulation rule discovered from data."""
    hypothesis: str           # human-readable description
    condition: str            # machine-parseable condition
    predicts: str             # "stable" or "unstable"
    n_matching: int           # formulations matching the condition
    n_correct: int            # of those, how many match the prediction
    precision: float          # n_correct / n_matching
    recall: float             # n_correct / total with that outcome
    p_value: float            # Fisher's exact test
    effect_size: float        # absolute difference in stability rate
    evidence: str             # "strong", "moderate", "weak"

    def __str__(self) -> str:
        return (
            f"  {self.hypothesis}\n"
            f"    Predicts: {self.predicts} | "
            f"Precision: {self.precision:.0%} | "
            f"Recall: {self.recall:.0%} | "
            f"p={self.p_value:.4f} | "
            f"Evidence: {self.evidence}\n"
            f"    Matches {self.n_matching} formulations, "
            f"{self.n_correct} correct"
        )


def _compute_features(record: ShampooRecord) -> dict[str, float]:
    """Compute named features for a record."""
    pcts = record.feature_vector
    features = {}

    for col, pct in zip(INGREDIENT_COLS, pcts):
        features[col] = pct
        inci = TRADE_TO_INCI[col]
        features[f"inci_{inci}"] = pct

    # Type aggregates
    for type_name in ["anionic", "nonionic", "amphoteric", "cationic",
                      "cationic_polymer", "nonionic_thickener"]:
        total = sum(p for c, p in zip(INGREDIENT_COLS, pcts)
                    if TRADE_TO_TYPE.get(c) == type_name)
        features[type_name] = total

    features["total_active"] = sum(pcts)
    features["n_ingredients"] = sum(1 for p in pcts if p > 0)
    features["water_pct"] = 100 - features["total_active"]

    total_charged = features["anionic"] + features["cationic"] + 1e-8
    features["charge_ratio"] = (features["anionic"] - features["cationic"]) / total_charged
    features["cationic_total"] = features["cationic"] + features["cationic_polymer"]

    return features


def test_hypothesis(
    records: list[ShampooRecord],
    condition_fn,
    predicts_stable: bool,
) -> Optional[DiscoveredRule]:
    """
    Test a hypothesis against the data.

    condition_fn: function(features_dict) -> bool
    predicts_stable: True if the condition predicts stability
    """
    matching = []
    not_matching = []

    for r in records:
        features = _compute_features(r)
        if condition_fn(features):
            matching.append(r)
        else:
            not_matching.append(r)

    if len(matching) < 10:  # too few to be meaningful
        return None

    if predicts_stable:
        correct = sum(1 for r in matching if r.stable)
        total_outcome = sum(1 for r in records if r.stable)
    else:
        correct = sum(1 for r in matching if not r.stable)
        total_outcome = sum(1 for r in records if not r.stable)

    precision = correct / len(matching) if matching else 0
    recall = correct / total_outcome if total_outcome else 0

    # Fisher's exact test: is the stability rate in matching significantly
    # different from the baseline?
    a = correct  # matching + predicted outcome
    b = len(matching) - correct  # matching + other outcome
    c = total_outcome - correct  # not matching + predicted outcome
    d = len(not_matching) - (total_outcome - correct)  # not matching + other

    # Effect size: stability rate in matching vs overall
    baseline_rate = total_outcome / len(records)
    matching_rate = correct / len(matching)
    effect_size = abs(matching_rate - baseline_rate)

    if SCIPY_AVAILABLE:
        _, p_value = stats.fisher_exact([[a, b], [c, d]])
    else:
        p_value = 1.0 if effect_size < 0.05 else 0.01

    if p_value < 0.001 and effect_size > 0.15:
        evidence = "strong"
    elif p_value < 0.01 and effect_size > 0.10:
        evidence = "moderate"
    elif p_value < 0.05:
        evidence = "weak"
    else:
        evidence = "not significant"

    return DiscoveredRule(
        hypothesis="",  # filled by caller
        condition="",
        predicts="stable" if predicts_stable else "unstable",
        n_matching=len(matching),
        n_correct=correct,
        precision=precision,
        recall=recall,
        p_value=p_value,
        effect_size=effect_size,
        evidence=evidence,
    )


def auto_sweep(
    records: list[ShampooRecord],
    features_to_test: list[str] | None = None,
) -> list[DiscoveredRule]:
    """
    Exhaustive single-feature sweep — finds all significant thresholds.

    Tests every feature at 25th, 50th, 75th percentiles in both directions.
    This is what a data scientist would do first, before creative hypotheses.
    No LLM needed — pure statistical testing.
    """
    all_features = {}
    for r in records:
        feats = _compute_features(r)
        for k, v in feats.items():
            all_features.setdefault(k, []).append(v)

    test_features = features_to_test or [
        k for k in all_features
        if k not in ("water_pct",) and len(set(all_features[k])) > 3
    ]

    discoveries = []
    for feat in test_features:
        values = all_features[feat]
        non_zero = [v for v in values if v > 0]
        if len(non_zero) < 20:
            continue

        percentiles = [25, 50, 75]
        thresholds = list(set(round(float(np.percentile(non_zero, p)), 2) for p in percentiles))

        for thresh in thresholds:
            for direction in ["stable", "unstable"]:
                predicts_stable = direction == "stable"

                def condition(features, _f=feat, _t=thresh):
                    return features.get(_f, 0) > _t

                rule = test_hypothesis(records, condition, predicts_stable)
                if rule is None:
                    continue

                if rule.evidence in ("strong", "moderate"):
                    rule.hypothesis = (
                        f"{feat} > {thresh:.1f} predicts {direction} "
                        f"(precision={rule.precision:.0%}, p={rule.p_value:.4f})"
                    )
                    rule.condition = json.dumps({"feature": feat, "operator": ">", "threshold": thresh})
                    discoveries.append(rule)

    # Deduplicate:
    # 1. Same threshold appears under trade name AND INCI name — keep INCI
    # 2. Same threshold has both stable/unstable directions — keep higher precision
    best_per_threshold = {}
    for rule in discoveries:
        cond = json.loads(rule.condition)
        feat = cond.get("feature", "")
        thresh = cond.get("threshold", 0)

        # Normalize: prefer INCI names, group by (n_matching, effect_size)
        group_key = f"{rule.n_matching}_{rule.effect_size:.4f}"

        existing = best_per_threshold.get(group_key)
        if existing is None:
            best_per_threshold[group_key] = rule
        else:
            # Keep higher precision (more informative direction)
            if rule.precision > existing.precision:
                best_per_threshold[group_key] = rule
            elif rule.precision == existing.precision and feat.startswith("inci_"):
                best_per_threshold[group_key] = rule

    return sorted(best_per_threshold.values(), key=lambda r: -r.effect_size)


SYSTEM_PROMPT = """You are a formulation scientist analyzing experimental data to discover
design rules. You have a dataset of 812 shampoo formulations with stability outcomes.

The formulations use these ingredient types:
- Anionic surfactants (5 types): cleansing, foam
- Nonionic surfactants (2 types): mildness, foam boosting
- Amphoteric surfactants (4 types): mildness, conditioning
- Cationic surfactant (1 type): conditioning, antistatic
- Cationic polymers (4 types): conditioning, viscosity
- Nonionic thickeners (2 types): viscosity building

Overall: 36.2% of formulations are stable, 63.8% unstable.

Your job: propose HYPOTHESES about what makes a formulation stable or unstable.
Each hypothesis must be a simple rule with a threshold, like:
- "Formulations with cationic_polymer > 2.5% are more likely unstable"
- "Formulations with anionic > 10% AND thickener > 3% tend to be stable"
- "Having more than 4 ingredients correlates with stability"

Respond with a JSON array of hypotheses. Each hypothesis:
{
  "description": "human-readable explanation of the rule",
  "feature": "feature_name",           // or "feature_a" + "feature_b" for interactions
  "operator": ">",                      // >, <, ==, between
  "threshold": 2.5,                     // numeric threshold
  "feature_b": null,                    // second feature for interaction rules
  "operator_b": null,
  "threshold_b": null,
  "predicts": "unstable",              // "stable" or "unstable"
  "reasoning": "why you think this matters chemically"
}

Available features: anionic, nonionic, amphoteric, cationic, cationic_polymer,
nonionic_thickener, total_active, n_ingredients, water_pct, charge_ratio,
cationic_total (= cationic + cationic_polymer)

Think like a scientist. What chemical principles would determine whether a
surfactant system phase-separates?"""


ITERATION_PROMPT = """Here are the results of testing your hypotheses:

{results}

DISCOVERED SO FAR (rules with strong/moderate evidence):
{discoveries}

Based on these findings:
1. Analyze which hypotheses were supported and why
2. Notice patterns in what works: which features, which thresholds
3. Propose NEW hypotheses that explore adjacent territory
   - Try tighter/looser thresholds on features that showed signal
   - Try interaction effects between features that individually showed signal
   - Try the OPPOSITE of hypotheses that failed (maybe the relationship is reversed)

Propose 5-8 new hypotheses as a JSON array. Be creative — look for
non-obvious interactions. The best discoveries are the ones that surprise."""


@dataclass
class DiscoveryResult:
    """Complete result from a discovery session."""
    discoveries: list[DiscoveredRule] = field(default_factory=list)
    all_tested: list[DiscoveredRule] = field(default_factory=list)
    iterations: int = 0
    total_duration_ms: int = 0

    def __str__(self) -> str:
        lines = ["", "=" * 70]
        lines.append("  OPENMIX DISCOVERY ENGINE — RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Iterations: {self.iterations}")
        lines.append(f"  Hypotheses tested: {len(self.all_tested)}")
        lines.append(f"  Rules discovered: {len(self.discoveries)}")
        lines.append(f"  Duration: {self.total_duration_ms / 1000:.1f}s")
        lines.append("")

        if self.discoveries:
            # Sort by evidence strength then effect size
            rank = {"strong": 0, "moderate": 1, "weak": 2}
            sorted_d = sorted(self.discoveries,
                            key=lambda r: (rank.get(r.evidence, 3), -r.effect_size))

            lines.append("-" * 70)
            lines.append("  DISCOVERED RULES (ranked by evidence)")
            lines.append("-" * 70)
            for i, rule in enumerate(sorted_d, 1):
                lines.append(f"  #{i}")
                lines.append(str(rule))
                lines.append("")
        else:
            lines.append("  No significant rules discovered.")

        lines.append("=" * 70)
        return "\n".join(lines)


class DiscoveryEngine:
    """
    Autonomous rule discovery from formulation data.

    The agent proposes hypotheses, tests them against real data,
    and iterates to discover formulation design principles.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 4,
        verbose: bool = True,
    ):
        if llm is not None:
            self._llm = llm
        else:
            self._llm = AnthropicProvider(model=model, api_key=api_key)
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _call_llm(self, messages: list[dict]) -> str:
        return self._llm.generate(system=SYSTEM_PROMPT, messages=messages)

    def _parse_hypotheses(self, text: str) -> list[dict]:
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return []
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []

    def _build_condition(self, hyp: dict):
        """Build a condition function from a hypothesis dict."""
        feat = hyp.get("feature", "")
        op = hyp.get("operator", ">")
        thresh = hyp.get("threshold", 0)
        feat_b = hyp.get("feature_b")
        op_b = hyp.get("operator_b")
        thresh_b = hyp.get("threshold_b")

        def _check(val, operator, threshold):
            if operator == ">":
                return val > threshold
            elif operator == "<":
                return val < threshold
            elif operator == ">=":
                return val >= threshold
            elif operator == "<=":
                return val <= threshold
            elif operator == "==":
                return abs(val - threshold) < 0.01
            elif operator == "between":
                if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                    return threshold[0] <= val <= threshold[1]
            return False

        def condition(features):
            val = features.get(feat, 0)
            result = _check(val, op, thresh)
            if feat_b and op_b is not None and thresh_b is not None:
                val_b = features.get(feat_b, 0)
                result = result and _check(val_b, op_b, thresh_b)
            return result

        return condition

    def run(self, dataset: ShampooStability | None = None) -> DiscoveryResult:
        ds = dataset or ShampooStability()
        records = ds.records
        result = DiscoveryResult()
        start_time = time.time()

        self._log("")
        self._log("=" * 70)
        self._log("  OPENMIX DISCOVERY ENGINE")
        self._log("=" * 70)
        self._log(f"  Dataset: {len(records)} formulations "
                  f"({ds.n_stable} stable, {ds.n_unstable} unstable)")
        self._log(f"  Baseline stability rate: {ds.n_stable/len(records)*100:.1f}%")
        self._log("")

        # Phase 1: Auto-sweep — exhaustive single-feature testing (no LLM)
        self._log("  Phase 1: Auto-sweep (exhaustive single-feature scan)")
        sweep_results = auto_sweep(records)
        self._log(f"  Found {len(sweep_results)} significant single-feature rules:")
        for rule in sweep_results:
            icon = {"strong": "***", "moderate": "**", "weak": "*"}.get(rule.evidence, " ")
            self._log(f"    [{icon}] {rule.hypothesis}")
            result.discoveries.append(rule)
            result.all_tested.append(rule)
        self._log("")

        # Phase 2: LLM-guided interaction hypotheses, informed by auto-sweep
        sweep_summary = "\n".join(
            f"  - {r.hypothesis}" for r in sweep_results[:10]
        ) or "  (no significant single-feature rules found)"

        self._log(f"  Phase 2: LLM-guided interaction hypotheses ({self.max_iterations} iterations)")
        self._log("")

        messages = [{"role": "user", "content":
            f"Here are significant single-feature findings from exhaustive testing:\n"
            f"{sweep_summary}\n\n"
            f"Now propose 5-8 INTERACTION hypotheses — combinations of features "
            f"that might predict stability better than single features alone. "
            f"Focus on two-feature interactions informed by the sweep results."}]

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1
            self._log(f"  [Iteration {iteration + 1}] Generating hypotheses...")

            raw = self._call_llm(messages)
            hypotheses = self._parse_hypotheses(raw)

            if not hypotheses:
                self._log("    Failed to parse hypotheses. Stopping.")
                break

            self._log(f"    Testing {len(hypotheses)} hypotheses...")
            self._log("")

            iteration_results = []

            for hyp in hypotheses:
                desc = hyp.get("description", "unknown")
                predicts = hyp.get("predicts", "unstable")
                predicts_stable = predicts == "stable"

                try:
                    condition = self._build_condition(hyp)
                    rule = test_hypothesis(records, condition, predicts_stable)
                except Exception:
                    rule = None

                if rule is None:
                    self._log(f"    [-] {desc[:70]}")
                    self._log("        Too few matches or parse error")
                    iteration_results.append(f"SKIP: {desc} (too few matches)")
                    continue

                rule.hypothesis = desc
                rule.condition = json.dumps({
                    k: hyp.get(k) for k in
                    ["feature", "operator", "threshold",
                     "feature_b", "operator_b", "threshold_b"]
                })
                result.all_tested.append(rule)

                icon = {"strong": "***", "moderate": "**",
                        "weak": "*", "not significant": " "}[rule.evidence]
                self._log(f"    [{icon}] {desc[:70]}")
                self._log(f"        Precision: {rule.precision:.0%}  "
                          f"Recall: {rule.recall:.0%}  "
                          f"p={rule.p_value:.4f}  "
                          f"Effect: {rule.effect_size:.0%}  "
                          f"[{rule.evidence}]")

                iteration_results.append(
                    f"{'DISCOVERED' if rule.evidence in ('strong', 'moderate') else 'NOT SIG'}: "
                    f"{desc} | precision={rule.precision:.0%} recall={rule.recall:.0%} "
                    f"p={rule.p_value:.4f} effect={rule.effect_size:.0%}"
                )

                if rule.evidence in ("strong", "moderate"):
                    result.discoveries.append(rule)

            self._log("")

            # Prepare next iteration
            results_text = "\n".join(iteration_results)
            discoveries_text = "\n".join(
                f"- {r.hypothesis} (precision={r.precision:.0%}, p={r.p_value:.4f})"
                for r in result.discoveries
            ) or "(none yet)"

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                ITERATION_PROMPT.format(
                    results=results_text,
                    discoveries=discoveries_text,
                )
            })

        result.total_duration_ms = int((time.time() - start_time) * 1000)

        if self.verbose:
            print(result)

        return result
