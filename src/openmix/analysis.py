"""
Post-experiment analysis — extract insights from trial history.

After an experiment runs, this module answers:
- Which ingredients appeared most in high-scoring formulas?
- What separates the best trials from the worst?
- What patterns did the agent discover?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from openmix.experiment import ExperimentLog, Trial


@dataclass
class Insight:
    """A single finding from the experiment."""
    category: str     # "ingredient", "pattern", "strategy"
    description: str
    evidence: str
    strength: str     # "strong", "moderate", "suggestive"


@dataclass
class ExperimentAnalysis:
    """Analysis of a completed experiment."""
    total_trials: int = 0
    unique_formulations: int = 0
    best_score: float = 0.0
    worst_score: float = 0.0
    mean_score: float = 0.0
    score_std: float = 0.0
    convergence_iteration: int | None = None
    insights: list[Insight] = field(default_factory=list)
    top_ingredients: list[tuple[str, float]] = field(default_factory=list)
    bottom_ingredients: list[tuple[str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  EXPERIMENT ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"  Trials: {self.total_trials} ({self.unique_formulations} unique)")
        lines.append(f"  Scores: {self.best_score:.1f} best / "
                     f"{self.mean_score:.1f} mean / {self.worst_score:.1f} worst")
        lines.append(f"  Std: {self.score_std:.1f}")

        if self.convergence_iteration:
            lines.append(f"  Best found at iteration: {self.convergence_iteration}")

        if self.top_ingredients:
            lines.append("")
            lines.append("  Ingredients in TOP formulas (score >= mean):")
            for name, freq in self.top_ingredients[:8]:
                bar = "#" * int(freq * 20)
                lines.append(f"    {name:<35} {freq:.0%} |{bar}")

        if self.bottom_ingredients:
            lines.append("")
            lines.append("  Ingredients in BOTTOM formulas (score < mean):")
            for name, freq in self.bottom_ingredients[:8]:
                bar = "#" * int(freq * 20)
                lines.append(f"    {name:<35} {freq:.0%} |{bar}")

        if self.insights:
            lines.append("")
            lines.append("  Insights:")
            for i, insight in enumerate(self.insights, 1):
                lines.append(f"    {i}. [{insight.strength}] {insight.description}")
                lines.append(f"       {insight.evidence}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze(log: ExperimentLog) -> ExperimentAnalysis:
    """Analyze a completed experiment log."""
    if not log.trials:
        return ExperimentAnalysis()

    scores = [t.stability.total for t in log.trials]
    mean_score = sum(scores) / len(scores)

    import numpy as np
    scores_arr = np.array(scores)

    # Find when best was first achieved
    best = max(scores)
    convergence_iter = None
    for t in log.trials:
        if t.stability.total == best:
            convergence_iter = t.iteration
            break

    # Split trials into top/bottom by score
    top_trials = [t for t in log.trials if t.stability.total >= mean_score]
    bottom_trials = [t for t in log.trials if t.stability.total < mean_score]

    # Ingredient frequency in top vs bottom
    top_ingredients = _ingredient_frequency(top_trials)
    bottom_ingredients = _ingredient_frequency(bottom_trials)

    # Extract insights
    insights = _extract_insights(log.trials, top_trials, bottom_trials,
                                  top_ingredients, bottom_ingredients, mean_score)

    analysis = ExperimentAnalysis(
        total_trials=len(log.trials),
        unique_formulations=log.unique_formulations,
        best_score=best,
        worst_score=min(scores),
        mean_score=round(mean_score, 1),
        score_std=round(float(scores_arr.std()), 1),
        convergence_iteration=convergence_iter,
        insights=insights,
        top_ingredients=sorted(top_ingredients.items(), key=lambda x: -x[1]),
        bottom_ingredients=sorted(bottom_ingredients.items(), key=lambda x: -x[1]),
    )

    return analysis


def _ingredient_frequency(trials: list[Trial]) -> dict[str, float]:
    """How often each ingredient appears in a set of trials (0-1)."""
    if not trials:
        return {}

    counts = Counter()
    for t in trials:
        for ing in t.formula.ingredients:
            counts[ing.inci_name] += 1

    return {name: count / len(trials) for name, count in counts.items()}


def _extract_insights(
    all_trials: list[Trial],
    top_trials: list[Trial],
    bottom_trials: list[Trial],
    top_freq: dict[str, float],
    bottom_freq: dict[str, float],
    mean_score: float,
) -> list[Insight]:
    """Extract actionable insights from trial comparison."""
    insights = []

    # Find ingredients that appear much more in top than bottom
    for name, top_f in top_freq.items():
        bottom_f = bottom_freq.get(name, 0)
        diff = top_f - bottom_f

        if diff > 0.3 and top_f > 0.5:
            insights.append(Insight(
                category="ingredient",
                description=f"{name} appears in {top_f:.0%} of high-scoring formulas "
                            f"vs {bottom_f:.0%} of low-scoring ones",
                evidence=f"Frequency difference: {diff:.0%}",
                strength="strong" if diff > 0.5 else "moderate",
            ))

    # Find ingredients that appear much more in bottom than top
    for name, bottom_f in bottom_freq.items():
        top_f = top_freq.get(name, 0)
        diff = bottom_f - top_f

        if diff > 0.3 and bottom_f > 0.5:
            insights.append(Insight(
                category="ingredient",
                description=f"{name} appears in {bottom_f:.0%} of low-scoring formulas "
                            f"vs {top_f:.0%} of high-scoring ones — may be destabilizing",
                evidence=f"Frequency difference: {diff:.0%}",
                strength="strong" if diff > 0.5 else "moderate",
            ))

    # Check sub-score patterns
    if top_trials and bottom_trials:
        top_ph = sum(t.stability.ph_suitability for t in top_trials) / len(top_trials)
        bottom_ph = sum(t.stability.ph_suitability for t in bottom_trials) / len(bottom_trials)

        if top_ph - bottom_ph > 3:
            insights.append(Insight(
                category="pattern",
                description=f"pH suitability is the key differentiator: "
                            f"{top_ph:.1f} in top vs {bottom_ph:.1f} in bottom formulas",
                evidence=f"pH sub-score gap: {top_ph - bottom_ph:.1f} points",
                strength="strong",
            ))

        top_compat = sum(t.stability.compatibility for t in top_trials) / len(top_trials)
        bottom_compat = sum(t.stability.compatibility for t in bottom_trials) / len(bottom_trials)

        if top_compat - bottom_compat > 3:
            insights.append(Insight(
                category="pattern",
                description=f"Compatibility drives the gap: "
                            f"{top_compat:.1f} in top vs {bottom_compat:.1f} in bottom",
                evidence=f"Compatibility sub-score gap: {top_compat - bottom_compat:.1f} points",
                strength="strong",
            ))

    # Score trajectory insight
    scores = [t.stability.total for t in all_trials]
    if len(scores) >= 5:
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        if second_mean > first_mean + 2:
            insights.append(Insight(
                category="strategy",
                description="Agent improved over time: second half scores "
                            f"averaged {second_mean:.1f} vs {first_mean:.1f} in first half",
                evidence=f"Improvement: +{second_mean - first_mean:.1f} points",
                strength="moderate",
            ))
        elif abs(second_mean - first_mean) < 1:
            insights.append(Insight(
                category="strategy",
                description="Agent plateaued early — scores were flat throughout. "
                            "Consider expanding the ingredient pool or relaxing constraints.",
                evidence=f"First half: {first_mean:.1f}, second half: {second_mean:.1f}",
                strength="suggestive",
            ))

    return insights
