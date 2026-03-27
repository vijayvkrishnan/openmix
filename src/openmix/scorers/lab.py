"""
Lab feedback scorers — real experimental data as evaluation.

For users with access to:
  - Cloud labs (Strateos, Emerald Cloud Lab)
  - Robotic platforms (Opentrons, Chemspeed)
  - Manual lab work (enter results by hand)
  - Existing experimental databases

The interface is simple: take a Formula, return a StabilityScore.
How you get that score is up to you.
"""

from __future__ import annotations

from abc import abstractmethod

from openmix.schema import Formula
from openmix.score import StabilityScore
from openmix.scorers.base import Scorer


class LabScorer(Scorer):
    """
    Base class for real experimental feedback.

    Subclass this to integrate with your lab infrastructure.
    The only method you need to implement is `run_experiment`.
    """

    @abstractmethod
    def run_experiment(self, formula: Formula) -> dict:
        """
        Run a physical experiment and return results.

        Must return a dict with at least:
            {"stable": True/False}

        Optionally:
            {"stable": True, "shelf_life_months": 18, "ph_measured": 5.2,
             "viscosity_cps": 3500, "phase_separation": False}
        """
        ...

    def __call__(self, formula: Formula) -> StabilityScore:
        results = self.run_experiment(formula)
        return self._results_to_score(results)

    def _results_to_score(self, results: dict) -> StabilityScore:
        """Convert lab results to a StabilityScore."""
        stable = results.get("stable", False)
        total = 90.0 if stable else 30.0

        score = StabilityScore(total=total)

        if stable:
            score.bonuses.append("Lab result: STABLE")
            score.compatibility = 35.0
            score.formula_integrity = 10.0
        else:
            score.penalties.append("Lab result: UNSTABLE")
            score.compatibility = 5.0

        # Enrich with specific measurements if available
        if "shelf_life_months" in results:
            months = results["shelf_life_months"]
            score.bonuses.append(f"Shelf life: {months} months")
            # Scale: 24+ months = full marks, 0 months = zero
            score.ph_suitability = round(min(months / 24, 1.0) * 25, 1)

        if "phase_separation" in results:
            if results["phase_separation"]:
                score.penalties.append("Phase separation observed")
                score.emulsion_balance = 0.0
            else:
                score.emulsion_balance = 20.0

        # Recalculate total from sub-scores
        score.total = round(max(0, min(100,
            score.compatibility + score.ph_suitability +
            score.emulsion_balance + score.formula_integrity +
            score.system_completeness
        )), 1)

        return score

    @property
    def name(self) -> str:
        return "lab/custom"


class ManualScorer(Scorer):
    """
    Manual entry — the scientist runs the experiment and types the result.

    Useful for:
      - Small labs without automation
      - Validating agent suggestions by hand
      - Collecting training data for future model scorers
    """

    def __init__(self, prompt: str = "Enter stability result"):
        self.prompt = prompt
        self._results: list[dict] = []

    def __call__(self, formula: Formula) -> StabilityScore:
        print(f"\n{'='*50}")
        print("MANUAL EVALUATION REQUIRED")
        print(f"{'='*50}")
        print(f"Formula: {formula.name or 'unnamed'}")
        for ing in sorted(formula.ingredients, key=lambda x: -x.percentage):
            print(f"  {ing.inci_name:<35} {ing.percentage:>5.1f}%")
        print()

        stable_input = input("Stable? (y/n): ").strip().lower()
        stable = stable_input in ("y", "yes", "true", "1")

        score_input = input("Stability score (0-100, or press Enter for auto): ").strip()

        if score_input:
            total = float(score_input)
            result = StabilityScore(total=max(0, min(100, total)))
            if stable:
                result.bonuses.append("Manual: stable")
            else:
                result.penalties.append("Manual: unstable")
        else:
            result = StabilityScore(
                total=85.0 if stable else 25.0,
                compatibility=35.0 if stable else 5.0,
                ph_suitability=20.0 if stable else 5.0,
                emulsion_balance=15.0 if stable else 5.0,
                formula_integrity=10.0,
                system_completeness=5.0,
            )

        self._results.append({
            "formula": formula.model_dump(),
            "stable": stable,
            "score": result.total,
        })

        return result

    @property
    def name(self) -> str:
        return "lab/manual"

    @property
    def collected_data(self) -> list[dict]:
        """Returns all manually entered results — training data for future models."""
        return self._results


