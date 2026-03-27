"""
OpenMix — computational mixture and formulation science.

    from openmix import Formula, validate, score, observe

    formula = Formula(
        ingredients=[("Water", 80.0), ("Glycerin", 10.0), ("Niacinamide", 5.0),
                     ("Phenoxyethanol", 1.0), ("Xanthan Gum", 0.5)],
        target_ph=5.5,
    )
    print(validate(formula))          # Rule-based interaction checks
    print(score(formula))             # Heuristic stability score
    print(observe(formula))           # Physics observation engine
    print(observe(formula,            # Discovery: investigate discrepancies
          mode="discovery"))
"""

__version__ = "0.2.0"

from openmix.schema import Formula, Ingredient, ValidationReport, Issue
from openmix.validate import validate
from openmix.score import score, StabilityScore
from openmix.observe import observe, FormulationObservation, Observation, ObserveMode
from openmix.knowledge.loader import load_knowledge
from openmix.experiment import Experiment, ExperimentLog

__all__ = [
    "Formula",
    "Ingredient",
    "ValidationReport",
    "Issue",
    "validate",
    "score",
    "StabilityScore",
    "observe",
    "FormulationObservation",
    "Observation",
    "ObserveMode",
    "Experiment",
    "ExperimentLog",
    "load_knowledge",
    "__version__",
]
