"""
OpenMix — computational mixture and formulation science.

    from openmix import Formula, validate, score

    formula = Formula(
        ingredients=[("Water", 80.0), ("Glycerin", 10.0), ("Niacinamide", 5.0),
                     ("Phenoxyethanol", 1.0), ("Xanthan Gum", 0.5)],
        target_ph=5.5,
    )
    print(validate(formula))
    print(score(formula))
"""

__version__ = "0.1.0"

from openmix.schema import Formula, Ingredient, ValidationReport, Issue
from openmix.validate import validate
from openmix.score import score, StabilityScore
from openmix.knowledge.loader import load_knowledge

__all__ = [
    "Formula",
    "Ingredient",
    "ValidationReport",
    "Issue",
    "validate",
    "score",
    "StabilityScore",
    "load_knowledge",
    "__version__",
]
