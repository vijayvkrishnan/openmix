"""
Formulation schema — structured representation for mixtures.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class Ingredient(BaseModel):
    """A single ingredient in a formulation."""

    inci_name: str = Field(description="INCI (International Nomenclature of Cosmetic Ingredients) name")
    percentage: float = Field(ge=0, le=100, description="Weight percentage in the formula")
    phase: Optional[str] = Field(default=None, description="Formulation phase (A/water, B/oil, C/cool-down, etc.)")
    function: Optional[str] = Field(default=None, description="Functional role (emulsifier, humectant, active, etc.)")
    cas_number: Optional[str] = Field(default=None, description="CAS Registry Number")
    smiles: Optional[str] = Field(default=None, description="SMILES string for molecular identity")


class Formula(BaseModel):
    """
    A complete formulation — the core data structure of OpenMix.

    Can be constructed from tuples for convenience:
        Formula(ingredients=[("Glycerin", 5.0), ("Water", 90.0)])
    Or from full Ingredient objects for richer data.
    """

    name: Optional[str] = Field(default=None, description="Formula name")
    ingredients: list[Ingredient] = Field(description="List of ingredients with percentages")
    target_ph: Optional[float] = Field(default=None, ge=0, le=14, description="Target pH of the formulation")
    category: Optional[str] = Field(default=None, description="Product category (skincare, supplement, beverage, home-care)")
    product_type: Optional[str] = Field(default=None, description="Specific product type (serum, moisturizer, shampoo, etc.)")

    @model_validator(mode="before")
    @classmethod
    def coerce_ingredient_tuples(cls, data):
        """Allow ingredients to be passed as (name, percentage) tuples."""
        if isinstance(data, dict) and "ingredients" in data:
            coerced = []
            for item in data["ingredients"]:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        coerced.append({"inci_name": item[0], "percentage": item[1]})
                    elif len(item) == 3:
                        coerced.append({"inci_name": item[0], "percentage": item[1], "phase": item[2]})
                    else:
                        coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(item)
                elif isinstance(item, Ingredient):
                    coerced.append(item)
                else:
                    coerced.append(item)
            data["ingredients"] = coerced
        return data

    @property
    def total_percentage(self) -> float:
        return sum(i.percentage for i in self.ingredients)

    @property
    def inci_names(self) -> list[str]:
        return [i.inci_name for i in self.ingredients]

    @property
    def inci_names_upper(self) -> set[str]:
        return {i.inci_name.upper().strip() for i in self.ingredients}


class Issue(BaseModel):
    """A single validation issue found in a formulation."""

    check: str = Field(description="Check type: compatibility, ph, hlb, solubility, charge, regulatory")
    severity: str = Field(description="error, warning, or info")
    message: str = Field(description="Human-readable description of the issue")
    ingredient: Optional[str] = Field(default=None, description="Primary ingredient involved")
    ingredient_b: Optional[str] = Field(default=None, description="Second ingredient (for pair interactions)")
    mechanism: Optional[str] = Field(default=None, description="Interaction mechanism (oxidation, pH_conflict, etc.)")
    details: Optional[dict] = Field(default=None, description="Additional structured data")


class ValidationReport(BaseModel):
    """Complete validation report for a formulation."""

    formula_name: Optional[str] = None
    overall_score: float = Field(description="Composite score 0-100")
    ph_score: Optional[float] = None
    hlb_score: Optional[float] = None
    solubility_score: Optional[float] = None
    issues: list[Issue] = Field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    ingredients_checked: int = 0

    @property
    def passed(self) -> bool:
        """True if no errors were found."""
        return self.errors == 0

    def __str__(self) -> str:
        lines = []
        lines.append(f"OpenMix Validation Report{f': {self.formula_name}' if self.formula_name else ''}")
        lines.append(f"Score: {self.overall_score:.0f}/100  |  {self.errors} errors, {self.warnings} warnings, {self.infos} info")
        lines.append("")

        for issue in self.issues:
            icon = {"error": "X", "warning": "!", "info": "-"}[issue.severity]
            prefix = f"[{icon}]"
            if issue.ingredient and issue.ingredient_b:
                lines.append(f"  {prefix} {issue.ingredient} + {issue.ingredient_b}")
            elif issue.ingredient:
                lines.append(f"  {prefix} {issue.ingredient}")
            else:
                lines.append(f"  {prefix}")
            lines.append(f"      {issue.message}")
            lines.append("")

        if not self.issues:
            lines.append("  No issues found.")

        return "\n".join(lines)
