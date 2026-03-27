"""
Constraint enforcement — validates formulas meet experiment requirements.

The LLM is instructed to follow constraints, but we verify programmatically.
Non-compliant formulas are rejected before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openmix.schema import Formula


@dataclass
class ConstraintViolation:
    constraint: str
    message: str


@dataclass
class ConstraintResult:
    passed: bool
    violations: list[ConstraintViolation] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "Constraints: all passed"
        lines = ["Constraint violations:"]
        for v in self.violations:
            lines.append(f"  - {v.constraint}: {v.message}")
        return "\n".join(lines)


def check_constraints(
    formula: Formula,
    required_ingredients: list[dict] | None = None,
    available_ingredients: list[str] | None = None,
    constraints: dict | None = None,
) -> ConstraintResult:
    """
    Check if a formula meets experiment constraints.

    Returns ConstraintResult with pass/fail and violation details.
    """
    violations = []
    inci_upper = {i.inci_name.upper().strip() for i in formula.ingredients}
    inci_pcts = {i.inci_name.upper().strip(): i.percentage for i in formula.ingredients}

    # Check required ingredients
    for req in (required_ingredients or []):
        name = req["name"].upper().strip()
        matched = None
        for inci in inci_upper:
            if name in inci or inci in name:
                matched = inci
                break

        if not matched:
            violations.append(ConstraintViolation(
                constraint="required_ingredient",
                message=f"Required ingredient '{req['name']}' not found in formula",
            ))
            continue

        pct = inci_pcts.get(matched, 0)
        min_pct = req.get("min_pct", 0)
        max_pct = req.get("max_pct", 100)

        if pct < min_pct:
            violations.append(ConstraintViolation(
                constraint="min_percentage",
                message=f"{req['name']} at {pct:.1f}% is below minimum {min_pct:.1f}%",
            ))
        if pct > max_pct:
            violations.append(ConstraintViolation(
                constraint="max_percentage",
                message=f"{req['name']} at {pct:.1f}% exceeds maximum {max_pct:.1f}%",
            ))

    # Check ingredient pool (if specified, formula should only use pool ingredients)
    if available_ingredients:
        pool_upper = {i.upper().strip() for i in available_ingredients}
        # Add required ingredients to pool
        for req in (required_ingredients or []):
            pool_upper.add(req["name"].upper().strip())
        # Add common bases
        pool_upper.update({"WATER", "AQUA", "PURIFIED WATER"})

        for inci in inci_upper:
            if not any(inci in p or p in inci for p in pool_upper):
                violations.append(ConstraintViolation(
                    constraint="ingredient_pool",
                    message=f"'{inci}' is not in the allowed ingredient pool",
                ))

    # Check constraints dict
    c = constraints or {}

    # Total percentage
    total = formula.total_percentage
    if "total_percentage" in c:
        target = c["total_percentage"]
        if abs(total - target) > 2.0:
            violations.append(ConstraintViolation(
                constraint="total_percentage",
                message=f"Total is {total:.1f}%, target is {target}%",
            ))

    # Max ingredients
    if "max_ingredients" in c:
        n = len(formula.ingredients)
        if n > c["max_ingredients"]:
            violations.append(ConstraintViolation(
                constraint="max_ingredients",
                message=f"{n} ingredients exceeds max {c['max_ingredients']}",
            ))

    # Target pH range
    if "target_ph" in c and formula.target_ph is not None:
        ph_range = c["target_ph"]
        if isinstance(ph_range, (list, tuple)) and len(ph_range) == 2:
            if formula.target_ph < ph_range[0] or formula.target_ph > ph_range[1]:
                violations.append(ConstraintViolation(
                    constraint="target_ph",
                    message=f"pH {formula.target_ph} outside range {ph_range[0]}-{ph_range[1]}",
                ))

    return ConstraintResult(
        passed=len(violations) == 0,
        violations=violations,
    )
