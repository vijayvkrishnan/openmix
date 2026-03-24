"""
Rule-based formulation validation with mode support.

Three modes: safety (strict), formulation (professional), discovery (permissive).
Hard rules always fire. Soft rules adjust severity by mode.
"""

from __future__ import annotations

from typing import Literal

from openmix.schema import Formula, Ingredient, Issue, ValidationReport
from openmix.knowledge.loader import Knowledge, InteractionRule, load_knowledge
from openmix.matching import match_ingredient


Mode = Literal["safety", "formulation", "discovery"]


def validate(
    formula: Formula,
    mode: Mode = "safety",
    knowledge: Knowledge | None = None,
) -> ValidationReport:
    """
    Run all validation checks on a formulation.

    Args:
        formula: The formulation to validate.
        mode: Validation mode — "safety", "formulation", or "discovery".
        knowledge: Optional custom knowledge base. Uses bundled if None.

    Returns:
        ValidationReport with scores, issues, and coverage info.
    """
    kb = knowledge or load_knowledge()

    all_issues: list[Issue] = []

    # Check 1: Ingredient interactions (hard + soft rules)
    interaction_issues = check_interactions(formula, kb, mode)
    all_issues.extend(interaction_issues)

    # Check 2: HLB balance (emulsion stability)
    hlb_issues, hlb_score = check_hlb_balance(formula, kb)
    all_issues.extend(hlb_issues)

    # Check 3: Formula sanity
    sanity_issues = check_formula_sanity(formula)
    all_issues.extend(sanity_issues)

    # Check 4: Coverage warning
    coverage_issues = check_coverage(formula, kb)
    all_issues.extend(coverage_issues)

    # Score
    errors = sum(1 for i in all_issues if i.severity == "error")
    warnings = sum(1 for i in all_issues if i.severity == "warning")
    infos = sum(1 for i in all_issues if i.severity == "info")

    score = max(0.0, 100.0 - (errors * 25.0) - (warnings * 10.0))

    return ValidationReport(
        formula_name=formula.name,
        overall_score=round(score, 1),
        hlb_score=hlb_score,
        issues=all_issues,
        errors=errors,
        warnings=warnings,
        infos=infos,
        ingredients_checked=len(formula.ingredients),
    )


# ---------------------------------------------------------------------------
# Check 1: Ingredient Interactions (Hard + Soft Rules)
# ---------------------------------------------------------------------------

def _get_ingredient_percentage(formula: Formula, inci_upper: str) -> float | None:
    """Get the percentage of an ingredient by uppercased INCI name."""
    for ing in formula.ingredients:
        if ing.inci_name.upper().strip() == inci_upper:
            return ing.percentage
    return None


def _check_conditions(rule: InteractionRule, formula: Formula,
                      a_match: str, b_match: str) -> bool:
    """
    Check if a soft rule's conditions are met.

    Returns True if conditions are met (rule should fire) or if no conditions.
    """
    conditions = rule.conditions
    if not conditions:
        return True

    # pH conditions
    if "ph_below" in conditions:
        if formula.target_ph is not None and formula.target_ph >= conditions["ph_below"]:
            return False

    if "ph_above" in conditions:
        if formula.target_ph is not None and formula.target_ph <= conditions["ph_above"]:
            return False

    # Concentration conditions
    pct_a = _get_ingredient_percentage(formula, a_match)
    pct_b = _get_ingredient_percentage(formula, b_match)

    if "min_concentration_a" in conditions:
        if pct_a is not None and pct_a < conditions["min_concentration_a"]:
            return False

    if "min_concentration_b" in conditions:
        if pct_b is not None and pct_b < conditions["min_concentration_b"]:
            return False

    if "min_concentration_either" in conditions:
        threshold = conditions["min_concentration_either"]
        a_below = pct_a is not None and pct_a < threshold
        b_below = pct_b is not None and pct_b < threshold
        if a_below and b_below:
            return False

    return True


def check_interactions(formula: Formula, kb: Knowledge, mode: Mode) -> list[Issue]:
    """Check for ingredient interactions using hard and soft rules."""
    issues: list[Issue] = []
    inci_set = formula.inci_names_upper

    for rule in kb.interaction_rules:
        # Check if rule applies to this mode
        if not rule.should_fire(mode):
            continue

        # Filter by category (but hard rules always fire)
        if rule.rule_type == "soft" and formula.category:
            if rule.category not in ("all", formula.category):
                continue

        # Match ingredients
        a_match = match_ingredient(rule.a, inci_set, kb.aliases)
        b_match = match_ingredient(rule.b, inci_set, kb.aliases)

        if not a_match or not b_match or a_match == b_match:
            continue

        # For soft rules, check conditions
        if rule.rule_type == "soft":
            if not _check_conditions(rule, formula, a_match, b_match):
                continue

        severity = rule.get_severity(mode)

        # Build message with context
        message = rule.message
        if rule.mitigation and mode in ("formulation", "discovery"):
            message += f" Mitigation: {rule.mitigation}"

        details = {
            "rule_type": rule.rule_type,
            "confidence": rule.confidence,
            "mechanism": rule.mechanism,
        }
        if rule.source:
            details["source"] = rule.source

        issues.append(Issue(
            check="compatibility",
            severity=severity,
            ingredient=a_match,
            ingredient_b=b_match,
            mechanism=rule.mechanism,
            message=message,
            details=details,
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 2: HLB Balance
# ---------------------------------------------------------------------------

def check_hlb_balance(formula: Formula, kb: Knowledge) -> tuple[list[Issue], float | None]:
    """Check if the emulsifier system HLB matches the oil phase requirement."""
    issues: list[Issue] = []

    oils = []
    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        required_hlb = kb.oil_hlb.get(key)
        if required_hlb is not None:
            oils.append({
                "name": ing.inci_name,
                "required_hlb": required_hlb,
                "percentage": ing.percentage,
            })

    if not oils:
        return issues, None

    total_oil_pct = sum(o["percentage"] for o in oils)
    if total_oil_pct == 0:
        return issues, None

    weighted_required_hlb = sum(
        o["required_hlb"] * (o["percentage"] / total_oil_pct)
        for o in oils
    )

    issues.append(Issue(
        check="hlb",
        severity="info",
        message=(
            f"Oil phase requires HLB ~{weighted_required_hlb:.1f} for stable o/w emulsion. "
            f"Oils: {', '.join(o['name'] for o in oils)}. "
            f"Ensure your emulsifier system provides this HLB value."
        ),
        details={
            "required_hlb": round(weighted_required_hlb, 1),
            "oils": [{"name": o["name"], "required_hlb": o["required_hlb"], "pct": o["percentage"]} for o in oils],
        },
    ))

    return issues, None


# ---------------------------------------------------------------------------
# Check 3: Formula Sanity
# ---------------------------------------------------------------------------

def check_formula_sanity(formula: Formula) -> list[Issue]:
    """Basic sanity checks on the formula."""
    issues: list[Issue] = []

    total = formula.total_percentage
    if total > 101:
        issues.append(Issue(
            check="sanity",
            severity="error",
            message=f"Formula totals {total:.1f}% -- exceeds 100%. Check ingredient percentages.",
            details={"total_percentage": round(total, 1)},
        ))
    elif total < 95:
        issues.append(Issue(
            check="sanity",
            severity="warning",
            message=f"Formula totals only {total:.1f}%. Missing ingredients or water phase may need adjustment.",
            details={"total_percentage": round(total, 1)},
        ))

    seen = set()
    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        if key in seen:
            issues.append(Issue(
                check="sanity",
                severity="warning",
                ingredient=ing.inci_name,
                message=f"Duplicate ingredient: {ing.inci_name} appears more than once.",
            ))
        seen.add(key)

    return issues


# ---------------------------------------------------------------------------
# Check 4: Coverage Warning
# ---------------------------------------------------------------------------

def check_coverage(formula: Formula, kb: Knowledge) -> list[Issue]:
    """
    Warn if the formula's category has thin rule coverage.

    This is critical for honesty — a 100/100 score in a domain with
    zero rules is meaningless. The report should say so.
    """
    issues: list[Issue] = []

    category = formula.category
    if not category:
        return issues

    coverage = kb.coverage_summary(category)
    dedicated = coverage.get("dedicated_to_category", 0) or 0
    applicable = coverage.get("applicable_to_category", 0) or 0

    # Categories with zero dedicated rules
    thin_coverage_categories = {"pharma", "food", "materials"}

    if category in thin_coverage_categories or dedicated == 0:
        issues.append(Issue(
            check="coverage",
            severity="warning",
            message=(
                f"COVERAGE WARNING: Category '{category}' has {dedicated} dedicated rules "
                f"({applicable} total including cross-category). "
                f"This score reflects only available checks and may miss domain-specific "
                f"interactions. Do not rely solely on this score for {category} product development. "
                f"Contributions for this domain are welcome."
            ),
            details={
                "category": category,
                "dedicated_rules": dedicated,
                "applicable_rules": applicable,
                "total_rules": coverage["total_rules"],
            },
        ))
    elif dedicated < 10:
        issues.append(Issue(
            check="coverage",
            severity="info",
            message=(
                f"Category '{category}' has {dedicated} dedicated rules "
                f"({applicable} total). Coverage is growing. "
                f"Consider additional domain-specific review."
            ),
            details={
                "category": category,
                "dedicated_rules": dedicated,
                "applicable_rules": applicable,
            },
        ))

    return issues
