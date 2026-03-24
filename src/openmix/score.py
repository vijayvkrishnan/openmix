"""
Heuristic stability scoring for formulations.

Returns a deterministic 0-100 score decomposed into sub-scores:
compatibility, pH suitability, emulsion balance, formula integrity,
and system completeness. Same formula always produces the same score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from openmix.schema import Formula, Issue
from openmix.knowledge.loader import Knowledge, InteractionRule, load_knowledge
from openmix.matching import match_ingredient


@dataclass
class StabilityScore:
    """Quantitative stability prediction with decomposed sub-scores."""

    total: float = 0.0            # 0-100 composite score

    compatibility: float = 0.0    # 0-35 pts: no dangerous interactions
    ph_suitability: float = 0.0   # 0-25 pts: ingredients work at target pH
    emulsion_balance: float = 0.0 # 0-20 pts: HLB system matched
    formula_integrity: float = 0.0 # 0-10 pts: percentages, no dupes
    system_completeness: float = 0.0 # 0-10 pts: preservative, sensible count

    penalties: list[str] = field(default_factory=list)
    bonuses: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Stability Score: {self.total:.1f}/100",
            f"  Compatibility:      {self.compatibility:.1f}/35",
            f"  pH Suitability:     {self.ph_suitability:.1f}/25",
            f"  Emulsion Balance:   {self.emulsion_balance:.1f}/20",
            f"  Formula Integrity:  {self.formula_integrity:.1f}/10",
            f"  System Completeness:{self.system_completeness:.1f}/10",
        ]
        if self.penalties:
            lines.append("  Penalties:")
            for p in self.penalties:
                lines.append(f"    - {p}")
        if self.bonuses:
            lines.append("  Bonuses:")
            for b in self.bonuses:
                lines.append(f"    + {b}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Known pH ranges for common ingredient classes
# ---------------------------------------------------------------------------

PH_RANGES: dict[str, tuple[float, float]] = {
    # Actives
    "ASCORBIC ACID": (2.0, 3.5),
    "L-ASCORBIC ACID": (2.0, 3.5),
    "SODIUM ASCORBYL PHOSPHATE": (5.0, 7.5),
    "NIACINAMIDE": (5.0, 7.0),
    "RETINOL": (5.5, 6.5),
    "RETINAL": (5.0, 6.5),
    "GLYCOLIC ACID": (3.0, 4.0),
    "LACTIC ACID": (3.5, 4.5),
    "SALICYLIC ACID": (2.5, 4.0),
    "MANDELIC ACID": (3.0, 4.0),
    "AZELAIC ACID": (4.0, 5.0),
    "KOJIC ACID": (4.0, 5.5),
    "ARBUTIN": (5.0, 7.0),
    "TRANEXAMIC ACID": (5.0, 7.0),
    "COPPER TRIPEPTIDE-1": (4.5, 6.5),
    "BAKUCHIOL": (5.0, 7.0),
    # Preservatives
    "PHENOXYETHANOL": (3.0, 8.0),
    "SODIUM BENZOATE": (2.0, 5.0),
    "POTASSIUM SORBATE": (2.0, 6.0),
    "BENZYL ALCOHOL": (3.0, 8.0),
    # Thickeners / polymers
    "CARBOMER": (5.0, 9.0),
    "XANTHAN GUM": (3.0, 12.0),
    "HYDROXYETHYLCELLULOSE": (2.0, 12.0),
    # Humectants
    "SODIUM HYALURONATE": (4.0, 8.0),
    "HYALURONIC ACID": (4.0, 7.0),
}

# Known preservative INCI names (any match = preservative system present)
PRESERVATIVES: set[str] = {
    "PHENOXYETHANOL", "SODIUM BENZOATE", "POTASSIUM SORBATE",
    "BENZYL ALCOHOL", "ETHYLHEXYLGLYCERIN", "CAPRYLYL GLYCOL",
    "METHYLPARABEN", "PROPYLPARABEN", "DMDM HYDANTOIN",
    "IMIDAZOLIDINYL UREA", "BENZISOTHIAZOLINONE",
    "METHYLISOTHIAZOLINONE", "CHLORPHENESIN", "DEHYDROACETIC ACID",
    "SORBIC ACID", "LEVULINIC ACID", "P-ANISIC ACID",
}


def score(
    formula: Formula,
    knowledge: Knowledge | None = None,
) -> StabilityScore:
    """
    Compute a quantitative stability prediction for a formulation.

    Returns a StabilityScore with a total 0-100 and decomposed sub-scores.
    This is deterministic — same formula always gets the same score.
    """
    kb = knowledge or load_knowledge()
    result = StabilityScore()

    result.compatibility = _score_compatibility(formula, kb, result)
    result.ph_suitability = _score_ph(formula, result)
    result.emulsion_balance = _score_hlb(formula, kb, result)
    result.formula_integrity = _score_integrity(formula, result)
    result.system_completeness = _score_completeness(formula, result)

    result.total = (
        result.compatibility
        + result.ph_suitability
        + result.emulsion_balance
        + result.formula_integrity
        + result.system_completeness
    )
    result.total = round(max(0, min(100, result.total)), 1)

    return result


# ---------------------------------------------------------------------------
# Sub-score: Compatibility (0-35)
# ---------------------------------------------------------------------------

def _score_compatibility(formula: Formula, kb: Knowledge,
                         result: StabilityScore) -> float:
    pts = 35.0
    inci_set = formula.inci_names_upper

    for rule in kb.interaction_rules:
        a_match = match_ingredient(rule.a, inci_set, kb.aliases)
        b_match = match_ingredient(rule.b, inci_set, kb.aliases)

        if not a_match or not b_match or a_match == b_match:
            continue

        if rule.rule_type == "hard":
            penalty = 35.0  # instant zero on this sub-score
            pts -= penalty
            result.penalties.append(
                f"HARD: {a_match} + {b_match} ({rule.mechanism})")
        else:
            # Soft rules: penalty weighted by confidence
            penalty = 5.0 * rule.confidence
            pts -= penalty
            result.penalties.append(
                f"SOFT ({rule.confidence:.1f}): {a_match} + {b_match}")

    return round(max(0, pts), 1)


# ---------------------------------------------------------------------------
# Sub-score: pH Suitability (0-25)
# ---------------------------------------------------------------------------

def _score_ph(formula: Formula, result: StabilityScore) -> float:
    if formula.target_ph is None:
        return 12.5  # neutral — no pH specified

    pts = 25.0
    checked = 0
    in_range = 0

    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        ph_range = PH_RANGES.get(key)
        if ph_range is None:
            continue

        checked += 1
        ph_min, ph_max = ph_range

        if ph_min <= formula.target_ph <= ph_max:
            in_range += 1
        else:
            distance = min(abs(formula.target_ph - ph_min),
                          abs(formula.target_ph - ph_max))
            if distance > 2.0:
                pts -= 8
                result.penalties.append(
                    f"pH: {ing.inci_name} needs {ph_min}-{ph_max}, "
                    f"formula is {formula.target_ph}")
            elif distance > 1.0:
                pts -= 4
                result.penalties.append(
                    f"pH: {ing.inci_name} suboptimal at pH {formula.target_ph}")
            else:
                pts -= 2

    if checked > 0 and in_range == checked:
        result.bonuses.append(
            f"All {checked} pH-sensitive ingredients in optimal range")

    return round(max(0, pts), 1)


# ---------------------------------------------------------------------------
# Sub-score: Emulsion Balance (0-20)
# ---------------------------------------------------------------------------

def _score_hlb(formula: Formula, kb: Knowledge,
               result: StabilityScore) -> float:
    oils = []

    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        required = kb.oil_hlb.get(key)
        if required is not None:
            oils.append({"name": key, "hlb": required, "pct": ing.percentage})

    if not oils:
        return 20.0  # No oil phase — aqueous system, no emulsion needed

    total_oil_pct = sum(o["pct"] for o in oils)
    if total_oil_pct == 0:
        return 20.0

    required_hlb = sum(
        o["hlb"] * (o["pct"] / total_oil_pct) for o in oils)

    # Check if emulsifiers are present (by function or common names)
    emulsifier_names = {
        "POLYSORBATE 20", "POLYSORBATE 60", "POLYSORBATE 80",
        "SORBITAN OLEATE", "SORBITAN STEARATE",
        "CETEARETH-20", "CETETH-20", "STEARETH-20", "STEARETH-2",
        "PEG-100 STEARATE", "GLYCERYL STEARATE",
        "GLYCERYL STEARATE SE",
    }
    has_emulsifier = any(
        ing.inci_name.upper().strip() in emulsifier_names
        or (ing.function and "emulsif" in ing.function.lower())
        for ing in formula.ingredients
    )

    if not has_emulsifier and total_oil_pct > 3:
        result.penalties.append(
            f"Oil phase ({total_oil_pct:.0f}%) with no emulsifier detected")
        return 5.0

    # If emulsifier present but we can't compute system HLB,
    # give partial credit
    result.bonuses.append(
        f"Oil phase requires HLB ~{required_hlb:.1f}")
    return 14.0  # Partial — full scoring needs emulsifier HLB data


# ---------------------------------------------------------------------------
# Sub-score: Formula Integrity (0-10)
# ---------------------------------------------------------------------------

def _score_integrity(formula: Formula, result: StabilityScore) -> float:
    pts = 10.0
    total = formula.total_percentage

    if 99.0 <= total <= 101.0:
        pass  # Perfect
    elif 95.0 <= total <= 105.0:
        pts -= 3
        result.penalties.append(f"Percentages sum to {total:.1f}%")
    else:
        pts -= 8
        result.penalties.append(f"Percentages sum to {total:.1f}% (should be ~100)")

    # Check duplicates
    seen = set()
    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        if key in seen:
            pts -= 2
            result.penalties.append(f"Duplicate: {ing.inci_name}")
        seen.add(key)

    return round(max(0, pts), 1)


# ---------------------------------------------------------------------------
# Sub-score: System Completeness (0-10)
# ---------------------------------------------------------------------------

def _score_completeness(formula: Formula, result: StabilityScore) -> float:
    pts = 0.0
    inci_set = formula.inci_names_upper

    # Preservative system present?
    has_preservative = bool(inci_set & PRESERVATIVES)
    has_water = any(
        n in inci_set
        for n in ("WATER", "AQUA", "PURIFIED WATER", "DEIONIZED WATER")
    )

    if has_preservative:
        pts += 4
        result.bonuses.append("Preservative system present")
    elif has_water:
        result.penalties.append("Water-based formula without preservative")

    # Reasonable ingredient count (5-15 is typical)
    n = len(formula.ingredients)
    if 5 <= n <= 15:
        pts += 3
    elif 3 <= n <= 20:
        pts += 2
    else:
        pts += 1
        result.penalties.append(f"Unusual ingredient count: {n}")

    # Has a base/solvent
    if has_water or any(
        n in inci_set for n in (
            "PROPYLENE GLYCOL", "BUTYLENE GLYCOL", "PROPANEDIOL",
            "ETHANOL", "ISOPROPYL ALCOHOL",
        )
    ):
        pts += 3
    else:
        pts += 1

    return round(min(10, pts), 1)
