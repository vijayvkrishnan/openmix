"""
Heuristic stability scoring for formulations.

Returns a deterministic 0-100 score decomposed into sub-scores:
compatibility, pH suitability, emulsion balance, formula integrity,
and system completeness. Same formula always produces the same score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openmix.schema import Formula
from openmix.knowledge.loader import Knowledge, load_knowledge
from openmix.knowledge.constants import PRESERVATIVE_NAMES
from openmix.matching import match_ingredient


# Sub-score weight allocation (total = 100).
# These weights are DESIGN CHOICES, not empirically calibrated values.
# They reflect the relative importance of each factor in a formulation
# stability assessment, based on formulation science priorities:
#   - Compatibility is weighted highest because a dangerous interaction
#     (e.g., toxic gas formation) overrides all other considerations.
#   - pH suitability is second because incorrect pH degrades active
#     ingredients and disrupts preservative efficacy.
#   - Emulsion balance is third because phase separation is the most
#     common stability failure mode in emulsion systems.
#   - Integrity and completeness are lower because they are structural
#     checks (percentages, duplicates) that are easy to fix.
#
# Future work: calibrate these weights against real stability outcomes
# (e.g., accelerated stability testing results).
SCORE_COMPATIBILITY = 35
SCORE_PH = 25
SCORE_EMULSION = 20
SCORE_INTEGRITY = 10
SCORE_COMPLETENESS = 10


@dataclass
class StabilityScore:
    """Quantitative stability prediction with decomposed sub-scores."""

    total: float = 0.0

    compatibility: float = 0.0      # 0-35: no dangerous interactions
    ph_suitability: float = 0.0     # 0-25: ingredients work at target pH
    emulsion_balance: float = 0.0   # 0-20: HLB system matched
    formula_integrity: float = 0.0  # 0-10: percentages sum to 100%, no dupes
    system_completeness: float = 0.0  # 0-10: preservative present, sensible count

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
# Optimal pH ranges for ingredient stability and efficacy.
#
# These ranges represent the pH window where each ingredient is stable,
# effective, and compatible with typical formulation systems. Outside
# these ranges, the ingredient may degrade, lose efficacy, or cause
# formulation instability.
#
# Sources:
#   Acids (AHAs/BHAs): Kornhauser et al., Clin Cosmet Investig Dermatol 2010; Tang &
#     Yang, JAAD 2000. Efficacy requires free acid form (below pKa).
#   Ascorbic acid: Telang, Indian Dermatol Online J 2013; Farris, Dermatol
#     Surg 2005. Stable below pH 3.5; oxidizes rapidly above pH 4.
#   Niacinamide: Gehring, Dermatol Ther 2004. Hydrolyzes to nicotinic acid
#     below pH 4; stable at pH 5-7.
#   Retinol/retinal: Temova Rakusa et al., J Cosmet Dermatol 2021. Stable at
#     slightly acidic to neutral pH; degrades in strongly acidic conditions.
#   Preservatives: Steinberg, Cosm & Toil 2006. Ranges reflect efficacy
#     windows (e.g., benzoic acid/sorbic acid require pH < 5 for ionization).
#   Carbomer: Lubrizol technical bulletin TDS-237. Thickens only when
#     neutralized (pH > 5); degrades above pH 9.
#   Hyaluronic acid: Essendoubi et al., Biopolymers 2011. Acid-catalyzed
#     hydrolysis below pH 4.
# ---------------------------------------------------------------------------

PH_RANGES: dict[str, tuple[float, float]] = {
    # Actives -- ranges where the ingredient is effective and stable
    "ASCORBIC ACID": (2.0, 3.5),             # must be free acid; oxidizes above pH 4
    "L-ASCORBIC ACID": (2.0, 3.5),
    "SODIUM ASCORBYL PHOSPHATE": (5.0, 7.5), # phosphate ester, stable at neutral pH
    "NIACINAMIDE": (5.0, 7.0),               # hydrolyzes to nicotinic acid below pH 4
    "RETINOL": (5.5, 6.5),                   # acid-labile; degrades below pH 5
    "RETINAL": (5.0, 6.5),
    "GLYCOLIC ACID": (3.0, 4.0),             # pKa 3.83; needs free acid form
    "LACTIC ACID": (3.5, 4.5),               # pKa 3.86
    "SALICYLIC ACID": (2.5, 4.0),            # pKa 2.97
    "MANDELIC ACID": (3.0, 4.0),             # pKa 3.41
    "AZELAIC ACID": (4.0, 5.0),              # pKa1 4.55
    "KOJIC ACID": (4.0, 5.5),               # pKa ~7.9; stable in acidic conditions
    "ARBUTIN": (5.0, 7.0),                   # hydrolyzes in strongly acidic conditions
    "TRANEXAMIC ACID": (5.0, 7.0),
    "COPPER TRIPEPTIDE-1": (4.5, 6.5),       # Cu2+ precipitates above pH 7
    "BAKUCHIOL": (5.0, 7.0),
    # Preservatives -- ranges where antimicrobial efficacy is maintained
    "PHENOXYETHANOL": (3.0, 8.0),            # effective across wide pH range
    "SODIUM BENZOATE": (2.0, 5.0),           # benzoic acid pKa 4.2; needs free acid
    "POTASSIUM SORBATE": (2.0, 6.0),         # sorbic acid pKa 4.76
    "BENZYL ALCOHOL": (3.0, 8.0),            # pH-independent mechanism
    # Thickeners / polymers
    "CARBOMER": (5.0, 9.0),                  # requires neutralization; degrades above pH 9
    "XANTHAN GUM": (3.0, 12.0),              # stable across wide pH range
    "HYDROXYETHYLCELLULOSE": (2.0, 12.0),    # pH-insensitive
    # Humectants
    "SODIUM HYALURONATE": (4.0, 8.0),        # acid hydrolysis below pH 4
    "HYALURONIC ACID": (4.0, 7.0),           # lower MW more susceptible to degradation
}

PRESERVATIVES: set[str] = PRESERVATIVE_NAMES | {
    "DMDM HYDANTOIN", "IMIDAZOLIDINYL UREA", "BENZISOTHIAZOLINONE",
    "METHYLISOTHIAZOLINONE", "LEVULINIC ACID", "P-ANISIC ACID",
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
