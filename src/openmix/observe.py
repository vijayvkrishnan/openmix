"""
Physics observation engine — the core of OpenMix evaluation.

Observes a formulation through the lens of molecular physics and domain
knowledge. Reports what it SEES, what it EXPECTED, and where they
disagree. Does not produce arbitrary scores.

Two modes use the same observations differently:
  Engineering: minimize concerns (optimize toward a stable formula)
  Discovery: investigate discrepancies (find where expectations are wrong)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from openmix.schema import Formula
from openmix.resolver import resolve, ResolvedIngredient
from openmix.knowledge.loader import load_knowledge, Knowledge
from openmix.knowledge.constants import (
    PRESERVATIVE_NAMES,
    SURFACTANT_CHARGE_DENSITY,
    Z_DANGER_LOW,
    Z_DANGER_HIGH,
    NONIONIC_SHIELDING_THRESHOLD,
    REDUCING_SUGAR_EXCIPIENTS,
    PEROXIDE_CONTAINING_EXCIPIENTS,
    ALKALINE_EXCIPIENTS,
    METAL_CONTAINING_EXCIPIENTS,
    FUNCTIONAL_GROUP_RISKS,
)
from openmix.knowledge.pka import load_pka_data, assess_ph_suitability, IngredientPKa
from openmix.matching import match_ingredient

ObserveMode = Literal["engineering", "discovery"]

# Confidence value scheme for observations.
# Each observation carries a confidence score (0-1) indicating how
# certain the observation engine is in its assessment. These values
# affect discourse classification: the gap between confidence levels
# determines whether a disagreement is a "correction" or "true disagreement."
#
# The scheme follows the evidence source:
#   1.0  -- Deterministic / arithmetic (percentages, duplicates)
#   0.9  -- Validated equation applied to literature data (Henderson-Hasselbalch
#           with published pKa, charge density from molecular weight)
#   0.8  -- Computational property from molecular structure (LogP confirmed
#           hydrophilic, well-established physics like charge ratio model)
#   0.7  -- Computational property with moderate uncertainty (LogP hydrophobic
#           concern, HLB estimate, knowledge base rule with cited source)
#   0.6  -- Heuristic or indirect inference (phase behavior estimate,
#           nonionic shielding model, coverage-based assessment)
#   0.5  -- Uncertain / low-confidence estimate (MW-based bioavailability,
#           weak LogP signal, initial discovery confidence)
CONF_DETERMINISTIC = 1.0
CONF_VALIDATED_EQUATION = 0.9
CONF_COMPUTATIONAL_STRONG = 0.8
CONF_COMPUTATIONAL_MODERATE = 0.7
CONF_HEURISTIC = 0.6
CONF_UNCERTAIN = 0.5


@dataclass
class Observation:
    """One physics observation about an ingredient or the formula."""
    category: str       # molecular, interaction, structural, phase, charge
    subject: str        # ingredient name or "formula"
    observed: str       # what we see
    expected: str       # what physics/rules predict
    agreement: str      # "confirmed", "uncertain", "discrepancy"
    detail: str         # human-readable explanation
    source: str         # "physics", "knowledge_base", "structural"
    confidence: float   # 0-1, how confident we are in the expectation


@dataclass
class Violation:
    """A known dangerous interaction detected."""
    severity: str       # "hard" or "soft"
    ingredients: list[str]
    mechanism: str
    message: str
    confidence: float
    source: str


@dataclass
class FormulationObservation:
    """
    Complete physics assessment of a formula.

    Not a score. A structured set of observations that the agent
    reads and reasons about. Mode determines interpretation:

      engineering — minimize concerns (build stable products)
      discovery   — investigate discrepancies (find where expectations are wrong)
    """
    formula_name: Optional[str] = None
    mode: ObserveMode = "engineering"
    observations: list[Observation] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    resolved_ingredients: dict[str, ResolvedIngredient] = field(default_factory=dict)
    resolution_rate: float = 0.0

    @property
    def hard_violations(self) -> int:
        return sum(1 for v in self.violations if v.severity == "hard")

    @property
    def soft_violations(self) -> int:
        return sum(1 for v in self.violations if v.severity == "soft")

    @property
    def concerns(self) -> list[Observation]:
        """Observations where physics suggests a potential issue."""
        return [o for o in self.observations if o.agreement == "discrepancy"]

    @property
    def signals(self) -> list[Violation]:
        """Soft violations — interesting interactions, not safety blockers."""
        return [v for v in self.violations if v.severity == "soft"]

    @property
    def discoveries(self) -> list[Observation]:
        """Low-confidence discrepancies — where our expectations may be wrong.

        These are the most interesting observations for discovery mode:
        the physics says X should happen, but the expectation confidence
        is low, meaning the knowledge base might be incomplete.
        """
        return [o for o in self.observations
                if o.agreement == "discrepancy" and o.confidence < 0.7]

    @property
    def concern_count(self) -> float:
        """Optimization signal — interpretation depends on mode.

        Engineering: hard violations + physics concerns + soft violations.
          Goal: minimize to zero.
        Discovery: hard violations only.
          Soft violations and discrepancies are signals to investigate, not fix.
        """
        if self.mode == "discovery":
            return float(self.hard_violations * 10)
        return (
            self.hard_violations * 10
            + len(self.concerns)
            + sum(v.confidence for v in self.violations if v.severity == "soft")
        )

    @property
    def concern_score(self) -> float:
        """0-100 score derived from concerns. 100 = no concerns. For backward compatibility."""
        raw = self.hard_violations * 25 + len(self.concerns) * 5 + self.soft_violations * 8
        return round(max(0, 100 - raw), 1)

    def __str__(self) -> str:
        lines = []
        lines.append(f"Physics Observation ({self.mode}): "
                     f"{self.formula_name or 'unnamed'}")
        lines.append(f"Resolved: {self.resolution_rate:.0%} of ingredients")
        lines.append("")

        if self.violations:
            lines.append(f"Violations ({self.hard_violations} hard, "
                        f"{self.soft_violations} soft):")
            for v in self.violations:
                tag = "HARD" if v.severity == "hard" else f"SOFT (conf {v.confidence:.1f})"
                lines.append(f"  [{tag}] {' + '.join(v.ingredients)}")
                lines.append(f"    {v.message}")
            lines.append("")

        # Group observations by category
        by_cat: dict[str, list[Observation]] = {}
        for obs in self.observations:
            by_cat.setdefault(obs.category, []).append(obs)

        for cat, obs_list in by_cat.items():
            lines.append(f"{cat.upper()}:")
            for obs in obs_list:
                icon = {"confirmed": " ", "uncertain": "?",
                        "discrepancy": "!"}[obs.agreement]
                lines.append(f"  [{icon}] {obs.subject}: {obs.observed}")
                if obs.agreement == "discrepancy":
                    lines.append(f"      Expected: {obs.expected}")
                    if obs.detail:
                        lines.append(f"      {obs.detail}")
            lines.append("")

        if self.mode == "discovery":
            lines.append(
                f"Hard violations: {self.hard_violations}  |  "
                f"Signals: {len(self.signals)}  |  "
                f"Knowledge gaps: {len(self.discoveries)}")
        else:
            lines.append(f"Concern count: {self.concern_count} "
                         f"(lower = better, 0 = no concerns)")

        return "\n".join(lines)


def observe(
    formula: Formula,
    knowledge: Knowledge | None = None,
    mode: ObserveMode = "engineering",
) -> FormulationObservation:
    """
    Observe a formulation through physics and domain knowledge.

    Returns structured observations, not a score.
    The agent reads these and decides what to do.

    Modes:
      engineering — minimize concerns, build stable products
      discovery   — investigate discrepancies, find where expectations are wrong
    """
    kb = knowledge or load_knowledge()
    result = FormulationObservation(formula_name=formula.name, mode=mode)

    # Resolve all ingredients
    for ing in formula.ingredients:
        resolved = resolve(ing.inci_name)
        result.resolved_ingredients[ing.inci_name] = resolved

    n_resolved = sum(1 for r in result.resolved_ingredients.values() if r.resolved)
    result.resolution_rate = n_resolved / len(formula.ingredients) if formula.ingredients else 0

    # Phase 1: Knowledge base violations
    _check_violations(formula, kb, result)

    # Phase 2: Molecular observations
    _observe_molecular(formula, result)

    # Phase 3: Structural observations
    _observe_structural(formula, result)

    # Phase 4: Phase / emulsion observations
    _observe_phase(formula, kb, result)

    # Phase 5: Charge observations
    _observe_charge(formula, result)

    # Phase 6: pH-ionization observations (Henderson-Hasselbalsh)
    if formula.target_ph is not None:
        _observe_ph(formula, result)

    # Phase 7: Surfactant charge balance (molar charge ratio)
    _observe_surfactant_charge(formula, result)

    # Phase 8: Mechanism-based drug-excipient interaction prediction
    if formula.category == "pharma":
        _observe_pharma_mechanisms(formula, result)

    return result


def _check_violations(formula: Formula, kb: Knowledge,
                       result: FormulationObservation):
    """Check knowledge base interaction rules."""
    inci_set = formula.inci_names_upper

    for rule in kb.interaction_rules:
        a_match = match_ingredient(rule.a, inci_set, kb.aliases)
        b_match = match_ingredient(rule.b, inci_set, kb.aliases)

        if not a_match or not b_match or a_match == b_match:
            continue

        result.violations.append(Violation(
            severity=rule.rule_type,
            ingredients=[a_match, b_match],
            mechanism=rule.mechanism,
            message=rule.message,
            confidence=rule.confidence,
            source=rule.source,
        ))


def _observe_molecular(formula: Formula, result: FormulationObservation):
    """Observe molecular properties — LogP, MW, solubility implications."""
    for ing in formula.ingredients:
        r = result.resolved_ingredients.get(ing.inci_name)
        if not r or not r.resolved or r.log_p is None:
            continue

        # LogP observation — hydrophobicity
        if r.log_p > 5.0 and ing.percentage > 2.0:
            result.observations.append(Observation(
                category="molecular",
                subject=ing.inci_name,
                observed=f"LogP {r.log_p:.1f} at {ing.percentage:.1f}% — hydrophobic",
                expected="Hydrophobic ingredients in aqueous systems need solubilization",
                agreement="discrepancy" if ing.percentage > 5.0 else "uncertain",
                detail=f"LogP {r.log_p:.1f} suggests poor water solubility. "
                       f"At {ing.percentage:.1f}%, ensure adequate emulsifier or solubilizer.",
                source="physics",
                confidence=CONF_COMPUTATIONAL_MODERATE if r.log_p > 6.0 else CONF_UNCERTAIN,
            ))
        elif r.log_p is not None and r.log_p < -2.0:
            result.observations.append(Observation(
                category="molecular",
                subject=ing.inci_name,
                observed=f"LogP {r.log_p:.1f} — very hydrophilic",
                expected="Highly hydrophilic ingredients dissolve readily in water",
                agreement="confirmed",
                detail="Good water solubility expected.",
                source="physics",
                confidence=CONF_COMPUTATIONAL_STRONG,
            ))

        # Molecular weight observation.
        # Lipinski MW < 500 applies to drug candidates for oral absorption
        # and topical penetration -- not relevant for pharma excipients
        # or non-drug formulation ingredients.
        mw = float(r.molecular_weight) if r.molecular_weight else None
        if mw and mw > 500 and formula.category != "pharma":
            result.observations.append(Observation(
                category="molecular",
                subject=ing.inci_name,
                observed=f"MW {mw:.0f} Da — large molecule",
                expected="Large molecules have limited skin/membrane penetration (Lipinski: MW < 500)",
                agreement="uncertain",
                detail="May have limited bioavailability or penetration depending on application.",
                source="physics",
                confidence=CONF_UNCERTAIN,
            ))


def _observe_structural(formula: Formula, result: FormulationObservation):
    """Observe structural properties — totals, preservatives, pH adjusters."""
    total = formula.total_percentage

    if total > 101 or total < 99:
        result.observations.append(Observation(
            category="structural",
            subject="formula",
            observed=f"Total: {total:.1f}%",
            expected="Formulations should total 100% (±1%)",
            agreement="discrepancy",
            detail="Adjust water/base phase to reach 100%.",
            source="structural",
            confidence=1.0,
        ))
    else:
        result.observations.append(Observation(
            category="structural",
            subject="formula",
            observed=f"Total: {total:.1f}%",
            expected="100%",
            agreement="confirmed",
            detail="",
            source="structural",
            confidence=1.0,
        ))

    # Preservative check (skip for solid dosage forms -- water is a processing aid, not final product)
    is_solid_dosage = formula.category == "pharma" and formula.product_type in (
        "tablet", "capsule", "powder", "granule", None,
    )
    inci_upper = formula.inci_names_upper
    has_preservative = bool(inci_upper & PRESERVATIVE_NAMES)
    has_water = any(n in inci_upper for n in ("WATER", "AQUA", "PURIFIED WATER"))

    if has_water and not has_preservative and not is_solid_dosage:
        result.observations.append(Observation(
            category="structural",
            subject="formula",
            observed="Water-based formula without detected preservative",
            expected="Water-based formulas require preservation against microbial growth",
            agreement="discrepancy",
            detail="Add a preservative system or confirm the formula has sufficient "
                   "antimicrobial protection through other means (low water activity, pH extremes).",
            source="structural",
            confidence=CONF_COMPUTATIONAL_STRONG,
        ))

    # Duplicate check
    seen = set()
    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        if key in seen:
            result.observations.append(Observation(
                category="structural",
                subject=ing.inci_name,
                observed="Appears more than once",
                expected="Each ingredient should appear once",
                agreement="discrepancy",
                detail="Combine into a single entry.",
                source="structural",
                confidence=1.0,
            ))
        seen.add(key)


def _observe_phase(formula: Formula, kb: Knowledge,
                    result: FormulationObservation):
    """Observe phase behavior — oil/water distribution, HLB needs."""
    hydrophobic = []
    for ing in formula.ingredients:
        r = result.resolved_ingredients.get(ing.inci_name)
        key = ing.inci_name.upper().strip()

        is_known_oil = key in kb.oil_hlb
        is_logp_hydrophobic = r and r.log_p is not None and r.log_p > 4.0

        if (is_known_oil or is_logp_hydrophobic) and ing.percentage > 1.0:
            hydrophobic.append((ing.inci_name, ing.percentage,
                                r.log_p if r else None))

    if hydrophobic:
        total_hydrophobic = sum(p for _, p, _ in hydrophobic)
        names = [f"{n} ({p:.1f}%)" for n, p, _ in hydrophobic]

        result.observations.append(Observation(
            category="phase",
            subject="formula",
            observed=f"Hydrophobic phase: {total_hydrophobic:.1f}% — {', '.join(names)}",
            expected="Oil-phase ingredients require emulsification in aqueous systems",
            agreement="uncertain",
            detail=f"Ensure adequate emulsifier for {total_hydrophobic:.1f}% oil phase.",
            source="physics",
            confidence=CONF_HEURISTIC,
        ))


def _observe_charge(formula: Formula, result: FormulationObservation):
    """Observe charge balance from resolved molecular properties."""
    anionics = []
    cationics = []

    for ing in formula.ingredients:
        r = result.resolved_ingredients.get(ing.inci_name)
        if not r or not r.charge_type:
            continue
        if r.charge_type == "anionic" and ing.percentage > 0.5:
            anionics.append(f"{ing.inci_name} ({ing.percentage:.1f}%)")
        elif r.charge_type == "cationic" and ing.percentage > 0.5:
            cationics.append(f"{ing.inci_name} ({ing.percentage:.1f}%)")

    if anionics and cationics:
        result.observations.append(Observation(
            category="charge",
            subject="formula",
            observed=f"Anionic: {', '.join(anionics)} — Cationic: {', '.join(cationics)}",
            expected="Mixing anionic and cationic species can cause precipitation",
            agreement="discrepancy",
            detail="Check whether these species interact at the given concentrations. "
                   "Amphoteric surfactants or nonionic alternatives may be needed.",
            source="physics",
            confidence=CONF_COMPUTATIONAL_MODERATE,
        ))
    elif anionics or cationics:
        charge_type = "anionic" if anionics else "cationic"
        result.observations.append(Observation(
            category="charge",
            subject="formula",
            observed=f"Uniformly {charge_type} charged species",
            expected="No charge conflicts expected",
            agreement="confirmed",
            detail="",
            source="physics",
            confidence=CONF_COMPUTATIONAL_STRONG,
        ))


# Module-level pKa data cache
_pka_data: dict[str, IngredientPKa] | None = None


def _get_pka_data() -> dict[str, IngredientPKa]:
    global _pka_data
    if _pka_data is None:
        _pka_data = load_pka_data()
    return _pka_data


def _observe_ph(formula: Formula, result: FormulationObservation):
    """Observe pH-ionization behavior via Henderson-Hasselbalch.

    For each ingredient with known pKa data, compute the ionization
    fraction at the formula's target pH and report whether the pH
    is suitable for that ingredient's intended function.
    """
    pka_db = _get_pka_data()
    if not pka_db or formula.target_ph is None:
        return

    target_ph = formula.target_ph

    for ing in formula.ingredients:
        key = ing.inci_name.upper().strip()
        pka_entry = pka_db.get(key)
        if not pka_entry:
            continue

        assessment = assess_ph_suitability(pka_entry, target_ph)
        if assessment["ionized_fraction"] is None:
            continue

        if assessment["suitable"]:
            result.observations.append(Observation(
                category="ph",
                subject=ing.inci_name,
                observed=assessment["detail"],
                expected=(
                    f"Optimal pH {pka_entry.optimal_ph_min}-"
                    f"{pka_entry.optimal_ph_max} for {ing.inci_name}"
                ),
                agreement="confirmed",
                detail="",
                source=f"Henderson-Hasselbalch, pKa {pka_entry.pka[0]}. "
                       f"{pka_entry.source}",
                confidence=CONF_VALIDATED_EQUATION,
            ))
        else:
            result.observations.append(Observation(
                category="ph",
                subject=ing.inci_name,
                observed=assessment["detail"],
                expected=(
                    f"Optimal pH {pka_entry.optimal_ph_min}-"
                    f"{pka_entry.optimal_ph_max} for {ing.inci_name}"
                ),
                agreement="discrepancy",
                detail="",
                source=f"Henderson-Hasselbalch, pKa {pka_entry.pka[0]}. "
                       f"{pka_entry.source}",
                confidence=CONF_VALIDATED_EQUATION,
            ))


def _observe_surfactant_charge(formula: Formula, result: FormulationObservation):
    """Observe surfactant charge balance using molar charge density.

    Computes the charge ratio Z = cationic_charge / anionic_charge using
    literature charge densities (meq/g). When Z approaches 1.0, the system
    is at risk of coacervation or precipitation.

    Sources: Wang & Dubin Langmuir 2023, Thompson Macromol. Chem. Phys. 2023.
    """
    total_cationic = 0.0   # meq per 100g formula
    total_anionic = 0.0
    total_nonionic_pct = 0.0
    total_surfactant_pct = 0.0
    cationic_species = []
    anionic_species = []

    for ing in formula.ingredients:
        name_upper = ing.inci_name.upper().strip()
        cd = SURFACTANT_CHARGE_DENSITY.get(name_upper)
        if cd is None:
            continue

        total_surfactant_pct += ing.percentage

        if cd > 0:
            charge_meq = cd * ing.percentage  # meq per 100g
            total_cationic += charge_meq
            cationic_species.append((ing.inci_name, ing.percentage, cd))
        elif cd < 0:
            charge_meq = abs(cd) * ing.percentage
            total_anionic += charge_meq
            anionic_species.append((ing.inci_name, ing.percentage, abs(cd)))
        else:
            total_nonionic_pct += ing.percentage

    # Only produce observations if both cationic and anionic are present
    if not cationic_species or not anionic_species:
        return

    z_ratio = total_cationic / (total_anionic + 1e-8)
    in_danger = Z_DANGER_LOW <= z_ratio <= Z_DANGER_HIGH

    cat_detail = ", ".join(
        f"{n} ({p:.1f}% x {cd:.1f} meq/g)" for n, p, cd in cationic_species
    )
    an_detail = ", ".join(
        f"{n} ({p:.1f}% x {cd:.1f} meq/g)" for n, p, cd in anionic_species
    )

    if in_danger:
        result.observations.append(Observation(
            category="charge",
            subject="formula",
            observed=f"Charge ratio Z = {z_ratio:.2f} "
                     f"(cationic: {total_cationic:.1f} meq, anionic: {total_anionic:.1f} meq)",
            expected=f"Z in range {Z_DANGER_LOW}-{Z_DANGER_HIGH} indicates "
                     "coacervation/precipitation risk (charge neutralization zone)",
            agreement="discrepancy",
            detail=f"Cationic: {cat_detail}. Anionic: {an_detail}. "
                   f"Near charge neutralization (Z~1) promotes phase separation.",
            source="Wang & Dubin Langmuir 2023, charge density model",
            confidence=CONF_COMPUTATIONAL_STRONG,
        ))
    else:
        result.observations.append(Observation(
            category="charge",
            subject="formula",
            observed=f"Charge ratio Z = {z_ratio:.2f} "
                     f"(cationic: {total_cationic:.1f} meq, anionic: {total_anionic:.1f} meq)",
            expected=f"Z outside danger zone ({Z_DANGER_LOW}-{Z_DANGER_HIGH}), "
                     "charge imbalance favors single-phase stability",
            agreement="confirmed",
            detail=f"Cationic: {cat_detail}. Anionic: {an_detail}.",
            source="Wang & Dubin Langmuir 2023, charge density model",
            confidence=CONF_COMPUTATIONAL_MODERATE,
        ))

    # Nonionic shielding assessment
    if total_surfactant_pct > 0:
        nonionic_fraction = total_nonionic_pct / total_surfactant_pct

        if nonionic_fraction < NONIONIC_SHIELDING_THRESHOLD and in_danger:
            result.observations.append(Observation(
                category="charge",
                subject="formula",
                observed=f"Nonionic shielding: {nonionic_fraction:.0%} of surfactant blend",
                expected=f">{NONIONIC_SHIELDING_THRESHOLD:.0%} nonionic surfactant "
                         "reduces precipitation risk via mixed micelle formation",
                agreement="discrepancy",
                detail="Nonionic surfactants form mixed micelles with anionic surfactants, "
                       "reducing free anionic monomer available for cationic complexation.",
                source="Soontravanich 2010, J. Surfactants Detergents",
                confidence=CONF_HEURISTIC,
            ))
        elif nonionic_fraction >= NONIONIC_SHIELDING_THRESHOLD:
            result.observations.append(Observation(
                category="charge",
                subject="formula",
                observed=f"Nonionic shielding: {nonionic_fraction:.0%} of surfactant blend",
                expected=f">{NONIONIC_SHIELDING_THRESHOLD:.0%} nonionic fraction "
                         "provides significant steric stabilization",
                agreement="confirmed",
                detail="Adequate nonionic surfactant to shield anionic-cationic interactions.",
                source="Soontravanich 2010, J. Surfactants Detergents",
                confidence=CONF_COMPUTATIONAL_MODERATE,
            ))


def _observe_pharma_mechanisms(formula: Formula, result: FormulationObservation):
    """Mechanism-based drug-excipient interaction prediction.

    Detects reactive functional groups from SMILES and checks whether
    any excipients in the formula have properties that trigger known
    degradation mechanisms. Works for ANY drug with detectable functional
    groups, not just drugs in the knowledge base rules.

    Example: a novel amine drug + lactose → Maillard risk, even if this
    specific pair has never been seen before.
    """
    try:
        from openmix.molecular import detect_functional_groups, RDKIT_AVAILABLE
    except ImportError:
        return
    if not RDKIT_AVAILABLE:
        return

    # Classify excipients present in the formula
    inci_upper = formula.inci_names_upper
    has_reducing_sugar = bool(inci_upper & REDUCING_SUGAR_EXCIPIENTS)
    has_peroxide = bool(inci_upper & PEROXIDE_CONTAINING_EXCIPIENTS)
    has_alkaline = bool(inci_upper & ALKALINE_EXCIPIENTS)
    has_metal = bool(inci_upper & set(METAL_CONTAINING_EXCIPIENTS.keys()))

    excipient_classes = {
        "reducing_sugar": has_reducing_sugar,
        "peroxide": has_peroxide,
        "alkaline": has_alkaline,
        "metal": has_metal,
    }

    # For each ingredient with resolved SMILES, detect functional groups.
    # If the seed cache resolved without SMILES, try PubChem directly
    # since functional group detection requires molecular structure.
    for ing in formula.ingredients:
        resolved = result.resolved_ingredients.get(ing.inci_name)
        if not resolved:
            continue

        smiles = resolved.smiles
        if not smiles:
            from openmix.resolver.pubchem import lookup_pubchem
            pubchem_data = lookup_pubchem(ing.inci_name)
            if pubchem_data:
                smiles = pubchem_data.get("smiles")
        if not smiles:
            continue

        groups = detect_functional_groups(smiles)
        if not groups:
            continue

        detected = [g for g, present in groups.items() if present]
        if not detected:
            continue

        # Check each detected group against excipient classes present
        for group_name in detected:
            risks = FUNCTIONAL_GROUP_RISKS.get(group_name, [])
            for risk in risks:
                exc_class = risk["excipient_class"]
                if not excipient_classes.get(exc_class, False):
                    continue

                # Find the specific excipient(s) that trigger this risk
                if exc_class == "reducing_sugar":
                    triggering = inci_upper & REDUCING_SUGAR_EXCIPIENTS
                elif exc_class == "peroxide":
                    triggering = inci_upper & PEROXIDE_CONTAINING_EXCIPIENTS
                elif exc_class == "alkaline":
                    triggering = inci_upper & ALKALINE_EXCIPIENTS
                elif exc_class == "metal":
                    triggering = inci_upper & set(METAL_CONTAINING_EXCIPIENTS.keys())
                else:
                    triggering = set()

                trigger_str = ", ".join(sorted(triggering))

                result.observations.append(Observation(
                    category="interaction",
                    subject=f"{ing.inci_name} + {trigger_str}",
                    observed=f"{ing.inci_name} contains {group_name.replace('_', ' ')} "
                             f"group (detected from SMILES)",
                    expected=f"{risk['mechanism']} risk with {exc_class.replace('_', ' ')} "
                             f"excipients ({trigger_str})",
                    agreement="discrepancy",
                    detail=risk["detail"],
                    source=f"mechanism-based prediction: {group_name} + {exc_class}",
                    confidence=risk["confidence"],
                ))
