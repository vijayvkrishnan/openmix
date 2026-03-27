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
from openmix.matching import match_ingredient

ObserveMode = Literal["engineering", "discovery"]


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
                confidence=0.7 if r.log_p > 6.0 else 0.5,
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
                confidence=0.8,
            ))

        # Molecular weight observation
        mw = float(r.molecular_weight) if r.molecular_weight else None
        if mw and mw > 500:
            result.observations.append(Observation(
                category="molecular",
                subject=ing.inci_name,
                observed=f"MW {mw:.0f} Da — large molecule",
                expected="Large molecules have limited skin penetration (Lipinski: MW < 500)",
                agreement="uncertain",
                detail="May have limited bioavailability or penetration depending on application.",
                source="physics",
                confidence=0.5,
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

    # Preservative check
    preservative_names = {
        "PHENOXYETHANOL", "SODIUM BENZOATE", "POTASSIUM SORBATE",
        "BENZYL ALCOHOL", "ETHYLHEXYLGLYCERIN", "CAPRYLYL GLYCOL",
        "METHYLPARABEN", "PROPYLPARABEN", "SORBIC ACID",
        "DEHYDROACETIC ACID", "CHLORPHENESIN",
    }
    inci_upper = formula.inci_names_upper
    has_preservative = bool(inci_upper & preservative_names)
    has_water = any(n in inci_upper for n in ("WATER", "AQUA", "PURIFIED WATER"))

    if has_water and not has_preservative:
        result.observations.append(Observation(
            category="structural",
            subject="formula",
            observed="Water-based formula without detected preservative",
            expected="Water-based formulas require preservation against microbial growth",
            agreement="discrepancy",
            detail="Add a preservative system or confirm the formula has sufficient "
                   "antimicrobial protection through other means (low water activity, pH extremes).",
            source="structural",
            confidence=0.8,
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
            confidence=0.6,
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
            confidence=0.7,
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
            confidence=0.8,
        ))
