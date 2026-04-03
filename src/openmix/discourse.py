"""
Discourse engine -- multi-perspective formulation evaluation.

Multiple computational perspectives evaluate the same formulation.
The discourse engine identifies where they agree, where one corrects
another, where they genuinely disagree, and where nobody has information.

The key distinction:
  Correction:        Asymmetric evidence. The stronger side wins.
  True disagreement: Comparable evidence. Both sides are defensible.
                     These are worth investigating.
  Knowledge gap:     No perspective has enough information.

Evidence hierarchy (higher = stronger):
  1. LLM reasoning (weakest -- model said so)
  2. Heuristic (rule of thumb, common practice)
  3. Rule-based (curated knowledge base with cited sources)
  4. Computational (physics computation -- RDKit, Henderson-Hasselbalch)
  5. Empirical proxy (data from similar systems, commercial products)
  6. Empirical direct (measured lab data for this exact system)

A claim at level N can correct a claim at level N-2 or below.
Claims at adjacent levels (N vs N-1) are treated as true disagreements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from openmix.schema import Formula
from openmix.observe import observe, FormulationObservation
from openmix.validate import validate, ValidationReport
from openmix.protocol import Protocol

if TYPE_CHECKING:
    from openmix.memory import ExperimentMemory


# The minimum gap in evidence levels for a correction (vs true disagreement)
CORRECTION_THRESHOLD = 2


class EvidenceLevel(IntEnum):
    """Strength of evidence behind a claim. Higher = stronger."""
    LLM_REASONING = 1
    HEURISTIC = 2
    RULE_BASED = 3
    COMPUTATIONAL = 4
    EMPIRICAL_PROXY = 5
    EMPIRICAL_DIRECT = 6


@dataclass
class Claim:
    """A single evaluative claim from one perspective."""
    perspective: str        # "physics", "chemistry", "data", "process"
    subject: str            # ingredient name, pair, or "formula"
    position: str           # "safe", "concern", "violation", "unknown"
    detail: str             # human-readable explanation
    evidence: str           # what the claim is based on
    evidence_level: EvidenceLevel
    confidence: float       # 0-1, how certain this perspective is


@dataclass
class DiscourseTopic:
    """
    One topic where perspectives have been compared.

    A topic is a specific subject (ingredient, pair, or formula property)
    where at least one perspective made a claim.
    """
    subject: str
    claims: list[Claim]
    classification: str     # "agreement", "correction", "true_disagreement", "knowledge_gap"
    summary: str
    winning_claim: Claim | None = None  # for corrections: which claim wins

    def __str__(self) -> str:
        icons = {
            "agreement": "[=]",
            "correction": "[>]",
            "true_disagreement": "[?]",
            "knowledge_gap": "[.]",
        }
        icon = icons.get(self.classification, "[ ]")
        lines = [f"  {icon} {self.subject}: {self.summary}"]
        for claim in self.claims:
            level_name = claim.evidence_level.name.lower().replace("_", " ")
            lines.append(
                f"      {claim.perspective}: {claim.detail} "
                f"(evidence: {level_name}, confidence: {claim.confidence:.1f})"
            )
        return "\n".join(lines)


@dataclass
class Discourse:
    """
    Complete multi-perspective assessment of a formulation.

    Not a score. A structured record of where perspectives agree,
    where one corrects another, where they genuinely disagree, and
    where nobody knows.
    """
    formula_name: str
    topics: list[DiscourseTopic] = field(default_factory=list)

    @property
    def agreements(self) -> list[DiscourseTopic]:
        return [t for t in self.topics if t.classification == "agreement"]

    @property
    def corrections(self) -> list[DiscourseTopic]:
        return [t for t in self.topics if t.classification == "correction"]

    @property
    def true_disagreements(self) -> list[DiscourseTopic]:
        return [t for t in self.topics if t.classification == "true_disagreement"]

    @property
    def knowledge_gaps(self) -> list[DiscourseTopic]:
        return [t for t in self.topics if t.classification == "knowledge_gap"]

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 70,
            f"  MULTI-PERSPECTIVE DISCOURSE: {self.formula_name}",
            "=" * 70,
        ]

        if self.agreements:
            lines.append("")
            lines.append(f"  AGREEMENTS ({len(self.agreements)}):")
            for topic in self.agreements:
                lines.append(str(topic))

        if self.corrections:
            lines.append("")
            lines.append(f"  CORRECTIONS ({len(self.corrections)}):")
            for topic in self.corrections:
                lines.append(str(topic))

        if self.true_disagreements:
            lines.append("")
            lines.append(f"  TRUE DISAGREEMENTS ({len(self.true_disagreements)}):")
            for topic in self.true_disagreements:
                lines.append(str(topic))

        if self.knowledge_gaps:
            lines.append("")
            lines.append(f"  KNOWLEDGE GAPS ({len(self.knowledge_gaps)}):")
            for topic in self.knowledge_gaps:
                lines.append(str(topic))

        lines.append("")
        lines.append(f"  Summary: {len(self.agreements)} agreements, "
                      f"{len(self.corrections)} corrections, "
                      f"{len(self.true_disagreements)} true disagreements, "
                      f"{len(self.knowledge_gaps)} knowledge gaps")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_rich(self):
        """Print discourse results with rich formatting (colors, panels)."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.rule import Rule

        console = Console()

        console.print()
        console.print(Rule(
            f"[bold]MULTI-PERSPECTIVE DISCOURSE: {self.formula_name}[/]",
            style="bright_blue",
        ))

        if self.agreements:
            console.print(f"\n  [bold]AGREEMENTS ({len(self.agreements)}):[/]")
            for topic in self.agreements:
                for claim in topic.claims:
                    is_mechanism = "mechanism-based" in claim.evidence.lower()
                    if is_mechanism:
                        console.print(Panel(
                            f"[bold bright_yellow]MECHANISM-BASED PREDICTION[/]\n\n"
                            f"[white]{claim.detail}[/]\n\n"
                            f"[dim]Subject: {topic.subject}[/]\n"
                            f"[dim]Evidence: {claim.evidence[:100]}[/]\n"
                            f"[dim]Confidence: {claim.confidence}[/]",
                            border_style="bright_yellow",
                            padding=(0, 2),
                        ))
                    elif claim.position == "violation":
                        console.print(
                            f"  [red][X][/] [bold]{topic.subject}[/]\n"
                            f"      [red]{claim.detail[:100]}[/]"
                        )
                    elif claim.position == "concern":
                        console.print(
                            f"  [yellow][=][/] [bold]{topic.subject}[/]\n"
                            f"      [dim]{claim.detail[:100]}[/]"
                        )
                    elif claim.position == "safe":
                        console.print(
                            f"  [green][=][/] [bold]{topic.subject}[/]: safe\n"
                            f"      [dim]{claim.detail[:80]}[/]"
                        )

        if self.corrections:
            console.print(f"\n  [bold]CORRECTIONS ({len(self.corrections)}):[/]")
            for topic in self.corrections:
                w = topic.winning_claim
                if w:
                    console.print(
                        f"  [bright_cyan][>][/] [bold]{topic.subject}[/]\n"
                        f"      Corrected by {w.perspective}: "
                        f"[dim]{w.detail[:80]}[/]"
                    )

        if self.true_disagreements:
            console.print(f"\n  [bold]TRUE DISAGREEMENTS ({len(self.true_disagreements)}):[/]")
            for topic in self.true_disagreements:
                console.print(
                    f"  [bright_blue][?][/] [bold]{topic.subject}[/]: "
                    f"[bright_blue]true disagreement[/]\n"
                    f"      [dim]{topic.summary[:100]}[/]"
                )

        if self.knowledge_gaps:
            console.print(f"\n  [bold]KNOWLEDGE GAPS ({len(self.knowledge_gaps)}):[/]")
            for topic in self.knowledge_gaps:
                console.print(
                    f"  [dim][.][/] [bold]{topic.subject}[/]\n"
                    f"      [dim]{topic.summary[:100]}[/]"
                )

        console.print(
            f"\n  [dim]Summary: {len(self.agreements)} agreements, "
            f"{len(self.corrections)} corrections, "
            f"{len(self.true_disagreements)} true disagreements, "
            f"{len(self.knowledge_gaps)} knowledge gaps[/]"
        )
        console.print(Rule(style="bright_blue", characters="-"))


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_topic(claims: list[Claim]) -> tuple[str, Claim | None]:
    """
    Classify a set of claims about the same subject.

    Returns (classification, winning_claim).
    winning_claim is set only for corrections.
    """
    if not claims:
        return "knowledge_gap", None

    # Single perspective -- no discourse possible
    if len(set(c.perspective for c in claims)) < 2:
        return "agreement", None

    # Split into "safe" and "concern/violation" camps
    safe_claims = [c for c in claims if c.position == "safe"]
    concern_claims = [c for c in claims if c.position in ("concern", "violation")]
    unknown_claims = [c for c in claims if c.position == "unknown"]

    # All agree
    if not concern_claims and not unknown_claims:
        return "agreement", None
    if not safe_claims and not unknown_claims:
        return "agreement", None

    # There's disagreement -- check evidence levels
    if safe_claims and concern_claims:
        max_safe_level = max(c.evidence_level for c in safe_claims)
        max_concern_level = max(c.evidence_level for c in concern_claims)

        gap = abs(max_safe_level - max_concern_level)

        if gap >= CORRECTION_THRESHOLD:
            # Significant evidence asymmetry -- stronger side wins
            if max_concern_level > max_safe_level:
                winner = max(concern_claims, key=lambda c: c.evidence_level)
            else:
                winner = max(safe_claims, key=lambda c: c.evidence_level)
            return "correction", winner

        # Comparable evidence -- true disagreement (mutual subjectivity)
        return "true_disagreement", None

    # Some perspectives have "unknown" -- knowledge gap
    if unknown_claims and len(unknown_claims) == len(claims):
        return "knowledge_gap", None

    # Mix of known and unknown -- report what we know
    return "agreement", None


# ---------------------------------------------------------------------------
# Claim extraction from existing perspectives
# ---------------------------------------------------------------------------

def _extract_physics_claims(obs: FormulationObservation) -> list[Claim]:
    """Extract structured claims from physics observations."""
    claims: list[Claim] = []

    # Molecular observations
    for o in obs.observations:
        if o.category in ("molecular", "phase", "charge"):
            position = {
                "confirmed": "safe",
                "uncertain": "concern",
                "discrepancy": "concern",
            }.get(o.agreement, "unknown")

            claims.append(Claim(
                perspective="physics",
                subject=o.subject,
                position=position,
                detail=o.detail,
                evidence=f"Observed: {o.observed}. Expected: {o.expected}",
                evidence_level=EvidenceLevel.COMPUTATIONAL,
                confidence=o.confidence,
            ))

    # Structural observations (preservative, total %)
    for o in obs.observations:
        if o.category == "structural" and o.agreement == "discrepancy":
            claims.append(Claim(
                perspective="physics",
                subject=o.subject,
                position="concern",
                detail=o.detail,
                evidence=o.observed,
                evidence_level=EvidenceLevel.HEURISTIC,
                confidence=o.confidence,
            ))

    # Mechanism-based interaction predictions (functional group + excipient class)
    for o in obs.observations:
        if o.category == "interaction" and o.agreement == "discrepancy":
            claims.append(Claim(
                perspective="physics",
                subject=o.subject,
                position="concern",
                detail=o.detail,
                evidence=f"Mechanism: {o.source}. {o.expected}",
                evidence_level=EvidenceLevel.COMPUTATIONAL,
                confidence=o.confidence,
            ))

    return claims


def _extract_chemistry_claims(
    obs: FormulationObservation,
    report: ValidationReport,
) -> list[Claim]:
    """Extract structured claims from chemistry validation."""
    claims: list[Claim] = []

    # Violations from observation engine (which checks the knowledge base)
    for v in obs.violations:
        subject = " + ".join(v.ingredients)
        claims.append(Claim(
            perspective="chemistry",
            subject=subject,
            position="violation" if v.severity == "hard" else "concern",
            detail=v.message,
            evidence=v.source,
            evidence_level=EvidenceLevel.RULE_BASED,
            confidence=v.confidence,
        ))

    # Issues from the validation report that aren't already covered by violations
    violation_subjects = {" + ".join(v.ingredients) for v in obs.violations}
    for issue in report.issues:
        subject = issue.ingredient or "formula"
        if issue.ingredient_b:
            subject = f"{issue.ingredient} + {issue.ingredient_b}"
        if subject.upper() in {s.upper() for s in violation_subjects}:
            continue

        position = {"error": "violation", "warning": "concern", "info": "safe"}.get(
            issue.severity, "concern"
        )
        claims.append(Claim(
            perspective="chemistry",
            subject=subject,
            position=position,
            detail=issue.message,
            evidence=issue.mechanism or "knowledge_base",
            evidence_level=EvidenceLevel.RULE_BASED,
            confidence=0.7,  # default for knowledge base rules without explicit confidence
        ))

    return claims


# ---------------------------------------------------------------------------
# Data perspective (experiment memory)
# ---------------------------------------------------------------------------

def _extract_data_claims(
    formula: Formula,
    memory: "ExperimentMemory",
) -> list[Claim]:
    """Extract claims from experiment memory.

    Checks if prior experiments have findings about the ingredients
    in this formula. Creates EMPIRICAL_PROXY claims that can create
    true disagreements with physics (COMPUTATIONAL) claims.
    """
    discoveries = memory.load_discoveries()
    if not discoveries:
        return []

    category = formula.category or "general"
    formula_ings = formula.inci_names_upper

    claims: list[Claim] = []
    seen: set[str] = set()  # avoid duplicate claims per ingredient

    for d in discoveries:
        if d.domain != category:
            continue

        for ing_name in d.ingredients:
            key = f"{ing_name.upper()}:{d.kind}"
            if ing_name.upper() not in formula_ings or key in seen:
                continue
            seen.add(key)

            if d.kind == "preference":
                claims.append(Claim(
                    perspective="data",
                    subject=ing_name,
                    position="safe",
                    detail=(f"Prior experiments: {d.finding} "
                            f"(confirmed in {d.evidence_count} experiments)"),
                    evidence=f"experiment memory, {d.evidence_count} experiments",
                    evidence_level=EvidenceLevel.EMPIRICAL_PROXY,
                    confidence=d.confidence,
                ))
            elif d.kind == "avoidance":
                claims.append(Claim(
                    perspective="data",
                    subject=ing_name,
                    position="concern",
                    detail=f"Prior experiments: {d.finding}",
                    evidence=f"experiment memory, {d.evidence_count} experiments",
                    evidence_level=EvidenceLevel.EMPIRICAL_PROXY,
                    confidence=d.confidence,
                ))

    return claims


# ---------------------------------------------------------------------------
# Process perspective (protocol evaluation)
# ---------------------------------------------------------------------------

# Maximum recommended processing temperatures (degrees C).
# Only includes ingredients where thermal degradation is well-established
# in the formulation chemistry literature.
#
# Sources:
#   Retinol/retinyl palmitate: Maia Campos et al., J Cosmet Dermatol 2019;
#     Boisnic et al., Int J Cosmet Sci 2005. Retinoids isomerize and oxidize
#     above 40C; retinyl palmitate is slightly more thermostable.
#   Ascorbic acid: Gallarate et al., Int J Pharm 1999; Telang, Indian
#     Dermatol Online J 2013. L-AA oxidizes rapidly above 40C in aqueous solution.
#   Sodium ascorbyl phosphate: More thermostable ester derivative of L-AA;
#     manufacturer guidance (BASF, DSM) recommends addition below 50C.
#   Tocopherol: Rietjens et al., Food Chem Toxicol 2002. Alpha-tocopherol
#     is relatively heat-stable but degrades above 60C in the presence of
#     metal ions and oxygen.
#   Ubiquinone (CoQ10): Beg et al., J Pharm Bioallied Sci 2011.
#     Degrades above 50C, especially in light-exposed formulations.
#   EGCG: Zeng et al., Food Chem 2019. Epimerizes and oxidizes above 50C
#     in aqueous systems; pH-dependent.
THERMAL_LIMITS: dict[str, float] = {
    "RETINOL": 40,
    "RETINYL PALMITATE": 45,
    "ASCORBIC ACID": 40,
    "L-ASCORBIC ACID": 40,
    "SODIUM ASCORBYL PHOSPHATE": 50,
    "TOCOPHEROL": 60,
    "UBIQUINONE": 50,
    "EPIGALLOCATECHIN GALLATE": 50,
}

# Ingredients that should be added in cool-down phase (below ~55C).
# Preservatives: volatile components flash off at high temperatures;
#   antimicrobial efficacy is reduced. Standard manufacturing practice
#   per PCPC (Personal Care Products Council) guidelines and CTFA
#   (Cosmetic, Toiletry and Fragrance Association) technical guidance.
# Actives: see THERMAL_LIMITS above for specific degradation temperatures.
COOLDOWN_REQUIRED: set[str] = {
    "RETINOL", "RETINYL PALMITATE",
    "ASCORBIC ACID", "L-ASCORBIC ACID",
    "PHENOXYETHANOL", "ETHYLHEXYLGLYCERIN", "CAPRYLYL GLYCOL",
    "BENZYL ALCOHOL", "POTASSIUM SORBATE", "SODIUM BENZOATE",
}


def _extract_process_claims(
    formula: Formula,
    protocol: Protocol,
) -> list[Claim]:
    """Evaluate a manufacturing protocol for feasibility.

    Checks:
    1. Heat-sensitive ingredients assigned to high-temperature phases
    2. Preservatives not in cool-down phase
    3. Oil phase present but no homogenization step
    4. Target pH specified but no pH adjustment step
    """
    claims: list[Claim] = []

    # 1. Thermal sensitivity checks
    for ing in formula.ingredients:
        name_upper = ing.inci_name.upper().strip()
        max_temp = THERMAL_LIMITS.get(name_upper)
        if max_temp is None:
            continue

        phase = protocol.phase_for_ingredient(ing.inci_name)
        if phase is None:
            continue

        if phase.target_temp_c is not None and phase.target_temp_c > max_temp:
            claims.append(Claim(
                perspective="process",
                subject=ing.inci_name,
                position="violation",
                detail=(f"{ing.inci_name} degrades above {max_temp}C "
                        f"but assigned to {phase.label} at {phase.target_temp_c}C"),
                evidence=f"thermal degradation limit: {max_temp}C",
                evidence_level=EvidenceLevel.RULE_BASED,
                confidence=0.9,  # literature-sourced thermal limits
            ))
        elif phase.target_temp_c is not None and phase.target_temp_c <= max_temp:
            claims.append(Claim(
                perspective="process",
                subject=ing.inci_name,
                position="safe",
                detail=(f"{ing.inci_name} assigned to {phase.label} "
                        f"at {phase.target_temp_c}C (limit: {max_temp}C)"),
                evidence=f"thermal degradation limit: {max_temp}C",
                evidence_level=EvidenceLevel.RULE_BASED,
                confidence=0.9,  # literature-sourced thermal limits
            ))

    # 2. Preservative phase check
    for ing in formula.ingredients:
        name_upper = ing.inci_name.upper().strip()
        if name_upper not in COOLDOWN_REQUIRED:
            continue

        phase = protocol.phase_for_ingredient(ing.inci_name)
        if phase is None:
            continue

        if phase.target_temp_c is not None and phase.target_temp_c > 55:
            claims.append(Claim(
                perspective="process",
                subject=ing.inci_name,
                position="concern",
                detail=(f"{ing.inci_name} should be added in cool-down phase "
                        f"(< 55C) but assigned to {phase.label} at {phase.target_temp_c}C"),
                evidence="standard manufacturing practice (PCPC/CTFA guidelines)",
                evidence_level=EvidenceLevel.HEURISTIC,
                confidence=0.8,  # well-established practice but not ingredient-specific data
            ))

    # 3. Oil phase without homogenization
    oil_pct = sum(
        ing.percentage for ing in formula.ingredients
        if ing.phase and ing.phase.upper() in ("B", "OIL", "OIL PHASE")
    )
    # Also check protocol phase assignments
    if not oil_pct:
        oil_phase = protocol.get_phase("B")
        if oil_phase:
            oil_pct = sum(
                ing.percentage for ing in formula.ingredients
                if ing.inci_name in oil_phase.ingredients
            )

    if oil_pct > 5:
        has_homogenize = any(s.action == "homogenize" for s in protocol.steps)
        if not has_homogenize:
            claims.append(Claim(
                perspective="process",
                subject="formula",
                position="concern",
                detail=(f"Oil phase is {oil_pct:.1f}% but no homogenization step. "
                        "Emulsion may be unstable without adequate particle size reduction."),
                evidence="emulsion manufacturing practice",
                evidence_level=EvidenceLevel.HEURISTIC,
                confidence=0.7,  # depends on oil phase composition; some systems self-emulsify
            ))

    # 4. Missing pH adjustment
    if formula.target_ph is not None:
        has_ph_step = any(s.action == "adjust_ph" for s in protocol.steps)
        if not has_ph_step:
            claims.append(Claim(
                perspective="process",
                subject="formula",
                position="concern",
                detail=(f"Target pH {formula.target_ph} specified but no "
                        "pH adjustment step in protocol"),
                evidence="manufacturing completeness",
                evidence_level=EvidenceLevel.HEURISTIC,
                confidence=0.6,  # pH may self-adjust from ingredient buffering
            ))

    return claims


def _subject_key(subject: str) -> str:
    """Normalize a subject for matching across perspectives."""
    parts = sorted(s.strip().upper() for s in subject.split("+"))
    return " + ".join(parts)


def _build_topics(all_claims: list[Claim]) -> list[DiscourseTopic]:
    """Group claims by subject and classify each topic."""
    by_subject: dict[str, list[Claim]] = {}
    for claim in all_claims:
        key = _subject_key(claim.subject)
        by_subject.setdefault(key, []).append(claim)

    topics: list[DiscourseTopic] = []
    for subject_key, claims in by_subject.items():
        classification, winner = classify_topic(claims)

        # Build summary with correct perspective-to-position mapping
        perspectives = sorted(set(c.perspective for c in claims))
        perspective_positions = {
            c.perspective: c.position for c in claims
        }

        if classification == "agreement":
            positions = sorted(set(c.position for c in claims))
            summary = f"{', '.join(perspectives)} agree: {positions[0]}"
        elif classification == "correction":
            corrector = winner.perspective if winner else "unknown"
            summary = f"Corrected by {corrector}: {winner.detail[:80]}" if winner else "Corrected"
        elif classification == "true_disagreement":
            parts = [f"{p}={perspective_positions.get(p, '?')}" for p in perspectives]
            summary = f"{', '.join(perspectives)} disagree ({', '.join(parts)})"
        else:
            summary = f"Insufficient data from {', '.join(perspectives)}"

        topics.append(DiscourseTopic(
            subject=claims[0].subject,  # use original casing
            claims=claims,
            classification=classification,
            summary=summary,
            winning_claim=winner,
        ))

    return topics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_discourse(
    formula: Formula,
    protocol: Protocol | None = None,
    memory: "ExperimentMemory | None" = None,
    observe_mode: str = "engineering",
    validate_mode: str = "formulation",
) -> Discourse:
    """
    Run multi-perspective evaluation and classify disagreements.

    Up to four perspectives, depending on what's provided:
      - Physics (always): molecular properties, phase behavior, solubility
      - Chemistry (always): interaction rules, pH, preservatives
      - Data (if memory provided): findings from prior experiments
      - Process (if protocol provided): manufacturing feasibility

    Returns a Discourse with classified topics: agreements, corrections,
    true disagreements, and knowledge gaps.

        from openmix import Formula
        from openmix.discourse import evaluate_discourse

        formula = Formula(
            ingredients=[("Retinol", 1.0), ("Ascorbic Acid", 10.0), ("Water", 89.0)],
            target_ph=3.5,
            category="skincare",
        )
        disc = evaluate_discourse(formula)
        print(disc)
        print(f"True disagreements: {len(disc.true_disagreements)}")
    """
    # Perspective 1: Physics (always)
    obs = observe(formula, mode=observe_mode)
    physics_claims = _extract_physics_claims(obs)

    # Perspective 2: Chemistry (always)
    report = validate(formula, mode=validate_mode)
    chemistry_claims = _extract_chemistry_claims(obs, report)

    all_claims = physics_claims + chemistry_claims

    # Perspective 3: Data (if memory available)
    if memory is not None:
        data_claims = _extract_data_claims(formula, memory)
        all_claims.extend(data_claims)

    # Perspective 4: Process (if protocol available)
    if protocol is not None:
        process_claims = _extract_process_claims(formula, protocol)
        all_claims.extend(process_claims)

    topics = _build_topics(all_claims)

    return Discourse(
        formula_name=formula.name or "unnamed",
        topics=topics,
    )
