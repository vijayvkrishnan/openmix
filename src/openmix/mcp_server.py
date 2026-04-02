"""
OpenMix MCP Server — formulation intelligence for AI agents.

Exposes OpenMix's core capabilities as MCP tools that any compatible
AI agent (Claude Code, Cursor, etc.) can call during conversation.

Security:
    - All tools are read-only. No file writes, no database modifications.
    - The ingredient resolver may make HTTP requests to PubChem
      (pubchem.ncbi.nlm.nih.gov) for ingredients not in the local seed
      cache (2,400+ ingredients). If you are working with proprietary
      or confidential ingredient names, disable PubChem lookups or
      pre-populate the seed cache.
    - The server runs over stdio (local only). It does not open
      network ports or accept remote connections.
    - No authentication is required. Access is controlled by who
      can start the process.

Usage:
    # Start the server
    python -m openmix.mcp_server

    # Configure in Claude Code settings (~/.claude.json):
    # "mcpServers": {
    #   "openmix": {
    #     "command": "python",
    #     "args": ["-m", "openmix.mcp_server"]
    #   }
    # }
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "openmix",
    instructions="Computational formulation science: observe mixtures through "
                 "molecular physics, validate ingredient interactions, resolve "
                 "molecular identity, evaluate formulations from multiple perspectives "
                 "(physics, chemistry, data, process) with disagreement classification. "
                 "2,400+ ingredients, 258 interaction rules, "
                 "6 domains (skincare, pharma, supplements, food, beverages, home care).",
)


@mcp.tool()
def observe_formulation(
    ingredients: list[dict],
    target_ph: float | None = None,
    category: str | None = None,
    mode: str = "engineering",
    name: str | None = None,
) -> str:
    """Observe a formulation through molecular physics.

    Reports what it sees, what it expected, and where they disagree.
    Checks: molecular properties (LogP, MW), charge compatibility,
    phase behavior, preservative systems, pH-ionization (Henderson-Hasselbalch),
    and 258 ingredient interaction rules across 6 domains.

    Two modes:
      engineering: discrepancies are problems to fix (default)
      discovery: discrepancies are signals to investigate

    Args:
        ingredients: List of {"inci_name": str, "percentage": float} dicts.
            Example: [{"inci_name": "Water", "percentage": 70.0},
                      {"inci_name": "Ascorbic Acid", "percentage": 15.0}]
        target_ph: Target pH of the formulation (enables pH-ionization observations).
        category: Product category (skincare, pharma, supplement, food, beverage, home_care).
        mode: "engineering" (minimize concerns) or "discovery" (investigate surprises).
        name: Optional formula name.
    """
    from openmix import Formula, observe

    formula = Formula(
        name=name,
        ingredients=ingredients,
        target_ph=target_ph,
        category=category,
    )
    obs = observe(formula, mode=mode)
    return str(obs)


@mcp.tool()
def validate_formulation(
    ingredients: list[dict],
    target_ph: float | None = None,
    category: str | None = None,
    mode: str = "safety",
    name: str | None = None,
) -> str:
    """Validate a formulation against the interaction knowledge base.

    Checks 258 rules (85 hard + 173 soft) across 6 domains. Each rule
    has a confidence score, literature source, and optional conditions.

    Three modes:
      safety: flag everything (default)
      formulation: real issues only, with mitigations
      discovery: only block genuinely dangerous reactions

    Args:
        ingredients: List of {"inci_name": str, "percentage": float} dicts.
        target_ph: Target pH.
        category: Product category.
        mode: "safety", "formulation", or "discovery".
        name: Optional formula name.
    """
    from openmix import Formula, validate

    formula = Formula(
        name=name,
        ingredients=ingredients,
        target_ph=target_ph,
        category=category,
    )
    report = validate(formula, mode=mode)
    return str(report)


@mcp.tool()
def resolve_ingredient(inci_name: str) -> str:
    """Resolve an ingredient name to its molecular identity and properties.

    Returns SMILES, molecular weight, LogP, TPSA, hydrogen bond donors/acceptors,
    and charge type. Uses a three-tier lookup: seed cache (2,400+ ingredients),
    user cache, and PubChem API fallback.

    Args:
        inci_name: INCI (International Nomenclature of Cosmetic Ingredients) name.
            Examples: "Niacinamide", "Ascorbic Acid", "Sodium Lauryl Sulfate"
    """
    from openmix.resolver import resolve

    r = resolve(inci_name)
    if not r.resolved:
        return f"{inci_name}: not resolved. Try a different name or check spelling."

    lines = [f"{inci_name}:"]
    if r.smiles:
        lines.append(f"  SMILES: {r.smiles}")
    if r.molecular_weight:
        lines.append(f"  Molecular weight: {r.molecular_weight}")
    if r.log_p is not None:
        lines.append(f"  LogP: {r.log_p}")
    if r.tpsa is not None:
        lines.append(f"  TPSA: {r.tpsa}")
    if r.hbd is not None:
        lines.append(f"  H-bond donors: {r.hbd}")
    if r.hba is not None:
        lines.append(f"  H-bond acceptors: {r.hba}")
    if r.charge_type:
        lines.append(f"  Charge type: {r.charge_type}")
    lines.append(f"  Source: {r.source}")
    return "\n".join(lines)


@mcp.tool()
def check_ingredient_compatibility(ingredient_a: str, ingredient_b: str) -> str:
    """Check whether two ingredients are compatible.

    Searches the knowledge base (258 rules, 6 domains) for known
    interactions between the two ingredients. Reports hard violations
    (dangerous), soft violations (conditional), or no known issues.

    Args:
        ingredient_a: First ingredient INCI name.
        ingredient_b: Second ingredient INCI name.
    """
    from openmix import Formula, validate

    formula = Formula(
        ingredients=[
            {"inci_name": ingredient_a, "percentage": 5.0},
            {"inci_name": ingredient_b, "percentage": 5.0},
            {"inci_name": "Water", "percentage": 90.0},
        ],
    )
    report = validate(formula, mode="safety")

    compat_issues = [i for i in report.issues if i.check == "compatibility"]
    if not compat_issues:
        return f"{ingredient_a} + {ingredient_b}: No known interaction in the knowledge base."

    lines = [f"{ingredient_a} + {ingredient_b}:"]
    for issue in compat_issues:
        lines.append(f"  [{issue.severity.upper()}] {issue.message}")
        if issue.mechanism:
            lines.append(f"    Mechanism: {issue.mechanism}")
    return "\n".join(lines)


@mcp.tool()
def ph_check(ingredient_name: str, target_ph: float) -> str:
    """Check whether a target pH is suitable for a specific ingredient.

    Uses Henderson-Hasselbalch equation with literature pKa values
    to compute ionization state and assess pH suitability.

    Args:
        ingredient_name: INCI name of the ingredient.
        target_ph: The pH to evaluate.
    """
    from openmix.knowledge.pka import load_pka_data, assess_ph_suitability

    pka_db = load_pka_data()
    key = ingredient_name.upper().strip()

    if key not in pka_db:
        return (f"{ingredient_name}: No pKa data available. "
                f"pH assessment requires literature pKa values.")

    entry = pka_db[key]
    result = assess_ph_suitability(entry, target_ph)

    if result["ionized_fraction"] is None:
        return f"{ingredient_name}: No ionization data."

    return result["detail"]


@mcp.tool()
def discourse_evaluation(
    ingredients: list[dict],
    target_ph: float | None = None,
    category: str | None = None,
    name: str | None = None,
) -> str:
    """Multi-perspective formulation evaluation with disagreement classification.

    Evaluates a formulation from multiple perspectives (physics, chemistry,
    data from prior experiments) and classifies where they agree, where one
    corrects another, and where they genuinely disagree.

    Disagreement types:
      agreement: all perspectives concur
      correction: one perspective has stronger evidence and overrides another
      true_disagreement: perspectives have comparable evidence but disagree
          (these are worth investigating -- potential discoveries)
      knowledge_gap: no perspective has enough information

    Args:
        ingredients: List of {"inci_name": str, "percentage": float} dicts.
        target_ph: Target pH of the formulation.
        category: Product category.
        name: Optional formula name.
    """
    from openmix import Formula
    from openmix.discourse import evaluate_discourse
    from openmix.memory import ExperimentMemory

    formula = Formula(
        name=name,
        ingredients=ingredients,
        target_ph=target_ph,
        category=category,
    )
    memory = ExperimentMemory()
    disc = evaluate_discourse(formula, memory=memory)
    return str(disc)


@mcp.tool()
def experiment_memory() -> str:
    """Inspect the experiment memory -- prior experiments and accumulated discoveries.

    Shows the experiment index (all past runs), discovery counts by type,
    and high-confidence findings. The experiment memory persists across
    runs and is used by the discourse engine's data perspective.
    """
    from openmix.memory import ExperimentMemory

    memory = ExperimentMemory()
    return memory.summary()


@mcp.tool()
def list_discoveries(domain: str | None = None) -> str:
    """List all accumulated discoveries from prior experiments.

    Discoveries are cross-run findings with confidence scores. They
    include ingredient preferences, avoidances, patterns, and concerns
    extracted from completed experiments.

    Args:
        domain: Filter by domain (skincare, supplement, beverage, etc.).
            If not provided, shows all domains.
    """
    from openmix.memory import ExperimentMemory

    memory = ExperimentMemory()
    discoveries = memory.load_discoveries()

    if not discoveries:
        return "No discoveries yet. Run experiments with `openmix run` to accumulate findings."

    if domain:
        discoveries = [d for d in discoveries if d.domain == domain]
        if not discoveries:
            return f"No discoveries for domain '{domain}'."

    lines = [f"Discoveries ({len(discoveries)} total):"]
    for d in sorted(discoveries, key=lambda x: -x.confidence):
        lines.append(
            f"  [{d.confidence:.2f}] [{d.kind}] {d.finding} "
            f"(domain: {d.domain}, {d.evidence_count} experiments)"
        )
    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
