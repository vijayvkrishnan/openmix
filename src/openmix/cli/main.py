"""
OpenMix CLI — validate and score formulations from the command line.

Usage:
    openmix validate formula.yaml
    openmix score formula.yaml
    openmix info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from openmix import __version__
from openmix.schema import Formula
from openmix.validate import validate
from openmix.score import score
from openmix.observe import observe
from openmix.knowledge.loader import load_knowledge


def _load_formula(filepath: Path) -> Formula:
    """Load a formula from YAML or JSON."""
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif filepath.suffix == ".json":
            data = json.load(f)
        else:
            print(f"Error: Unsupported format: {filepath.suffix}", file=sys.stderr)
            sys.exit(1)

    try:
        return Formula(**data)
    except Exception as e:
        print(f"Error parsing formula: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args):
    """Validate a formulation."""
    formula = _load_formula(Path(args.file))
    mode = args.mode or "safety"
    report = validate(formula, mode=mode)
    print(report)

    if args.json:
        print("\n--- JSON ---")
        print(report.model_dump_json(indent=2))

    sys.exit(0 if report.passed else 1)


def cmd_score(args):
    """Score a formulation."""
    formula = _load_formula(Path(args.file))
    result = score(formula)
    print(result)
    sys.exit(0)


def cmd_observe(args):
    """Observe a formulation through physics."""
    formula = _load_formula(Path(args.file))
    mode = args.mode or "engineering"
    result = observe(formula, mode=mode)
    print(result)

    if args.json:
        import json as _json
        data = {
            "mode": result.mode,
            "resolution_rate": result.resolution_rate,
            "hard_violations": result.hard_violations,
            "soft_violations": result.soft_violations,
            "concern_count": result.concern_count,
            "concerns": len(result.concerns),
            "signals": len(result.signals),
            "discoveries": len(result.discoveries),
            "observations": [
                {"category": o.category, "subject": o.subject,
                 "observed": o.observed, "agreement": o.agreement,
                 "confidence": o.confidence}
                for o in result.observations
            ],
        }
        print("\n--- JSON ---")
        print(_json.dumps(data, indent=2))

    sys.exit(0)


def cmd_discourse(args):
    """Run multi-perspective discourse evaluation."""
    from openmix.discourse import evaluate_discourse
    from openmix.memory import ExperimentMemory
    from openmix.protocol import Protocol, Phase, ProcessStep

    formula = _load_formula(Path(args.file))

    # Load protocol if provided
    protocol = None
    if args.protocol:
        protocol_path = Path(args.protocol)
        if not protocol_path.exists():
            print(f"Error: Protocol file not found: {protocol_path}", file=sys.stderr)
            sys.exit(1)
        with open(protocol_path, "r", encoding="utf-8") as f:
            pdata = yaml.safe_load(f)
        phases = [Phase(**p) for p in pdata.get("phases", [])]
        steps = [ProcessStep(**s) for s in pdata.get("steps", [])]
        protocol = Protocol(
            phases=phases,
            steps=steps,
            equipment=pdata.get("equipment", []),
            batch_size_g=pdata.get("batch_size_g", 100.0),
        )

    # Load experiment memory
    memory = ExperimentMemory() if not args.no_memory else None

    disc = evaluate_discourse(
        formula,
        protocol=protocol,
        memory=memory,
    )
    disc.print_rich()

    sys.exit(0 if not disc.true_disagreements else 1)


def cmd_experiment(args):
    """Run an autonomous formulation experiment."""
    from openmix.experiment import Experiment

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    exp = Experiment.from_file(filepath, verbose=True,
                               use_memory=not args.no_memory)
    log = exp.run()

    if args.save:
        log.save(args.save)
        print(f"\nExperiment log saved to {args.save}")

    sys.exit(0 if log.converged else 1)


def cmd_run(args):
    """Run an experiment from a natural language brief."""
    from openmix.experiment import Experiment

    brief = " ".join(args.brief)
    save_plan = args.save_plan

    exp = Experiment.from_brief(brief, verbose=True, save_plan=save_plan,
                                use_memory=not args.no_memory)
    log = exp.run()

    if args.save:
        log.save(args.save)
        print(f"\nExperiment log saved to {args.save}")

    sys.exit(0 if log.converged else 1)


def cmd_demo(args):
    """Run a built-in demo -- no files or API keys needed."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from openmix import Formula, validate, observe as observe_fn
    from openmix.discourse import evaluate_discourse
    from openmix.protocol import Protocol, Phase, ProcessStep

    console = Console()
    console.print()
    console.print(Panel(
        "[bold]RDKit is for molecules. OpenMix is for mixtures.[/]\n\n"
        "[dim]Multi-perspective formulation evaluation with\n"
        "evidence-based disagreement classification.[/]",
        title=f"[bold]OpenMix[/]  [dim]v{__version__}[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))
    console.print()

    # Demo 1: Multi-perspective discourse (the headline feature)
    console.print(Rule("[bold cyan]Discourse: Multi-perspective evaluation[/]"))
    console.print("  [dim]Four perspectives evaluate the same formula + protocol.[/]")
    console.print("  [dim]Disagreements are classified, not hidden.[/]")
    console.print()

    serum = Formula(
        name="Retinol + Vitamin C Serum",
        ingredients=[
            ("Water", 58.0),
            ("Ascorbic Acid", 15.0),
            ("Squalane", 10.0),
            ("Glycerin", 8.0),
            ("Niacinamide", 4.0),
            ("Retinol", 1.0),
            ("Cetyl Alcohol", 2.0),
            ("Phenoxyethanol", 1.0),
            ("Tocopherol", 0.5),
            ("Xanthan Gum", 0.3),
            ("Disodium EDTA", 0.1),
            ("Citric Acid", 0.1),
        ],
        target_ph=3.5,
        category="skincare",
    )

    # Deliberately flawed protocol: actives in heat phase
    protocol = Protocol(
        phases=[
            Phase("A", "Water Phase", 75.0,
                  ["Water", "Glycerin", "Ascorbic Acid", "Niacinamide",
                   "Xanthan Gum", "Citric Acid", "Disodium EDTA"]),
            Phase("B", "Oil Phase", 75.0,
                  ["Squalane", "Retinol", "Cetyl Alcohol", "Tocopherol"]),
            Phase("C", "Cool-Down", 40.0, ["Phenoxyethanol"]),
        ],
        steps=[
            ProcessStep("heat", "A", {"temp_c": 75, "duration_min": 10}),
            ProcessStep("heat", "B", {"temp_c": 75, "duration_min": 10}),
            ProcessStep("combine", "B", {"into": "A", "mixing_rpm": 500}),
            ProcessStep("cool", "all", {"target_c": 40}),
            ProcessStep("add", "C", {}),
        ],
        equipment=["overhead stirrer"],
        batch_size_g=100.0,
    )

    disc = evaluate_discourse(serum, protocol=protocol)
    disc.print_rich()

    # Demo 2: Validate a dangerous formula
    console.print(Rule("[bold yellow]Validate: Catches dangerous interactions[/]"))
    console.print()

    dangerous = Formula(
        name="Household Cleaner",
        ingredients=[
            ("Sodium Hypochlorite", 5.0),
            ("Ammonia", 3.0),
            ("Water", 92.0),
        ],
        category="home_care",
    )
    report = validate(dangerous, mode="safety")
    print(report)

    # Demo 3: Physics observation engine
    console.print(Rule("[bold green]Observe: Physics observation engine[/]"))
    console.print()

    clean = Formula(
        name="Simple Moisturizer",
        ingredients=[
            ("Water", 72.0),
            ("Glycerin", 8.0),
            ("Caprylic/Capric Triglyceride", 10.0),
            ("Cetearyl Alcohol", 4.0),
            ("Polysorbate 60", 3.0),
            ("Phenoxyethanol", 1.0),
            ("Tocopherol", 0.5),
            ("Xanthan Gum", 0.5),
            ("Citric Acid", 0.5),
            ("Disodium EDTA", 0.1),
        ],
        target_ph=5.5,
        category="skincare",
    )
    obs = observe_fn(clean)
    print(obs)

    console.print()
    console.print(Panel(
        "[bold]openmix discourse[/] formula.yaml        Multi-perspective evaluation\n"
        "[bold]openmix observe[/] formula.yaml           Physics observations\n"
        "[bold]openmix validate[/] formula.yaml          Rule-based validation\n"
        '[bold]openmix run[/] "Design a stable serum"    Autonomous experiment (needs API key)\n'
        "[bold]openmix memory[/]                         Inspect experiment memory",
        title="[bold]Next steps[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))


def cmd_memory(args):
    """Inspect experiment memory."""
    from openmix.memory import ExperimentMemory

    memory = ExperimentMemory()

    if args.clear:
        import shutil
        if memory.base_dir.exists():
            shutil.rmtree(memory.base_dir)
            print("  Experiment memory cleared.")
        else:
            print("  No experiment memory found.")
        return

    if args.discoveries:
        discoveries = memory.load_discoveries()
        if not discoveries:
            print("  No discoveries yet. Run some experiments first.")
            return
        for d in sorted(discoveries, key=lambda x: -x.confidence):
            print(f"  [{d.confidence:.2f}] [{d.kind}] {d.finding}")
            print(f"         domain: {d.domain}, "
                  f"evidence: {d.evidence_count} experiments")
        return

    print(memory.summary())


def cmd_info(args):
    """Show OpenMix info and knowledge stats."""
    kb = load_knowledge()
    hard = len(kb.hard_rules)
    soft = len(kb.soft_rules)

    print(f"OpenMix v{__version__}")
    print(f"  Interaction rules: {len(kb.interaction_rules)} ({hard} hard + {soft} soft)")
    print(f"  Oil HLB entries:   {len(kb.oil_hlb)}")
    print(f"  Alias groups:      {len(kb.aliases)}")
    print()

    categories = {}
    for rule in kb.interaction_rules:
        categories[rule.category] = categories.get(rule.category, 0) + 1
    print("  Rules by category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    print()
    mechanisms = {}
    for rule in kb.interaction_rules:
        mechanisms[rule.mechanism] = mechanisms.get(rule.mechanism, 0) + 1
    print("  Rules by mechanism:")
    for mech, count in sorted(mechanisms.items(), key=lambda x: -x[1]):
        print(f"    {mech}: {count}")


def _welcome():
    """Welcome screen — first thing a new user sees."""
    kb = load_knowledge()
    hard = len(kb.hard_rules)
    soft = len(kb.soft_rules)
    total = len(kb.interaction_rules)
    domains = len(set(r.category for r in kb.interaction_rules))

    print(f"""
  OpenMix v{__version__}
  Autonomous Formulation Science

  The lab does the testing. OpenMix does the noticing.
  {total} interaction rules ({hard} hard + {soft} soft) | {domains} domains

  Get started:

    openmix demo
      Try it now. Physics observations, validation, two modes.
      No API key needed.

    openmix observe formula.yaml
      Physics observation engine. Reports what it sees, what it expected,
      and where they disagree. Two modes: engineering / discovery.

    openmix run "Design a stable vitamin C serum under $30/kg"
      Run an autonomous formulation experiment from natural language.
      Requires ANTHROPIC_API_KEY or OPENAI_API_KEY.

    openmix validate formula.yaml
      Rule-based validation (3 modes: safety / formulation / discovery).

    openmix info
      Show knowledge base statistics.

  Python:

    from openmix import Formula, observe, validate, Experiment
    obs = observe(formula, mode="discovery")
    result = Experiment.from_brief("your research question").run()

  Docs: https://github.com/vijayvkrishnan/openmix
""")


def main():
    parser = argparse.ArgumentParser(
        prog="openmix",
        description="OpenMix -- computational formulation science",
    )
    parser.add_argument("--version", action="version", version=f"openmix {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # validate
    vp = subparsers.add_parser("validate", help="Validate a formulation")
    vp.add_argument("file", help="Path to formula YAML or JSON")
    vp.add_argument("--mode", choices=["safety", "formulation", "discovery"],
                    help="Validation mode (default: safety)")
    vp.add_argument("--json", action="store_true", help="Also output JSON")
    vp.set_defaults(func=cmd_validate)

    # score
    sp = subparsers.add_parser("score", help="Score a formulation")
    sp.add_argument("file", help="Path to formula YAML or JSON")
    sp.set_defaults(func=cmd_score)

    # observe
    op = subparsers.add_parser("observe", help="Observe a formulation through physics")
    op.add_argument("file", help="Path to formula YAML or JSON")
    op.add_argument("--mode", choices=["engineering", "discovery"],
                    default="engineering",
                    help="Observation mode (default: engineering)")
    op.add_argument("--json", action="store_true", help="Also output JSON")
    op.set_defaults(func=cmd_observe)

    # discourse
    dcp = subparsers.add_parser("discourse",
                                help="Multi-perspective evaluation with disagreement classification")
    dcp.add_argument("file", help="Path to formula YAML or JSON")
    dcp.add_argument("--protocol", help="Path to protocol YAML")
    dcp.add_argument("--no-memory", action="store_true",
                     help="Disable experiment memory")
    dcp.set_defaults(func=cmd_discourse)

    # run (natural language)
    rp = subparsers.add_parser("run", help="Run an experiment from natural language")
    rp.add_argument("brief", nargs="+", help="Research brief in natural language")
    rp.add_argument("--save", help="Save experiment log to file")
    rp.add_argument("--save-plan", help="Save generated experiment plan as YAML")
    rp.add_argument("--no-memory", action="store_true",
                    help="Disable experiment memory (no persistence or prior knowledge)")
    rp.set_defaults(func=cmd_run)

    # experiment (from YAML)
    ep = subparsers.add_parser("experiment", help="Run from an experiment YAML file")
    ep.add_argument("file", help="Path to experiment YAML")
    ep.add_argument("--save", help="Save experiment log to file")
    ep.add_argument("--no-memory", action="store_true",
                    help="Disable experiment memory (no persistence or prior knowledge)")
    ep.set_defaults(func=cmd_experiment)

    # demo
    dp = subparsers.add_parser("demo", help="Run a built-in demo (no API key needed)")
    dp.set_defaults(func=cmd_demo)

    # memory
    mp = subparsers.add_parser("memory", help="Inspect experiment memory")
    mp.add_argument("--discoveries", action="store_true",
                    help="Show all discoveries")
    mp.add_argument("--clear", action="store_true",
                    help="Clear all experiment memory")
    mp.set_defaults(func=cmd_memory)

    # info
    ip = subparsers.add_parser("info", help="Show knowledge base stats")
    ip.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        _welcome()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
