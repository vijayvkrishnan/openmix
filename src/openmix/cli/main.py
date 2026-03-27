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


def cmd_experiment(args):
    """Run an autonomous formulation experiment."""
    from openmix.experiment import Experiment

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    exp = Experiment.from_file(filepath, verbose=True)
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

    exp = Experiment.from_brief(brief, verbose=True, save_plan=save_plan)
    log = exp.run()

    if args.save:
        log.save(args.save)
        print(f"\nExperiment log saved to {args.save}")

    sys.exit(0 if log.converged else 1)


def cmd_demo(args):
    """Run a built-in demo — no files or API keys needed."""
    from openmix import Formula, validate, score as compute_score

    print(f"OpenMix v{__version__} -- Demo\n")

    # Demo 1: Validate a problematic formula
    print("=" * 60)
    print("  1. Validate: Catches ingredient interactions")
    print("=" * 60)

    formula = Formula(
        name="Anti-Aging Serum",
        ingredients=[
            ("Retinol", 1.0),
            ("Glycolic Acid", 8.0),
            ("Benzoyl Peroxide", 2.5),
            ("Copper Tripeptide-1", 1.0),
            ("Ascorbic Acid", 15.0),
            ("Niacinamide", 5.0),
            ("Water", 60.5),
            ("Phenoxyethanol", 1.0),
            ("Glycerin", 6.0),
        ],
        target_ph=3.5,
        category="skincare",
    )

    report = validate(formula, mode="safety")
    print(report)

    # Demo 2: Score a clean formula
    print("=" * 60)
    print("  2. Score: Quantitative stability prediction")
    print("=" * 60)

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
    print(f"\n  {clean.name}")
    result = compute_score(clean)
    print(result)

    print("\n" + "=" * 60)
    print("  Next steps:")
    print("  - openmix validate your_formula.yaml")
    print("  - openmix score your_formula.yaml")
    print("  - openmix run \"Design a stable vitamin C serum\"  (needs API key)")
    print("=" * 60)


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

  The autoresearch framework for chemistry.
  {total} interaction rules ({hard} hard + {soft} soft) | {domains} domains

  Get started:

    openmix demo
      Try it now. Validates a formula, shows stability scoring.
      No API key needed.

    openmix run "Design a stable vitamin C serum under $30/kg"
      Run an autonomous formulation experiment from natural language.
      Requires ANTHROPIC_API_KEY or OPENAI_API_KEY.

    openmix validate formula.yaml
      Validate a formulation file against the knowledge base.

    openmix score formula.yaml
      Score a formulation (0-100 with decomposed sub-scores).

    openmix info
      Show knowledge base statistics.

  Python:

    from openmix import Formula, validate, score, Experiment
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

    # run (natural language)
    rp = subparsers.add_parser("run", help="Run an experiment from natural language")
    rp.add_argument("brief", nargs="+", help="Research brief in natural language")
    rp.add_argument("--save", help="Save experiment log to file")
    rp.add_argument("--save-plan", help="Save generated experiment plan as YAML")
    rp.set_defaults(func=cmd_run)

    # experiment (from YAML)
    ep = subparsers.add_parser("experiment", help="Run from an experiment YAML file")
    ep.add_argument("file", help="Path to experiment YAML")
    ep.add_argument("--save", help="Save experiment log to file")
    ep.set_defaults(func=cmd_experiment)

    # demo
    dp = subparsers.add_parser("demo", help="Run a built-in demo (no API key needed)")
    dp.set_defaults(func=cmd_demo)

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
