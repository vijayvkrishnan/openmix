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

    # info
    ip = subparsers.add_parser("info", help="Show knowledge base stats")
    ip.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
