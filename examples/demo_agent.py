#!/usr/bin/env python3
"""
OpenMix Agent Demo — iterative formulation optimization.

The agent doesn't just "check and fix" — it optimizes. Each iteration
proposes a formula, scores it quantitatively, analyzes what worked,
and improves. Like gradient descent, but for chemistry.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/demo_agent.py
    python examples/demo_agent.py skincare
    python examples/demo_agent.py supplement
    python examples/demo_agent.py beverage
    python examples/demo_agent.py "Your custom brief here"
"""

import sys
import os

from openmix.agent import FormulationAgent


BRIEFS = {
    "skincare": (
        "Design a high-performance anti-aging serum with Retinol and Vitamin C "
        "(L-Ascorbic Acid) at effective concentrations. Include copper peptides "
        "for collagen synthesis. Target pH around 3.5. Must have a complete "
        "preservative system and sum to exactly 100%."
    ),
    "supplement": (
        "Design a comprehensive multi-mineral supplement capsule containing "
        "Calcium, Iron, Zinc, and Magnesium at clinically relevant doses. "
        "Include Vitamin D3 and B12 for synergy. Ingredients as percentage "
        "of capsule fill weight, must sum to 100%."
    ),
    "beverage": (
        "Design a vitamin-fortified sparkling water with Ascorbic Acid "
        "(Vitamin C) for immunity, preserved with Sodium Benzoate. "
        "Add green tea extract for antioxidants and iron for energy. "
        "Target pH 3.5. Must sum to 100%."
    ),
}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        brief = BRIEFS.get(arg, arg)
    else:
        brief = BRIEFS["skincare"]

    agent = FormulationAgent(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        max_iterations=5,
        target_score=90.0,
        verbose=True,
    )

    result = agent.run(brief)
    sys.exit(0 if result.converged else 1)


if __name__ == "__main__":
    main()
