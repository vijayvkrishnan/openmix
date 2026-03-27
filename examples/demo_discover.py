#!/usr/bin/env python3
"""
OpenMix Discovery Engine — autonomous rule discovery from formulation data.

The agent proposes hypotheses about what makes formulations stable or unstable,
tests them against real experimental data, and iterates to find design principles.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/demo_discover.py
"""

import os
import sys

from openmix.discover import DiscoveryEngine
from openmix.benchmarks import ShampooStability


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    ds = ShampooStability()

    engine = DiscoveryEngine(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        max_iterations=4,
        verbose=True,
    )

    result = engine.run(ds)
    sys.exit(0 if result.discoveries else 1)


if __name__ == "__main__":
    main()
