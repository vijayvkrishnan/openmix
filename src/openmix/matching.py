"""
Ingredient name matching — shared between validate and score.
"""

from __future__ import annotations


def match_ingredient(pair_name: str, inci_set: set[str],
                     aliases: dict[str, list[str]]) -> str | None:
    """
    Check if a pair ingredient name matches any ingredient in the formulation.

    Matching: exact -> alias expansion -> substring.
    """
    if pair_name in inci_set:
        return pair_name

    for alias in aliases.get(pair_name, []):
        for inci in inci_set:
            if inci.startswith(alias) or alias.startswith(inci):
                return inci

    for inci in inci_set:
        if pair_name in inci or inci in pair_name:
            return inci

    return None
