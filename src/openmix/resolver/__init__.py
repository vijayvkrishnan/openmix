"""
Ingredient resolver — INCI name to molecular properties.

Resolves any ingredient to its molecular identity and physicochemical
properties through a three-tier lookup:
  1. Local cache (instant, ~2K common ingredients ship with the package)
  2. Extended cache (downloaded on first use, grows over time)
  3. PubChem API (runtime fallback for unknown ingredients)

    from openmix.resolver import resolve

    props = resolve("Niacinamide")
    # {'smiles': 'c1ccc(c(c1)C(=O)N)N', 'log_p': -0.35, 'mw': 122.12, ...}
"""

from openmix.resolver.resolve import resolve, resolve_many, ResolvedIngredient

__all__ = ["resolve", "resolve_many", "ResolvedIngredient"]
