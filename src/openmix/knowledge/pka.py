"""
pKa-based pH reasoning for formulation science.

Given an ingredient's pKa and the formula's target pH, computes the
ionization fraction via Henderson-Hasselbalch and reports whether the
pH is favorable for the ingredient's intended function.

    fraction_ionized = 1 / (1 + 10^(pKa - pH))   # for acids
    fraction_ionized = 1 / (1 + 10^(pH - pKa))   # for bases

This is standard physical chemistry applied to formulation. The pKa
data comes from published literature values, not estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


PKA_DATA_PATH = Path(__file__).parent / "data" / "pka_data.yaml"


@dataclass
class IngredientPKa:
    """pKa data for a single ingredient."""
    inci_name: str
    pka: list[float]            # one or more pKa values (polyprotic)
    acid_or_base: str           # "acid", "base", or "amphoteric"
    optimal_ph_min: float       # optimal pH range lower bound
    optimal_ph_max: float       # optimal pH range upper bound
    concern_below: Optional[str] = None   # what happens below optimal range
    concern_above: Optional[str] = None   # what happens above optimal range
    source: str = ""


def load_pka_data(data_path: str | Path | None = None) -> dict[str, IngredientPKa]:
    """Load pKa database from YAML. Returns dict keyed by uppercased INCI name."""
    path = Path(data_path) if data_path else PKA_DATA_PATH
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    result = {}
    for entry in raw:
        pka_vals = entry.get("pka", [])
        if isinstance(pka_vals, (int, float)):
            pka_vals = [float(pka_vals)]

        item = IngredientPKa(
            inci_name=entry["name"],
            pka=pka_vals,
            acid_or_base=entry.get("type", "acid"),
            optimal_ph_min=entry.get("optimal_ph_min", 0),
            optimal_ph_max=entry.get("optimal_ph_max", 14),
            concern_below=entry.get("concern_below"),
            concern_above=entry.get("concern_above"),
            source=entry.get("source", ""),
        )
        result[item.inci_name.upper().strip()] = item

    return result


def ionization_fraction(pka: float, ph: float, acid: bool = True) -> float:
    """
    Compute the fraction of molecules in ionized (deprotonated) form.

    For an acid HA ⇌ H+ + A-:
        fraction_ionized = 1 / (1 + 10^(pKa - pH))

    For a base B + H2O ⇌ BH+ + OH-:
        fraction_protonated = 1 / (1 + 10^(pH - pKa))
        (returns fraction in charged BH+ form)

    Returns a value between 0.0 and 1.0.
    """
    try:
        if acid:
            return 1.0 / (1.0 + math.pow(10, pka - ph))
        else:
            return 1.0 / (1.0 + math.pow(10, ph - pka))
    except OverflowError:
        return 0.0 if (pka - ph > 0 and acid) else 1.0


def assess_ph_suitability(
    ingredient: IngredientPKa,
    target_ph: float,
) -> dict:
    """
    Assess whether the target pH is suitable for this ingredient.

    Returns dict with:
        suitable: bool
        ionized_fraction: float (for the primary pKa)
        detail: str (human-readable explanation)
    """
    if not ingredient.pka:
        return {"suitable": True, "ionized_fraction": None, "detail": ""}

    primary_pka = ingredient.pka[0]
    is_acid = ingredient.acid_or_base in ("acid", "amphoteric")
    frac = ionization_fraction(primary_pka, target_ph, acid=is_acid)

    # Check if pH is in the optimal range
    in_range = ingredient.optimal_ph_min <= target_ph <= ingredient.optimal_ph_max

    if in_range:
        if is_acid:
            state = f"{(1 - frac) * 100:.0f}% non-ionized" if frac < 0.5 else f"{frac * 100:.0f}% ionized"
        else:
            state = f"{frac * 100:.0f}% protonated" if frac > 0.5 else f"{(1 - frac) * 100:.0f}% deprotonated"

        return {
            "suitable": True,
            "ionized_fraction": frac,
            "detail": (
                f"{ingredient.inci_name} at pH {target_ph}: {state} "
                f"(pKa {primary_pka}). Within optimal range "
                f"{ingredient.optimal_ph_min}-{ingredient.optimal_ph_max}."
            ),
        }

    # Outside optimal range
    if target_ph < ingredient.optimal_ph_min:
        concern = ingredient.concern_below or "below optimal pH range"
    else:
        concern = ingredient.concern_above or "above optimal pH range"

    if is_acid:
        state = f"{(1 - frac) * 100:.0f}% non-ionized" if frac < 0.5 else f"{frac * 100:.0f}% ionized"
    else:
        state = f"{frac * 100:.0f}% protonated" if frac > 0.5 else f"{(1 - frac) * 100:.0f}% deprotonated"

    return {
        "suitable": False,
        "ionized_fraction": frac,
        "detail": (
            f"{ingredient.inci_name} at pH {target_ph}: {state} "
            f"(pKa {primary_pka}). {concern.rstrip('.')}. "
            f"Optimal pH: {ingredient.optimal_ph_min}-{ingredient.optimal_ph_max}."
        ),
    }
