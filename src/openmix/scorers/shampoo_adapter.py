"""
Shampoo model feature adapter.

Converts a Formula (with INCI names) into the feature vector
the trained shampoo model expects (18 trade-name percentages + derived).
"""

from __future__ import annotations

import numpy as np

from openmix.schema import Formula
from openmix.benchmarks.shampoo import (
    INGREDIENT_COLS, TRADE_TO_INCI, TRADE_TO_TYPE,
)

# Reverse mapping: INCI -> trade name
INCI_TO_TRADE: dict[str, str] = {}
for trade, inci in TRADE_TO_INCI.items():
    inci_upper = inci.upper().strip()
    if inci_upper not in INCI_TO_TRADE:
        INCI_TO_TRADE[inci_upper] = trade


def formula_to_shampoo_features(formula: Formula) -> np.ndarray | None:
    """
    Convert a Formula to the shampoo model's feature vector.

    Returns None if the formula contains no recognizable shampoo ingredients.
    """
    # Map formula ingredients to trade-name columns
    percentages = np.zeros(len(INGREDIENT_COLS), dtype=np.float32)
    matched = 0

    for ing in formula.ingredients:
        inci_upper = ing.inci_name.upper().strip()
        trade = INCI_TO_TRADE.get(inci_upper)
        if trade and trade in INGREDIENT_COLS:
            idx = INGREDIENT_COLS.index(trade)
            percentages[idx] = ing.percentage
            matched += 1

    if matched == 0:
        return None

    # Derived features (same as tier1_features)
    total_active = percentages.sum()
    n_ingredients = (percentages > 0).sum()
    water_pct = max(0, 100 - total_active)

    anionic_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                      if TRADE_TO_TYPE.get(c) == "anionic")
    nonionic_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                       if TRADE_TO_TYPE.get(c) == "nonionic")
    amphoteric_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                         if TRADE_TO_TYPE.get(c) == "amphoteric")
    cationic_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                       if TRADE_TO_TYPE.get(c) == "cationic")
    polymer_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                      if "polymer" in TRADE_TO_TYPE.get(c, ""))
    thickener_pct = sum(p for c, p in zip(INGREDIENT_COLS, percentages)
                        if "thickener" in TRADE_TO_TYPE.get(c, ""))

    total_charged = anionic_pct + cationic_pct + 1e-8
    charge_ratio = (anionic_pct - cationic_pct) / total_charged

    derived = np.array([
        total_active, n_ingredients, water_pct,
        anionic_pct, nonionic_pct, amphoteric_pct,
        cationic_pct, polymer_pct, thickener_pct,
        charge_ratio,
    ], dtype=np.float32)

    return np.concatenate([percentages, derived])
