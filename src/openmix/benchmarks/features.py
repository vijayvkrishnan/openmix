"""
Feature engineering for FormulaBench.

Three feature tiers, each building on the last:
  Tier 1: Raw composition (ingredient percentages)
  Tier 2: Computed properties (molecular descriptors, aggregated)
  Tier 3: Domain knowledge (OpenMix rules encoded as features)

The hypothesis: Tier 1 < Tier 2 < Tier 3, because domain knowledge
provides inductive bias that raw data and molecular properties don't.
"""

from __future__ import annotations

import numpy as np

from openmix.score import score as compute_score
from openmix.knowledge.loader import load_knowledge, Knowledge
from openmix.matching import match_ingredient
from openmix.benchmarks.shampoo import (
    ShampooRecord, INGREDIENT_COLS, TRADE_TO_TYPE, TRADE_TO_SMILES,
)


def tier1_features(record: ShampooRecord) -> np.ndarray:
    """
    Tier 1: Raw composition.
    18 ingredient percentages + derived ratios.
    """
    percentages = np.array(record.feature_vector, dtype=np.float32)

    # Derived: total active %, number of ingredients, water %
    total_active = percentages.sum()
    n_ingredients = (percentages > 0).sum()
    water_pct = max(0, 100 - total_active)

    # Surfactant type ratios
    anionic_pct = 0.0
    nonionic_pct = 0.0
    amphoteric_pct = 0.0
    cationic_pct = 0.0
    polymer_pct = 0.0
    thickener_pct = 0.0

    for col, pct in zip(INGREDIENT_COLS, percentages):
        if pct == 0:
            continue
        stype = TRADE_TO_TYPE.get(col, "")
        if stype == "anionic":
            anionic_pct += pct
        elif stype == "nonionic":
            nonionic_pct += pct
        elif stype == "amphoteric":
            amphoteric_pct += pct
        elif stype == "cationic":
            cationic_pct += pct
        elif stype == "cationic_polymer":
            polymer_pct += pct
        elif stype == "nonionic_thickener":
            thickener_pct += pct

    # Charge balance: anionic vs cationic (key for shampoo stability)
    total_charged = anionic_pct + cationic_pct + 1e-8
    charge_ratio = (anionic_pct - cationic_pct) / total_charged

    derived = np.array([
        total_active,
        n_ingredients,
        water_pct,
        anionic_pct,
        nonionic_pct,
        amphoteric_pct,
        cationic_pct,
        polymer_pct,
        thickener_pct,
        charge_ratio,
    ], dtype=np.float32)

    return np.concatenate([percentages, derived])


def tier2_features(record: ShampooRecord) -> np.ndarray:
    """
    Tier 2: Tier 1 + molecular descriptors.
    Aggregates molecular properties across the formulation.
    """
    t1 = tier1_features(record)

    # Try to compute molecular features from SMILES
    mol_features = _aggregate_molecular_properties(record)

    return np.concatenate([t1, mol_features])


def tier3_features(record: ShampooRecord, kb: Knowledge | None = None) -> np.ndarray:
    """
    Tier 3: Tier 2 + domain knowledge features.
    Encodes OpenMix rules and scoring as ML features.
    """
    t2 = tier2_features(record)
    kb = kb or load_knowledge()

    formula = record.to_formula()

    # OpenMix stability sub-scores
    stability = compute_score(formula, knowledge=kb)
    score_features = np.array([
        stability.total,
        stability.compatibility,
        stability.ph_suitability,
        stability.emulsion_balance,
        stability.formula_integrity,
        stability.system_completeness,
        len(stability.penalties),
        len(stability.bonuses),
    ], dtype=np.float32)

    # Count interaction rules that fire
    inci_set = formula.inci_names_upper
    hard_violations = 0
    soft_violations = 0
    total_confidence_penalty = 0.0

    for rule in kb.interaction_rules:
        a_match = match_ingredient(rule.a, inci_set, kb.aliases)
        b_match = match_ingredient(rule.b, inci_set, kb.aliases)
        if a_match and b_match and a_match != b_match:
            if rule.rule_type == "hard":
                hard_violations += 1
            else:
                soft_violations += 1
                total_confidence_penalty += rule.confidence

    rule_features = np.array([
        hard_violations,
        soft_violations,
        total_confidence_penalty,
    ], dtype=np.float32)

    return np.concatenate([t2, score_features, rule_features])


def _aggregate_molecular_properties(record: ShampooRecord) -> np.ndarray:
    """Compute weighted molecular descriptor statistics."""
    try:
        from openmix.molecular import compute_properties, is_available
        if not is_available():
            return np.zeros(8, dtype=np.float32)
    except ImportError:
        return np.zeros(8, dtype=np.float32)

    mw_list = []
    logp_list = []
    tpsa_list = []
    weights = []

    for col, pct in zip(INGREDIENT_COLS, record.feature_vector):
        if pct == 0:
            continue
        smiles = TRADE_TO_SMILES.get(col)
        if not smiles:
            continue

        props = compute_properties(smiles)
        if props:
            mw_list.append(props["molecular_weight"])
            logp_list.append(props["log_p"])
            tpsa_list.append(props["tpsa"])
            weights.append(pct)

    if not weights:
        return np.zeros(8, dtype=np.float32)

    weights = np.array(weights)
    weights = weights / weights.sum()

    return np.array([
        np.average(mw_list, weights=weights),      # weighted mean MW
        np.average(logp_list, weights=weights),     # weighted mean LogP
        np.average(tpsa_list, weights=weights),     # weighted mean TPSA
        np.std(mw_list) if len(mw_list) > 1 else 0,   # MW diversity
        np.std(logp_list) if len(logp_list) > 1 else 0, # LogP diversity
        np.min(logp_list),                          # most hydrophilic component
        np.max(logp_list),                          # most hydrophobic component
        np.max(logp_list) - np.min(logp_list),      # hydrophilic-lipophilic span
    ], dtype=np.float32)


# Feature names for interpretability
TIER1_NAMES = (
    INGREDIENT_COLS +
    ["total_active", "n_ingredients", "water_pct",
     "anionic_pct", "nonionic_pct", "amphoteric_pct",
     "cationic_pct", "polymer_pct", "thickener_pct",
     "charge_ratio"]
)

TIER2_NAMES = (
    TIER1_NAMES +
    ["mean_mw", "mean_logp", "mean_tpsa",
     "mw_diversity", "logp_diversity",
     "min_logp", "max_logp", "logp_span"]
)

TIER3_NAMES = (
    TIER2_NAMES +
    ["score_total", "score_compatibility", "score_ph",
     "score_hlb", "score_integrity", "score_completeness",
     "n_penalties", "n_bonuses",
     "hard_violations", "soft_violations", "confidence_penalty"]
)
