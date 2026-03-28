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
from openmix.observe import observe
from openmix.resolver import resolve
from openmix.knowledge.loader import load_knowledge, Knowledge
from openmix.matching import match_ingredient
from openmix.benchmarks.shampoo import (
    ShampooRecord, INGREDIENT_COLS, TRADE_TO_TYPE,
    TRADE_TO_INCI,
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
    """
    Compute weighted molecular descriptor statistics.

    Uses the resolver (seed cache + PubChem) for molecular properties.
    Falls back to RDKit if available for ingredients not in the cache.
    """
    mw_list = []
    logp_list = []
    tpsa_list = []
    weights = []

    for col, pct in zip(INGREDIENT_COLS, record.feature_vector):
        if pct == 0:
            continue
        # Resolve via INCI name (uses seed cache, no network needed for common ingredients)
        inci = TRADE_TO_INCI.get(col, col)
        resolved = resolve(inci)
        if resolved.resolved:
            mw = float(resolved.molecular_weight) if resolved.molecular_weight else None
            logp = resolved.log_p
            tpsa = resolved.tpsa
            if mw is not None:
                mw_list.append(mw)
            if logp is not None:
                logp_list.append(logp)
                weights.append(pct)
            if tpsa is not None:
                tpsa_list.append(tpsa)

    if not weights:
        return np.zeros(8, dtype=np.float32)

    w = np.array(weights)
    w = w / w.sum()

    # Pad tpsa_list to match weights if needed
    tpsa_arr = np.array(tpsa_list[:len(weights)]) if tpsa_list else np.zeros(len(weights))
    mw_arr = np.array(mw_list[:len(weights)]) if mw_list else np.zeros(len(weights))

    return np.array([
        float(np.average(mw_arr, weights=w)) if len(mw_arr) == len(w) else 0,
        float(np.average(logp_list, weights=w)),
        float(np.average(tpsa_arr, weights=w)) if len(tpsa_arr) == len(w) else 0,
        float(np.std(mw_list)) if len(mw_list) > 1 else 0,
        float(np.std(logp_list)) if len(logp_list) > 1 else 0,
        float(np.min(logp_list)),
        float(np.max(logp_list)),
        float(np.max(logp_list) - np.min(logp_list)),
    ], dtype=np.float32)


def tier4_features(record: ShampooRecord, kb: Knowledge | None = None) -> np.ndarray:
    """
    Tier 4: Tier 3 + pairwise interaction features + observation features.

    This is where formulation-awareness matters. Instead of treating
    ingredients independently, we compute features that capture HOW
    ingredients interact — charge compatibility, LogP deltas, and
    concentration-weighted interaction terms.
    """
    t3 = tier3_features(record, kb)
    interaction = _pairwise_interaction_features(record)

    formula = record.to_formula()
    obs = observe(formula)

    obs_features = np.array([
        obs.concern_count,
        obs.hard_violations,
        obs.soft_violations,
        len(obs.concerns),
        obs.resolution_rate,
        obs.concern_score,
        # Charge conflict: binary
        1.0 if any(o.category == "charge" and o.agreement == "discrepancy"
                    for o in obs.observations) else 0.0,
        # Phase concern: hydrophobic phase percentage
        _extract_hydrophobic_pct(obs),
    ], dtype=np.float32)

    return np.concatenate([t3, interaction, obs_features])


def _pairwise_interaction_features(record: ShampooRecord) -> np.ndarray:
    """
    Compute pairwise interaction features between ingredients.

    This is the key insight: formulation stability depends on HOW
    ingredients interact, not just what's present. A cationic surfactant
    at 1% with anionic at 10% is different from cationic at 5% with
    anionic at 5%.

    Features:
    - Cationic-anionic interaction strength (product of concentrations)
    - Cationic polymer × anionic surfactant interaction (coacervation signal)
    - Cationic polymer × cationic surfactant competition
    - Max pairwise charge incompatibility
    - Surfactant diversity (number of distinct surfactant types)
    - Amphoteric buffering ratio (amphoteric / total charged)
    - Thickener-to-surfactant ratio
    """
    # Collect concentrations by surfactant type
    type_pcts: dict[str, float] = {}
    for col, pct in zip(INGREDIENT_COLS, record.feature_vector):
        if pct == 0:
            continue
        stype = TRADE_TO_TYPE.get(col, "unknown")
        type_pcts[stype] = type_pcts.get(stype, 0) + pct

    anionic = type_pcts.get("anionic", 0)
    cationic = type_pcts.get("cationic", 0)
    amphoteric = type_pcts.get("amphoteric", 0)
    nonionic = type_pcts.get("nonionic", 0)
    cat_polymer = type_pcts.get("cationic_polymer", 0)
    thickener = type_pcts.get("nonionic_thickener", 0)

    total_surf = anionic + cationic + amphoteric + nonionic + 1e-8
    total_charged = anionic + cationic + 1e-8

    # Cationic-anionic interaction strength
    # Higher values = more charge interaction (destabilizing unless managed)
    cat_anion_interaction = cationic * anionic

    # Cationic polymer × anionic surfactant (coacervation — the key stability mechanism)
    # This is the primary driver in conditioning shampoos
    coacervation_potential = cat_polymer * anionic

    # Cationic polymer × cationic surfactant (competition for anionic sites)
    cat_competition = cat_polymer * cationic

    # Amphoteric buffering: amphoteric surfactants moderate charge interactions
    amphoteric_buffer = amphoteric / total_charged

    # Surfactant type diversity
    n_types = sum(1 for v in type_pcts.values() if v > 0)

    # Thickener-to-surfactant ratio
    thickener_ratio = thickener / total_surf

    # Nonionic fraction (stabilizing in most systems)
    nonionic_fraction = nonionic / total_surf

    # Polymer loading relative to surfactant (too much or too little matters)
    polymer_loading = cat_polymer / total_surf

    return np.array([
        cat_anion_interaction,
        coacervation_potential,
        cat_competition,
        amphoteric_buffer,
        float(n_types),
        thickener_ratio,
        nonionic_fraction,
        polymer_loading,
    ], dtype=np.float32)


def _extract_hydrophobic_pct(obs) -> float:
    """Extract hydrophobic phase percentage from phase observations."""
    for o in obs.observations:
        if o.category == "phase" and "Hydrophobic phase:" in o.observed:
            try:
                pct_str = o.observed.split(":")[1].split("%")[0].strip()
                return float(pct_str)
            except (IndexError, ValueError):
                pass
    return 0.0


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

TIER4_NAMES = (
    TIER3_NAMES +
    ["cat_anion_interaction", "coacervation_potential", "cat_competition",
     "amphoteric_buffer", "n_surfactant_types", "thickener_ratio",
     "nonionic_fraction", "polymer_loading",
     "obs_concern_count", "obs_hard_violations", "obs_soft_violations",
     "obs_n_concerns", "obs_resolution_rate",
     "obs_concern_score", "obs_charge_conflict", "obs_hydrophobic_pct"]
)
