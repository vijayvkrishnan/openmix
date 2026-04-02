#!/usr/bin/env python3
"""
Thermodynamic Interaction Features -- Breaking the 0.52 LSO Ceiling.

Hypothesis: Thermodynamically motivated features (squared differences from
regular solution theory, pseudo-Hansen distance, temperature scaling) improve
out-of-distribution generalization in mixture solubility prediction.

Two approaches:
  1. Add thermodynamic features alongside standard molecular descriptors
  2. Residual learning: fit a physics model first, ML learns the corrections

Evaluation: Leave-solutes-out (LSO) on MixtureSolDB (146K records, 807 solutes).
Baseline: R^2 0.52 with 23 standard molecular features + XGBoost.

Usage:
    python examples/run_thermodynamic_experiment.py
    python examples/run_thermodynamic_experiment.py --quick  # 10K records
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from openmix.benchmarks.mixture_solubility import MixtureSolubility, MixtureSolRecord

# ---------------------------------------------------------------------------
# Molecular property computation (RDKit preferred, PubChem fallback)
# ---------------------------------------------------------------------------

try:
    from openmix.molecular import compute_properties, is_available as _rdkit_check
    HAS_RDKIT = _rdkit_check()
except ImportError:
    HAS_RDKIT = False

_PUBCHEM_INTERVAL = 0.25
_props_cache: dict[str, dict] = {}


def _pubchem_by_smiles(smiles: str) -> dict | None:
    encoded = urllib.parse.quote(smiles, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded}/property/"
        f"IsomericSMILES,MolecularWeight,XLogP,"
        f"HBondDonorCount,HBondAcceptorCount,TPSA/JSON"
    )
    try:
        time.sleep(_PUBCHEM_INTERVAL)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            p = props[0]
            return {
                "log_p": p.get("XLogP"),
                "mw": p.get("MolecularWeight"),
                "tpsa": p.get("TPSA"),
                "hbd": p.get("HBondDonorCount"),
                "hba": p.get("HBondAcceptorCount"),
            }
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, OSError):
        pass
    return None


def _compute_from_smiles(smiles: str) -> dict:
    if HAS_RDKIT:
        props = compute_properties(smiles)
        if props:
            return {
                "log_p": props.get("log_p"),
                "mw": props.get("molecular_weight"),
                "tpsa": props.get("tpsa"),
                "hbd": props.get("hbd"),
                "hba": props.get("hba"),
            }
    result = _pubchem_by_smiles(smiles)
    return result or {}


def _get_props(smiles: str) -> dict:
    if smiles not in _props_cache:
        _props_cache[smiles] = _compute_from_smiles(smiles)
    return _props_cache[smiles]


def _safe(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key)
    return float(v) if v is not None else default


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def _mixture_props(record: MixtureSolRecord) -> tuple[dict, dict, dict, float, float]:
    """Extract properties for solute and both solvents."""
    solute = _get_props(record.solute_smiles)
    solv1 = _get_props(record.solvent1_smiles)
    solv2 = _get_props(record.solvent2_smiles)
    return solute, solv1, solv2, record.solvent1_fraction, record.solvent2_fraction


def baseline_features(record: MixtureSolRecord) -> np.ndarray:
    """
    Baseline: 23 molecular features (reproduces the 0.52 LSO result).

    3 composition + 15 individual descriptors + 5 interaction terms.
    """
    solute, solv1, solv2, f1, f2 = _mixture_props(record)

    comp = [f1, f2, record.temperature_k]

    solute_f = [_safe(solute, k) for k in ("log_p", "mw", "tpsa", "hbd", "hba")]
    solv1_f = [_safe(solv1, k) for k in ("log_p", "mw", "tpsa", "hbd", "hba")]
    solv2_f = [_safe(solv2, k) for k in ("log_p", "mw", "tpsa", "hbd", "hba")]

    # Interaction features (absolute differences)
    solute_logp = _safe(solute, "log_p")
    mix_logp = _safe(solv1, "log_p") * f1 + _safe(solv2, "log_p") * f2
    mix_tpsa = _safe(solv1, "tpsa") * f1 + _safe(solv2, "tpsa") * f2
    mix_hba = _safe(solv1, "hba") * f1 + _safe(solv2, "hba") * f2
    mix_hbd = _safe(solv1, "hbd") * f1 + _safe(solv2, "hbd") * f2

    interactions = [
        mix_logp,
        abs(solute_logp - mix_logp),                            # logp_delta
        abs(_safe(solv1, "log_p") - _safe(solv2, "log_p")),    # solvent_logp_span
        abs(_safe(solute, "tpsa") - mix_tpsa),                  # tpsa_delta
        _safe(solute, "hbd") * mix_hba + _safe(solute, "hba") * mix_hbd,  # hbond_complement
    ]

    return np.array(comp + solute_f + solv1_f + solv2_f + interactions, dtype=np.float32)


def thermodynamic_features(record: MixtureSolRecord) -> np.ndarray:
    """
    Physics-informed interaction features from regular solution theory.

    Regular solution theory: ΔH_mix ∝ (δ₁ - δ₂)²
    Hansen solubility: d² = 4(δD₁-δD₂)² + (δP₁-δP₂)² + (δH₁-δH₂)²

    We approximate Hansen components from molecular descriptors:
      δD (dispersion) ← LogP
      δP (polar) ← TPSA
      δH (H-bonding) ← HBD/HBA mismatch

    Returns 8 thermodynamic features.
    """
    solute, solv1, solv2, f1, f2 = _mixture_props(record)
    T = record.temperature_k

    solute_logp = _safe(solute, "log_p")
    solute_tpsa = _safe(solute, "tpsa")
    solute_mw = _safe(solute, "mw")
    solute_hbd = _safe(solute, "hbd")
    solute_hba = _safe(solute, "hba")

    mix_logp = _safe(solv1, "log_p") * f1 + _safe(solv2, "log_p") * f2
    mix_tpsa = _safe(solv1, "tpsa") * f1 + _safe(solv2, "tpsa") * f2
    mix_mw = _safe(solv1, "mw") * f1 + _safe(solv2, "mw") * f2
    mix_hbd = _safe(solv1, "hbd") * f1 + _safe(solv2, "hbd") * f2
    mix_hba = _safe(solv1, "hba") * f1 + _safe(solv2, "hba") * f2

    # Squared differences (regular solution theory: ΔH ∝ Δδ²)
    logp_diff_sq = (solute_logp - mix_logp) ** 2
    tpsa_diff_sq = (solute_tpsa - mix_tpsa) ** 2

    # H-bond mismatch: solute donors need solvent acceptors, and vice versa
    hb_mismatch = (solute_hbd - mix_hba) ** 2 + (solute_hba - mix_hbd) ** 2

    # H-bond complementarity (product form, not mismatch)
    hb_complement = solute_hbd * mix_hba + solute_hba * mix_hbd

    # Pseudo-Hansen distance: 4:1:1 weighting (Hansen's empirical finding)
    hansen_dist_sq = 4.0 * logp_diff_sq + tpsa_diff_sq + hb_mismatch
    hansen_dist = np.sqrt(hansen_dist_sq)

    # Size ratio (molar volume proxy)
    mw_ratio = solute_mw / (mix_mw + 1e-8)

    # Temperature scaling: -RT ln(S) = ΔG_mix, so ln(S) ∝ -ΔH/T
    hansen_over_T = hansen_dist / T

    return np.array([
        logp_diff_sq,       # dispersion mismatch²
        tpsa_diff_sq,       # polarity mismatch²
        hb_mismatch,        # H-bond mismatch²
        hb_complement,      # H-bond complementarity
        mw_ratio,           # size compatibility
        hansen_dist,         # pseudo-Hansen distance
        hansen_over_T,       # temperature-scaled Hansen
        np.log(T),          # entropy term
    ], dtype=np.float32)


def combined_features(record: MixtureSolRecord) -> np.ndarray:
    """Baseline molecular features + thermodynamic features."""
    return np.concatenate([baseline_features(record), thermodynamic_features(record)])


BASELINE_NAMES = [
    "solv1_frac", "solv2_frac", "temperature_k",
    "solute_logp", "solute_mw", "solute_tpsa", "solute_hbd", "solute_hba",
    "solv1_logp", "solv1_mw", "solv1_tpsa", "solv1_hbd", "solv1_hba",
    "solv2_logp", "solv2_mw", "solv2_tpsa", "solv2_hbd", "solv2_hba",
    "mix_logp", "logp_delta", "solvent_logp_span", "tpsa_delta", "hbond_complement",
]

THERMO_NAMES = [
    "logp_diff_sq", "tpsa_diff_sq", "hb_mismatch", "hb_complement_thermo",
    "mw_ratio", "hansen_dist", "hansen_over_T", "log_T",
]


# ---------------------------------------------------------------------------
# Physics model
# ---------------------------------------------------------------------------

def fit_physics_model(X_thermo_train, y_train):
    """
    Linear physics model: LogS ~= f(thermodynamic features).

    Uses Ridge regression (slight regularization) on 8 thermodynamic
    features. This is the physics prior -- it captures the dominant
    trend (like-dissolves-like, temperature dependence) without any
    compound-specific information.
    """
    model = Ridge(alpha=1.0)
    model.fit(X_thermo_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def eval_model(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    quick = "--quick" in sys.argv
    max_records = 10_000 if quick else None

    print("=" * 70)
    print("  THERMODYNAMIC INTERACTION FEATURES -- LSO EXPERIMENT")
    print("=" * 70)
    print()

    # Load data
    print("Loading MixtureSolDB...", end=" ", flush=True)
    ds = MixtureSolubility(binary_only=True, max_records=max_records)
    print(ds)

    # Resolve molecular properties
    all_smiles = set()
    for r in ds.records:
        all_smiles.add(r.solute_smiles)
        all_smiles.add(r.solvent1_smiles)
        all_smiles.add(r.solvent2_smiles)

    print(f"Resolving {len(all_smiles)} unique molecules...", end=" ", flush=True)
    t0 = time.time()
    resolved = sum(1 for smi in all_smiles if _get_props(smi))
    print(f"{resolved}/{len(all_smiles)} ({time.time()-t0:.1f}s)")
    print()

    # LSO split (same protocol as baseline: 10% of solutes held out, seed 42)
    n_hold = max(10, ds.unique_solutes // 10)
    train_recs, test_recs = ds.split_leave_solutes_out(n_held_out=n_hold, seed=42)
    test_solutes = len(set(r.solute_smiles for r in test_recs))

    print(f"Leave-Solutes-Out: {test_solutes} held-out solutes")
    print(f"Train: {len(train_recs)}  |  Test: {len(test_recs)}")
    print()

    # Build feature matrices
    print("Building features...", end=" ", flush=True)
    t0 = time.time()

    X_base_train = np.array([baseline_features(r) for r in train_recs], dtype=np.float32)
    X_base_test = np.array([baseline_features(r) for r in test_recs], dtype=np.float32)

    X_thermo_train = np.array([thermodynamic_features(r) for r in train_recs], dtype=np.float32)
    X_thermo_test = np.array([thermodynamic_features(r) for r in test_recs], dtype=np.float32)

    X_combined_train = np.hstack([X_base_train, X_thermo_train])
    X_combined_test = np.hstack([X_base_test, X_thermo_test])

    y_train = np.array([r.log_solubility for r in train_recs], dtype=np.float32)
    y_test = np.array([r.log_solubility for r in test_recs], dtype=np.float32)

    print(f"done ({time.time()-t0:.1f}s)")
    print()

    results = {}

    # ----- Model A: Baseline (molecular features -> XGBoost) -----
    print("  [A] Baseline: 23 molecular features -> XGBoost")
    model_a = train_xgb(X_base_train, y_train)
    preds_a = model_a.predict(X_base_test)
    results["A: Baseline (23 mol)"] = eval_model(y_test, preds_a)
    print(f"      R^2: {results['A: Baseline (23 mol)']['r2']:.4f}  "
          f"MAE: {results['A: Baseline (23 mol)']['mae']:.4f}")

    # Feature importance for baseline
    imp_a = model_a.feature_importances_
    top3_a = np.argsort(imp_a)[::-1][:3]
    print(f"      Top features: {', '.join(BASELINE_NAMES[i] for i in top3_a)}")
    print()

    # ----- Model B: Baseline + thermodynamic features -> XGBoost -----
    print("  [B] + Thermodynamic features: 31 features -> XGBoost")
    model_b = train_xgb(X_combined_train, y_train)
    preds_b = model_b.predict(X_combined_test)
    results["B: + Thermo (31)"] = eval_model(y_test, preds_b)
    print(f"      R^2: {results['B: + Thermo (31)']['r2']:.4f}  "
          f"MAE: {results['B: + Thermo (31)']['mae']:.4f}")

    # Feature importance for combined
    all_names = BASELINE_NAMES + THERMO_NAMES
    imp_b = model_b.feature_importances_
    top5_b = np.argsort(imp_b)[::-1][:5]
    print(f"      Top features: {', '.join(all_names[i] for i in top5_b)}")

    # How much importance do thermo features get?
    thermo_importance = imp_b[len(BASELINE_NAMES):].sum()
    print(f"      Thermo feature importance: {thermo_importance:.1%}")
    print()

    # ----- Model C: Physics model only (linear on thermodynamic features) -----
    print("  [C] Physics model only: Ridge on 8 thermodynamic features")
    physics_model = fit_physics_model(X_thermo_train, y_train)
    preds_c = physics_model.predict(X_thermo_test)
    results["C: Physics only (8)"] = eval_model(y_test, preds_c)
    print(f"      R^2: {results['C: Physics only (8)']['r2']:.4f}  "
          f"MAE: {results['C: Physics only (8)']['mae']:.4f}")

    # Physics model coefficients (interpretable)
    print("      Coefficients:")
    for name, coef in zip(THERMO_NAMES, physics_model.coef_):
        if abs(coef) > 0.001:
            print(f"        {name:<25} {coef:+.4f}")
    print(f"        {'intercept':<25} {physics_model.intercept_:+.4f}")
    print()

    # ----- Model D: Stacked (molecular + physics prediction -> XGBoost) -----
    print("  [D] Stacked: molecular features + physics prediction -> XGBoost")
    physics_pred_train = physics_model.predict(X_thermo_train).reshape(-1, 1)
    physics_pred_test = physics_model.predict(X_thermo_test).reshape(-1, 1)
    X_stacked_train = np.hstack([X_base_train, physics_pred_train])
    X_stacked_test = np.hstack([X_base_test, physics_pred_test])

    model_d = train_xgb(X_stacked_train, y_train)
    preds_d = model_d.predict(X_stacked_test)
    results["D: Stacked (24)"] = eval_model(y_test, preds_d)
    print(f"      R^2: {results['D: Stacked (24)']['r2']:.4f}  "
          f"MAE: {results['D: Stacked (24)']['mae']:.4f}")

    # How much importance does the physics prediction get?
    physics_pred_imp = model_d.feature_importances_[-1]
    print(f"      Physics prediction importance: {physics_pred_imp:.1%}")
    print()

    # ----- Model E: Residual learning (physics + ML on residuals) -----
    print("  [E] Residual: physics prediction + XGBoost on residuals")
    residuals_train = y_train - physics_model.predict(X_thermo_train)
    model_e = train_xgb(X_base_train, residuals_train)
    residual_preds = model_e.predict(X_base_test)
    preds_e = preds_c + residual_preds  # physics + correction
    results["E: Residual"] = eval_model(y_test, preds_e)
    print(f"      R^2: {results['E: Residual']['r2']:.4f}  "
          f"MAE: {results['E: Residual']['mae']:.4f}")

    # Decomposition: how much does each component contribute?
    r2_physics_component = r2_score(y_test, preds_c)
    r2_residual_only = r2_score(y_test - preds_c, residual_preds)
    print(f"      Physics component R^2: {r2_physics_component:.4f}")
    print(f"      Residual model R^2 (on residuals): {r2_residual_only:.4f}")
    print()

    # ===== Summary =====
    print("=" * 70)
    print("  RESULTS SUMMARY (Leave-Solutes-Out)")
    print("=" * 70)
    print()
    print(f"  {'Model':<35} {'R^2':>8} {'MAE':>8} {'vs Baseline':>12}")
    print("  " + "-" * 63)

    baseline_r2 = results["A: Baseline (23 mol)"]["r2"]
    for name, r in results.items():
        delta = r["r2"] - baseline_r2
        delta_str = f"{delta:+.4f}" if name != "A: Baseline (23 mol)" else "---"
        print(f"  {name:<35} {r['r2']:>8.4f} {r['mae']:>8.4f} {delta_str:>12}")

    print()
    print("=" * 70)

    # Interpretation
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_r2 = results[best_name]["r2"]
    if best_r2 > baseline_r2 + 0.01:
        print(f"\n  Result: {best_name} improved LSO R^2 by {best_r2 - baseline_r2:+.4f}")
        print("  Thermodynamic priors improve out-of-distribution generalization.")
    elif best_r2 > baseline_r2:
        print(f"\n  Result: Marginal improvement ({best_r2 - baseline_r2:+.4f}). "
              "Not conclusive.")
    else:
        print("\n  Result: No improvement from thermodynamic features on LSO.")
        print("  The 0.52 ceiling may be an information-theoretic limit of")
        print("  descriptor-based prediction for this dataset.")


if __name__ == "__main__":
    run()
