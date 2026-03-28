#!/usr/bin/env python3
"""
MixtureSolDB Baselines — molecular features for mixture solubility prediction.

Tests whether molecular descriptors from the resolver improve solubility
prediction in binary solvent mixtures.

With 810 unique solutes and 135 solvents, the model cannot memorize the
ingredient space. Molecular features are essential for generalization.

Two evaluation protocols:
  1. Random split (standard)
  2. Leave-solutes-out (can we predict solubility for unseen compounds?)

Usage:
    python examples/run_mixture_baselines.py
    python examples/run_mixture_baselines.py --quick   # 10K records for fast iteration
"""

from __future__ import annotations

import sys
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from openmix.benchmarks.mixture_solubility import MixtureSolubility, MixtureSolRecord


# ---------------------------------------------------------------------------
# Molecular property computation
# ---------------------------------------------------------------------------

import json
import time
import urllib.request
import urllib.error
import urllib.parse

_PUBCHEM_INTERVAL = 0.25

# Try RDKit first (fast, local), fall back to PubChem API
try:
    from openmix.molecular import compute_properties, is_available as _rdkit_available
    HAS_RDKIT = _rdkit_available()
except ImportError:
    HAS_RDKIT = False


def _pubchem_by_smiles(smiles: str) -> dict | None:
    """Look up molecular properties from PubChem using SMILES."""
    encoded = urllib.parse.quote(smiles, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded}/property/"
        f"IsomericSMILES,CanonicalSMILES,MolecularWeight,XLogP,"
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
    """Compute molecular properties from SMILES. RDKit if available, else PubChem."""
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


_props_cache: dict[str, dict] = {}


def _get_props(smiles: str) -> dict:
    if smiles not in _props_cache:
        _props_cache[smiles] = _compute_from_smiles(smiles)
    return _props_cache[smiles]


def composition_features(record: MixtureSolRecord) -> np.ndarray:
    """
    Tier 1: Composition only.
    Solvent fractions + temperature. No molecular information.
    """
    return np.array([
        record.solvent1_fraction,
        record.solvent2_fraction,
        record.temperature_k,
    ], dtype=np.float32)


def molecular_features(record: MixtureSolRecord) -> np.ndarray:
    """
    Tier 2: Composition + molecular descriptors for all three components.

    For each component (solute, solvent1, solvent2), extract:
    LogP, MW, TPSA, HBD, HBA. Then compute interaction terms:
    LogP differences, MW ratios, and mixture-weighted properties.
    """
    t1 = composition_features(record)

    solute = _get_props(record.solute_smiles)
    solv1 = _get_props(record.solvent1_smiles)
    solv2 = _get_props(record.solvent2_smiles)

    def _safe(d, key, default=0.0):
        v = d.get(key)
        return float(v) if v is not None else default

    # Individual molecular descriptors (5 per component = 15)
    solute_feats = [
        _safe(solute, "log_p"),
        _safe(solute, "mw"),
        _safe(solute, "tpsa"),
        _safe(solute, "hbd"),
        _safe(solute, "hba"),
    ]
    solv1_feats = [
        _safe(solv1, "log_p"),
        _safe(solv1, "mw"),
        _safe(solv1, "tpsa"),
        _safe(solv1, "hbd"),
        _safe(solv1, "hba"),
    ]
    solv2_feats = [
        _safe(solv2, "log_p"),
        _safe(solv2, "mw"),
        _safe(solv2, "tpsa"),
        _safe(solv2, "hbd"),
        _safe(solv2, "hba"),
    ]

    # Interaction features (the formulation-aware part)
    solute_logp = _safe(solute, "log_p")
    solv1_logp = _safe(solv1, "log_p")
    solv2_logp = _safe(solv2, "log_p")
    f1 = record.solvent1_fraction
    f2 = record.solvent2_fraction

    # Weighted solvent LogP (the mixture's effective polarity)
    mix_logp = solv1_logp * f1 + solv2_logp * f2

    # LogP delta: solute vs mixture (like dissolves like)
    logp_delta = abs(solute_logp - mix_logp)

    # LogP delta: between the two solvents (mixture heterogeneity)
    solvent_logp_span = abs(solv1_logp - solv2_logp)

    # TPSA-based polarity matching
    solute_tpsa = _safe(solute, "tpsa")
    mix_tpsa = _safe(solv1, "tpsa") * f1 + _safe(solv2, "tpsa") * f2
    tpsa_delta = abs(solute_tpsa - mix_tpsa)

    # H-bond complementarity
    solute_hbd = _safe(solute, "hbd")
    solute_hba = _safe(solute, "hba")
    mix_hba = _safe(solv1, "hba") * f1 + _safe(solv2, "hba") * f2
    mix_hbd = _safe(solv1, "hbd") * f1 + _safe(solv2, "hbd") * f2
    # Donor-acceptor complementarity: solute donors need solvent acceptors
    hbond_complement = solute_hbd * mix_hba + solute_hba * mix_hbd

    interaction_feats = [
        mix_logp,
        logp_delta,
        solvent_logp_span,
        tpsa_delta,
        hbond_complement,
    ]

    all_feats = (
        list(t1)
        + solute_feats + solv1_feats + solv2_feats
        + interaction_feats
    )
    return np.array(all_feats, dtype=np.float32)


COMPOSITION_NAMES = [
    "solv1_frac", "solv2_frac", "temperature_k",
]

MOLECULAR_NAMES = (
    COMPOSITION_NAMES
    + ["solute_logp", "solute_mw", "solute_tpsa", "solute_hbd", "solute_hba"]
    + ["solv1_logp", "solv1_mw", "solv1_tpsa", "solv1_hbd", "solv1_hba"]
    + ["solv2_logp", "solv2_mw", "solv2_tpsa", "solv2_hbd", "solv2_hba"]
    + ["mix_logp", "logp_delta", "solvent_logp_span",
       "tpsa_delta", "hbond_complement"]
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(X_train, y_train, X_test, y_test, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "model": model,
    }


def run():
    quick = "--quick" in sys.argv
    max_records = 10000 if quick else None

    print("Loading MixtureSolDB...", flush=True)
    ds = MixtureSolubility(binary_only=True, max_records=max_records)
    print(ds)
    print()

    # Precompute molecular properties for all unique SMILES
    all_smiles = set()
    for r in ds.records:
        all_smiles.add(r.solute_smiles)
        all_smiles.add(r.solvent1_smiles)
        all_smiles.add(r.solvent2_smiles)

    print(f"Resolving {len(all_smiles)} unique molecules...", end=" ", flush=True)
    t0 = time.time()
    resolved = 0
    for smi in all_smiles:
        props = _get_props(smi)
        if props:
            resolved += 1
    print(f"{resolved}/{len(all_smiles)} resolved ({time.time()-t0:.1f}s)")
    print()

    # ===== Protocol 1: Random Split =====
    print("=" * 70)
    print("  PROTOCOL 1: Random Split")
    print("=" * 70)
    print()

    train, val, test = ds.split_random(test_size=0.1, val_size=0.1, seed=42)
    print(f"  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    print()

    tiers = {
        "Tier 1: Composition Only": (composition_features, COMPOSITION_NAMES),
        "Tier 2: + Molecular Descriptors": (molecular_features, MOLECULAR_NAMES),
    }

    random_results = {}
    for name, (feat_fn, feat_names) in tiers.items():
        print(f"  Building features for {name}...", flush=True)
        X_train = np.array([feat_fn(r) for r in train], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test], dtype=np.float32)
        y_train = np.array([r.log_solubility for r in train], dtype=np.float32)
        y_test = np.array([r.log_solubility for r in test], dtype=np.float32)

        result = evaluate(X_train, y_train, X_test, y_test)
        random_results[name] = result

        # Top features
        importances = result["model"].feature_importances_
        top5 = np.argsort(importances)[::-1][:5]
        top_feats = [(feat_names[i], importances[i]) for i in top5 if i < len(feat_names)]

        print(f"    Features: {X_train.shape[1]}")
        print(f"    MAE:  {result['mae']:.4f}")
        print(f"    R^2:  {result['r2']:.4f}")
        print(f"    Top:  {', '.join(f'{n} ({v:.3f})' for n, v in top_feats[:3])}")
        print()

    # ===== Protocol 2: Leave-Solutes-Out =====
    print("=" * 70)
    print("  PROTOCOL 2: Leave-Solutes-Out (generalization to unseen compounds)")
    print("=" * 70)
    print()

    n_hold = max(10, ds.unique_solutes // 10)  # Hold out 10% of solutes
    train_lso, test_lso = ds.split_leave_solutes_out(n_held_out=n_hold, seed=42)
    test_solutes = len(set(r.solute_smiles for r in test_lso))
    print(f"  Held out: {test_solutes} solutes ({n_hold} requested)")
    print(f"  Train: {len(train_lso)}  |  Test: {len(test_lso)}")
    print()

    lso_results = {}
    for name, (feat_fn, feat_names) in tiers.items():
        print(f"  Building features for {name}...", flush=True)
        X_train = np.array([feat_fn(r) for r in train_lso], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test_lso], dtype=np.float32)
        y_train = np.array([r.log_solubility for r in train_lso], dtype=np.float32)
        y_test = np.array([r.log_solubility for r in test_lso], dtype=np.float32)

        result = evaluate(X_train, y_train, X_test, y_test)
        lso_results[name] = result

        print(f"    MAE:  {result['mae']:.4f}")
        print(f"    R^2:  {result['r2']:.4f}")
        print()

    # ===== Summary =====
    print("=" * 70)
    print(f"  {'Model':<42} {'Random':>10} {'LSO':>10}")
    print(f"  {'':42} {'MAE / R2':>10} {'MAE / R2':>10}")
    print("  " + "-" * 62)
    for name in tiers:
        r = random_results[name]
        l = lso_results[name]
        print(f"  {name:<42} "
              f"{r['mae']:.3f}/{r['r2']:.3f}  "
              f"{l['mae']:.3f}/{l['r2']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
