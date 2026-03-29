#!/usr/bin/env python3
"""
Drug-Excipient Compatibility Baselines — fingerprint and molecular features.

Tests whether molecular descriptors from CID resolution improve compatibility
prediction beyond raw PubChem fingerprints.

With 470 unique drugs and 266 unique excipients, the leave-drugs-out split
tests whether the model can generalize to drugs it has never seen — which is
the practical question for pharmaceutical formulators.

Two feature tiers:
  1. Fingerprint-only: concatenated PubChem fingerprint bits (1,762 features)
  2. + Molecular descriptors: LogP, MW, TPSA, HBD, HBA for both drug and
     excipient, plus interaction terms (LogP delta, TPSA delta, MW ratio,
     H-bond complementarity)

Two evaluation protocols:
  1. Random stratified split (standard)
  2. Leave-drugs-out (can we predict compatibility for unseen drugs?)

Dataset notes:
  - Highly imbalanced: 83.5% compatible / 16.5% incompatible
  - We report AUROC (insensitive to class balance) alongside accuracy
  - PubChem fingerprints are 881-bit CACTVS substructure keys

Usage:
    python examples/run_drug_excipient_baselines.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from openmix.benchmarks.drug_excipient import (
    DrugExcipientCompatibility,
    DrugExcipientRecord,
    FP_BITS,
)


# ---------------------------------------------------------------------------
# Molecular property resolution from PubChem CID
# ---------------------------------------------------------------------------

_PUBCHEM_INTERVAL = 0.25

# Try RDKit first (fast, local), fall back to PubChem API
try:
    from openmix.molecular import compute_properties, is_available as _rdkit_available
    HAS_RDKIT = _rdkit_available()
except ImportError:
    HAS_RDKIT = False


def _pubchem_props_by_cid(cid: int) -> dict | None:
    """Look up molecular properties from PubChem using CID."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/property/"
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
                "smiles": p.get("IsomericSMILES"),
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


def _rdkit_props_from_smiles(smiles: str) -> dict:
    """Compute properties from SMILES via RDKit (if available)."""
    if not HAS_RDKIT:
        return {}
    props = compute_properties(smiles)
    if props:
        return {
            "log_p": props.get("log_p"),
            "mw": props.get("molecular_weight"),
            "tpsa": props.get("tpsa"),
            "hbd": props.get("hbd"),
            "hba": props.get("hba"),
        }
    return {}


_cid_cache: dict[int, dict] = {}


def _get_cid_props(cid: int) -> dict:
    """Get molecular properties for a PubChem CID (cached)."""
    if cid not in _cid_cache:
        props = _pubchem_props_by_cid(cid)
        if props:
            _cid_cache[cid] = props
            # Try RDKit for additional/more reliable descriptors
            smiles = props.get("smiles")
            if smiles and HAS_RDKIT:
                rdkit_props = _rdkit_props_from_smiles(smiles)
                # RDKit values override PubChem if available (more precise)
                for key in ("log_p", "mw", "tpsa", "hbd", "hba"):
                    if rdkit_props.get(key) is not None:
                        _cid_cache[cid][key] = rdkit_props[key]
        else:
            _cid_cache[cid] = {}
    return _cid_cache[cid]


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def fingerprint_features(record: DrugExcipientRecord) -> np.ndarray:
    """
    Tier 1: Fingerprint-only features.

    Concatenated PubChem CACTVS fingerprint bits for drug and excipient.
    881 bits each = 1,762 total features.
    """
    return np.array(record.feature_vector, dtype=np.float32)


def molecular_features(record: DrugExcipientRecord) -> np.ndarray:
    """
    Tier 2: Fingerprints + molecular descriptors + interaction terms.

    For each compound (drug, excipient), extract:
    LogP, MW, TPSA, HBD, HBA. Then compute interaction terms:
    LogP delta, TPSA delta, MW ratio, H-bond complementarity.
    """
    fp = fingerprint_features(record)

    drug_props = _get_cid_props(record.drug_cid)
    exc_props = _get_cid_props(record.excipient_cid)

    def _safe(d: dict, key: str, default: float = 0.0) -> float:
        v = d.get(key)
        return float(v) if v is not None else default

    # Individual molecular descriptors (5 per compound = 10)
    drug_logp = _safe(drug_props, "log_p")
    drug_mw = _safe(drug_props, "mw")
    drug_tpsa = _safe(drug_props, "tpsa")
    drug_hbd = _safe(drug_props, "hbd")
    drug_hba = _safe(drug_props, "hba")

    exc_logp = _safe(exc_props, "log_p")
    exc_mw = _safe(exc_props, "mw")
    exc_tpsa = _safe(exc_props, "tpsa")
    exc_hbd = _safe(exc_props, "hbd")
    exc_hba = _safe(exc_props, "hba")

    # Interaction features
    # LogP delta: polarity mismatch (like dissolves like)
    logp_delta = abs(drug_logp - exc_logp)

    # TPSA delta: polar surface area mismatch
    tpsa_delta = abs(drug_tpsa - exc_tpsa)

    # MW ratio: size compatibility (drug MW / excipient MW)
    mw_ratio = drug_mw / exc_mw if exc_mw > 0 else 0.0

    # H-bond complementarity: drug donors matching excipient acceptors
    hbond_complement = drug_hbd * exc_hba + drug_hba * exc_hbd

    # Sum and product of LogP (captures both polarity and interaction potential)
    logp_sum = drug_logp + exc_logp
    logp_product = drug_logp * exc_logp

    mol_feats = np.array([
        drug_logp, drug_mw, drug_tpsa, drug_hbd, drug_hba,
        exc_logp, exc_mw, exc_tpsa, exc_hbd, exc_hba,
        logp_delta, tpsa_delta, mw_ratio, hbond_complement,
        logp_sum, logp_product,
    ], dtype=np.float32)

    return np.concatenate([fp, mol_feats])


FINGERPRINT_NAMES = [
    f"drug_fp{i}" for i in range(FP_BITS)
] + [
    f"exc_fp{i}" for i in range(FP_BITS)
]

MOLECULAR_NAMES = FINGERPRINT_NAMES + [
    "drug_logp", "drug_mw", "drug_tpsa", "drug_hbd", "drug_hba",
    "exc_logp", "exc_mw", "exc_tpsa", "exc_hbd", "exc_hba",
    "logp_delta", "tpsa_delta", "mw_ratio", "hbond_complement",
    "logp_sum", "logp_product",
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Train XGBClassifier and return metrics. Handles class imbalance."""
    # Compute scale_pos_weight for imbalanced classes
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "auroc": roc_auc_score(y_test, probs),
        "f1": f1_score(y_test, preds),
        "f1_incompat": f1_score(y_test, preds, pos_label=0),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    resolve_cids = "--resolve" in sys.argv

    print("Loading DE-INTERACT drug-excipient compatibility dataset...", flush=True)
    ds = DrugExcipientCompatibility()
    print(ds)
    print()

    # Dataset statistics
    print("Dataset statistics:")
    print(f"  Total pairs:       {len(ds)}")
    print(f"  Compatible:        {ds.n_compatible} ({100*ds.n_compatible/len(ds):.1f}%)")
    print(f"  Incompatible:      {ds.n_incompatible} ({100*ds.n_incompatible/len(ds):.1f}%)")
    print(f"  Unique drugs:      {ds.unique_drugs}")
    print(f"  Unique excipients: {ds.unique_excipients}")
    print(f"  Features:          {FP_BITS * 2} fingerprint bits")
    print()

    tiers: dict[str, tuple] = {
        "Tier 1: Fingerprints Only (1762 bits)": (
            fingerprint_features, FINGERPRINT_NAMES
        ),
    }

    if resolve_cids:
        # Resolve molecular properties for all unique CIDs
        all_cids = set()
        for r in ds.records:
            all_cids.add(r.drug_cid)
            all_cids.add(r.excipient_cid)

        print(
            f"Resolving molecular properties for {len(all_cids)} unique CIDs...",
            end=" ", flush=True,
        )
        t0 = time.time()
        resolved = 0
        for cid in all_cids:
            props = _get_cid_props(cid)
            if props:
                resolved += 1
        print(f"{resolved}/{len(all_cids)} resolved ({time.time()-t0:.1f}s)")
        print()

        tiers["Tier 2: + Molecular Descriptors"] = (
            molecular_features, MOLECULAR_NAMES
        )
    else:
        print(
            "  (Pass --resolve to add Tier 2 molecular descriptor features. "
            "Requires PubChem API or RDKit.)"
        )
        print()

    # ===== Protocol 1: Random Stratified Split =====
    print("=" * 70)
    print("  PROTOCOL 1: Random Stratified Split")
    print("=" * 70)
    print()

    train, val, test = ds.split_random(test_size=0.15, val_size=0.15, seed=42)
    print(f"  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    print(
        f"  Test class balance: "
        f"{sum(1 for r in test if r.compatible)} compatible, "
        f"{sum(1 for r in test if not r.compatible)} incompatible"
    )
    print()

    random_results: dict[str, dict] = {}
    for name, (feat_fn, feat_names) in tiers.items():
        print(f"  Building features for {name}...", flush=True)
        X_train = np.array([feat_fn(r) for r in train], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test], dtype=np.float32)
        y_train = np.array(
            [1 if r.compatible else 0 for r in train], dtype=np.int32
        )
        y_test = np.array(
            [1 if r.compatible else 0 for r in test], dtype=np.int32
        )

        result = evaluate(X_train, y_train, X_test, y_test)
        random_results[name] = result

        print(f"    Features: {X_train.shape[1]}")
        print(f"    Accuracy:        {result['accuracy']:.4f}")
        print(f"    AUROC:           {result['auroc']:.4f}")
        print(f"    F1 (compatible): {result['f1']:.4f}")
        print(f"    F1 (incompat.):  {result['f1_incompat']:.4f}")

        # Top features by importance (only show a few for fingerprints)
        importances = result["model"].feature_importances_
        top10 = np.argsort(importances)[::-1][:10]
        top_feats = [
            (feat_names[i], importances[i])
            for i in top10
            if i < len(feat_names)
        ]
        print(
            f"    Top 5:  "
            f"{', '.join(f'{n} ({v:.3f})' for n, v in top_feats[:5])}"
        )
        print()

    # ===== Protocol 2: Leave-Drugs-Out =====
    print("=" * 70)
    print("  PROTOCOL 2: Leave-Drugs-Out (generalization to unseen drugs)")
    print("=" * 70)
    print()

    n_hold = max(10, ds.unique_drugs // 10)  # Hold out ~10% of drugs
    train_ldo, test_ldo = ds.split_leave_drugs_out(n_held_out=n_hold, seed=42)
    test_drugs = len(set(r.drug_cid for r in test_ldo))
    print(f"  Held out: {test_drugs} drugs ({n_hold} requested)")
    print(f"  Train: {len(train_ldo)}  |  Test: {len(test_ldo)}")
    n_test_compat = sum(1 for r in test_ldo if r.compatible)
    n_test_incompat = sum(1 for r in test_ldo if not r.compatible)
    print(
        f"  Test class balance: {n_test_compat} compatible, "
        f"{n_test_incompat} incompatible"
    )
    print()

    ldo_results: dict[str, dict] = {}
    for name, (feat_fn, feat_names) in tiers.items():
        print(f"  Building features for {name}...", flush=True)
        X_train = np.array([feat_fn(r) for r in train_ldo], dtype=np.float32)
        X_test = np.array([feat_fn(r) for r in test_ldo], dtype=np.float32)
        y_train = np.array(
            [1 if r.compatible else 0 for r in train_ldo], dtype=np.int32
        )
        y_test = np.array(
            [1 if r.compatible else 0 for r in test_ldo], dtype=np.int32
        )

        # Guard against degenerate test sets (all one class)
        if len(set(y_test)) < 2:
            print("    Skipping — test set has only one class")
            print()
            continue

        result = evaluate(X_train, y_train, X_test, y_test)
        ldo_results[name] = result

        print(f"    Accuracy:        {result['accuracy']:.4f}")
        print(f"    AUROC:           {result['auroc']:.4f}")
        print(f"    F1 (compatible): {result['f1']:.4f}")
        print(f"    F1 (incompat.):  {result['f1_incompat']:.4f}")
        print()

    # ===== Summary =====
    print("=" * 70)
    print(f"  {'Model':<42} {'Random':>14} {'LDO':>14}")
    print(f"  {'':42} {'Acc / AUROC':>14} {'Acc / AUROC':>14}")
    print("  " + "-" * 70)
    for name in tiers:
        r_res = random_results.get(name)
        l_res = ldo_results.get(name)
        r_str = (
            f"{r_res['accuracy']:.3f}/{r_res['auroc']:.3f}"
            if r_res else "  N/A"
        )
        l_str = (
            f"{l_res['accuracy']:.3f}/{l_res['auroc']:.3f}"
            if l_res else "  N/A"
        )
        print(f"  {name:<42} {r_str:>14}  {l_str:>14}")
    print("=" * 70)
    print()
    print("Notes:")
    print("  - AUROC is the primary metric (insensitive to class imbalance)")
    print("  - F1 (incompat.) measures detection of problematic pairs")
    print("  - LDO = Leave-Drugs-Out (generalization to unseen drugs)")
    if not resolve_cids:
        print("  - Run with --resolve for Tier 2 molecular descriptor features")


if __name__ == "__main__":
    run()
