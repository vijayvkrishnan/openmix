"""Tests for the ingredient resolver."""

import json
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from openmix.resolver.resolve import (
    resolve,
    resolve_many,
    ResolvedIngredient,
    _infer_charge,
    _session_cache,
)
from openmix.resolver.cache import load_seed_cache, SEED_DATA_PATH

# Use sys.modules for unambiguous module reference (avoids name collision
# between the resolve module and resolve function on Python 3.10)
_resolve_mod = sys.modules["openmix.resolver.resolve"]


# ---------------------------------------------------------------------------
# ResolvedIngredient dataclass
# ---------------------------------------------------------------------------

def test_resolved_ingredient_defaults():
    r = ResolvedIngredient(inci_name="Test")
    assert r.resolved is False
    assert r.smiles is None
    assert r.log_p is None
    assert r.charge_type is None


def test_is_hydrophobic():
    r = ResolvedIngredient(inci_name="Oil", log_p=5.0, resolved=True)
    assert r.is_hydrophobic is True
    assert r.is_hydrophilic is False


def test_is_hydrophilic():
    r = ResolvedIngredient(inci_name="Sugar", log_p=-2.0, resolved=True)
    assert r.is_hydrophilic is True
    assert r.is_hydrophobic is False


def test_neither_hydro():
    r = ResolvedIngredient(inci_name="Mid", log_p=1.5, resolved=True)
    assert r.is_hydrophobic is False
    assert r.is_hydrophilic is False


def test_to_dict():
    r = ResolvedIngredient(inci_name="Glycerin", smiles="OCC(O)CO",
                            molecular_weight=92.09, resolved=True)
    d = r.to_dict()
    assert d["inci_name"] == "Glycerin"
    assert d["smiles"] == "OCC(O)CO"
    assert d["resolved"] is True


# ---------------------------------------------------------------------------
# Charge inference
# ---------------------------------------------------------------------------

def test_infer_anionic():
    assert _infer_charge("CCCCOS(=O)(=O)[O-]") == "anionic"


def test_infer_cationic():
    assert _infer_charge("CCCC[N+](C)(C)C") == "cationic"


def test_infer_amphoteric():
    assert _infer_charge("[O-]C(=O)CC[N+](C)C") == "amphoteric"


def test_infer_nonionic():
    assert _infer_charge("OCCO") == "nonionic"


# ---------------------------------------------------------------------------
# Seed cache
# ---------------------------------------------------------------------------

def test_seed_cache_loads():
    """Seed cache should load without errors and have entries."""
    cache = load_seed_cache()
    assert isinstance(cache, dict)
    # Should ship with at least some ingredients
    if SEED_DATA_PATH.exists():
        assert len(cache) > 0


def test_seed_cache_keys_are_uppercased():
    """All keys in seed cache should be uppercased."""
    cache = load_seed_cache()
    for key in cache:
        assert key == key.upper().strip(), f"Key not uppercased: {key}"


# ---------------------------------------------------------------------------
# Resolution chain (mocked)
# ---------------------------------------------------------------------------

def _clear_session_cache():
    """Clear the module-level session cache between tests."""
    _session_cache.clear()


def test_resolve_from_seed_cache():
    """Seed cache hit should resolve immediately."""
    _clear_session_cache()
    fake_seed = {"GLYCERIN": {"smiles": "OCC(O)CO", "mw": 92.09, "log_p": -1.76}}
    with patch.object(_resolve_mod, "_get_seed_cache", return_value=fake_seed), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}), \
         patch.object(_resolve_mod, "lookup_pubchem", return_value=None):
        result = resolve("Glycerin")
    assert result.resolved is True
    assert result.source == "seed"
    assert result.molecular_weight == 92.09


def test_resolve_from_user_cache():
    """User cache should be checked after seed cache miss."""
    _clear_session_cache()
    fake_user = {"NIACINAMIDE": {"smiles": "c1ccc(c(c1)C(=O)N)N", "mw": 122.12}}
    with patch.object(_resolve_mod, "_get_seed_cache", return_value={}), \
         patch.object(_resolve_mod, "_get_user_cache", return_value=fake_user), \
         patch.object(_resolve_mod, "lookup_pubchem", return_value=None):
        result = resolve("Niacinamide")
    assert result.resolved is True
    assert result.source == "cache"


def test_resolve_from_pubchem():
    """PubChem fallback should be used when caches miss."""
    _clear_session_cache()
    pubchem_data = {"smiles": "O=C(O)C1CC1", "mw": 86.09, "log_p": 0.5}
    with patch.object(_resolve_mod, "_get_seed_cache", return_value={}), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}), \
         patch.object(_resolve_mod, "lookup_pubchem", return_value=pubchem_data), \
         patch.object(_resolve_mod, "save_to_user_cache") as mock_save:
        result = resolve("Cyclopropanecarboxylic Acid")
    assert result.resolved is True
    assert result.source == "pubchem"
    # Should save to user cache
    mock_save.assert_called_once()


def test_resolve_unresolved():
    """Ingredient not found anywhere should return unresolved."""
    _clear_session_cache()
    with patch.object(_resolve_mod, "_get_seed_cache", return_value={}), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}), \
         patch.object(_resolve_mod, "lookup_pubchem", return_value=None):
        result = resolve("Made Up Ingredient XYZ")
    assert result.resolved is False
    assert result.smiles is None


def test_resolve_session_cache():
    """Second call for same ingredient should hit session cache."""
    _clear_session_cache()
    fake_seed = {"GLYCERIN": {"smiles": "OCC(O)CO", "mw": 92.09}}
    with patch.object(_resolve_mod, "_get_seed_cache", return_value=fake_seed), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}):
        r1 = resolve("Glycerin")
        # Second call — seed cache shouldn't be checked again
        r2 = resolve("Glycerin")
    assert r1 is r2  # Same object from session cache


def test_resolve_case_insensitive():
    """Resolution should be case-insensitive."""
    _clear_session_cache()
    fake_seed = {"GLYCERIN": {"smiles": "OCC(O)CO", "mw": 92.09}}
    with patch.object(_resolve_mod, "_get_seed_cache", return_value=fake_seed), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}):
        r1 = resolve("glycerin")
        _clear_session_cache()
        r2 = resolve("GLYCERIN")
    assert r1.resolved is True
    assert r2.resolved is True


def test_resolve_many():
    """resolve_many should return dict keyed by INCI name."""
    _clear_session_cache()
    fake_seed = {
        "GLYCERIN": {"smiles": "OCC(O)CO", "mw": 92.09},
        "WATER": {"smiles": "O", "mw": 18.02},
    }
    with patch.object(_resolve_mod, "_get_seed_cache", return_value=fake_seed), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}):
        results = resolve_many(["Glycerin", "Water"])
    assert "Glycerin" in results
    assert "Water" in results
    assert results["Glycerin"].resolved is True


# ---------------------------------------------------------------------------
# RDKit enrichment (only runs if RDKit is available)
# ---------------------------------------------------------------------------

def test_rdkit_enrichment_when_available():
    """If RDKit is available, enrichment should fill in properties."""
    _clear_session_cache()
    fake_seed = {"GLYCERIN": {"smiles": "OCC(O)CO"}}

    # We need to check if RDKit is actually available
    try:
        from openmix.molecular import is_available
        has_rdkit = is_available()
    except ImportError:
        has_rdkit = False

    with patch.object(_resolve_mod, "_get_seed_cache", return_value=fake_seed), \
         patch.object(_resolve_mod, "_get_user_cache", return_value={}):
        result = resolve("Glycerin")

    if has_rdkit:
        # RDKit should have computed these from SMILES
        assert result.molecular_weight is not None
        assert result.log_p is not None
    else:
        # Without RDKit, only what the cache provides
        assert result.resolved is True
