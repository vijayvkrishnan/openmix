"""
Three-tier ingredient resolution: local cache -> extended cache -> PubChem API.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from openmix.resolver.cache import load_seed_cache, load_user_cache, save_to_user_cache
from openmix.resolver.pubchem import lookup_pubchem

try:
    from openmix.molecular import compute_properties, compute_hlb_griffin, is_available as rdkit_available
    HAS_RDKIT = rdkit_available()
except ImportError:
    HAS_RDKIT = False


@dataclass
class ResolvedIngredient:
    """Molecular identity and properties for an ingredient."""
    inci_name: str
    smiles: Optional[str] = None
    molecular_weight: Optional[float] = None
    log_p: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    tpsa: Optional[float] = None
    hlb: Optional[float] = None
    charge_type: Optional[str] = None  # anionic, cationic, nonionic, amphoteric
    resolved: bool = False
    source: Optional[str] = None  # "seed", "cache", "pubchem", "rdkit"

    @property
    def is_hydrophobic(self) -> bool:
        return self.log_p is not None and self.log_p > 3.0

    @property
    def is_hydrophilic(self) -> bool:
        return self.log_p is not None and self.log_p < 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Module-level caches
_seed_cache: dict[str, dict] | None = None
_user_cache: dict[str, dict] | None = None
_session_cache: dict[str, ResolvedIngredient] = {}


def _get_seed_cache() -> dict[str, dict]:
    global _seed_cache
    if _seed_cache is None:
        _seed_cache = load_seed_cache()
    return _seed_cache


def _get_user_cache() -> dict[str, dict]:
    global _user_cache
    if _user_cache is None:
        _user_cache = load_user_cache()
    return _user_cache


def _enrich_with_rdkit(result: ResolvedIngredient) -> ResolvedIngredient:
    """Compute additional properties from SMILES using RDKit."""
    if not HAS_RDKIT or not result.smiles:
        return result

    props = compute_properties(result.smiles)
    if props:
        result.molecular_weight = result.molecular_weight or props.get("molecular_weight")
        result.log_p = result.log_p or props.get("log_p")
        result.hbd = result.hbd or props.get("hbd")
        result.hba = result.hba or props.get("hba")
        result.tpsa = result.tpsa or props.get("tpsa")

    if result.hlb is None:
        result.hlb = compute_hlb_griffin(result.smiles)

    # Infer charge type from SMILES
    if result.charge_type is None and result.smiles:
        result.charge_type = _infer_charge(result.smiles)

    return result


def _infer_charge(smiles: str) -> str:
    """Infer charge type from SMILES string."""
    has_neg = "[O-]" in smiles or "S(=O)(=O)[O-]" in smiles or "C(=O)[O-]" in smiles
    has_pos = "[N+]" in smiles or "[NH3+]" in smiles
    if has_neg and has_pos:
        return "amphoteric"
    elif has_neg:
        return "anionic"
    elif has_pos:
        return "cationic"
    return "nonionic"


def _from_cache_entry(inci: str, entry: dict, source: str) -> ResolvedIngredient:
    """Build a ResolvedIngredient from a cache dict."""
    result = ResolvedIngredient(
        inci_name=inci,
        smiles=entry.get("smiles"),
        molecular_weight=entry.get("mw") or entry.get("molecular_weight"),
        log_p=entry.get("log_p") or entry.get("logp"),
        hbd=entry.get("hbd"),
        hba=entry.get("hba"),
        tpsa=entry.get("tpsa"),
        hlb=entry.get("hlb"),
        charge_type=entry.get("charge_type"),
        resolved=True,
        source=source,
    )
    return _enrich_with_rdkit(result)


def resolve(inci_name: str) -> ResolvedIngredient:
    """
    Resolve an INCI name to molecular properties.

    Lookup order: session cache -> seed cache -> user cache -> PubChem API.
    Results are cached for subsequent calls.
    """
    key = inci_name.upper().strip()

    # Session cache (fastest)
    if key in _session_cache:
        return _session_cache[key]

    # Seed cache (ships with package, ~2K ingredients)
    seed = _get_seed_cache()
    if key in seed:
        result = _from_cache_entry(inci_name, seed[key], "seed")
        _session_cache[key] = result
        return result

    # User cache (grows over time from PubChem lookups)
    user = _get_user_cache()
    if key in user:
        result = _from_cache_entry(inci_name, user[key], "cache")
        _session_cache[key] = result
        return result

    # PubChem API (runtime fallback)
    pubchem_result = lookup_pubchem(inci_name)
    if pubchem_result:
        result = _from_cache_entry(inci_name, pubchem_result, "pubchem")
        # Save to user cache for next time
        save_to_user_cache(key, pubchem_result)
        _session_cache[key] = result
        return result

    # Unresolved
    result = ResolvedIngredient(inci_name=inci_name, resolved=False)
    _session_cache[key] = result
    return result


def resolve_many(inci_names: list[str]) -> dict[str, ResolvedIngredient]:
    """Resolve multiple ingredients. Returns dict keyed by INCI name."""
    return {name: resolve(name) for name in inci_names}
