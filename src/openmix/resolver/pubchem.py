"""
PubChem API lookups — runtime fallback for unknown ingredients.

Rate-limited to 5 requests/second per PubChem guidelines.
Results are cached locally after first lookup.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
import urllib.error

_last_request_time = 0.0
_MIN_INTERVAL = 0.25  # 4 req/sec max


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get_json(url: str, timeout: int = 10) -> dict | None:
    """Fetch JSON from URL with rate limiting."""
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def lookup_pubchem(inci_name: str) -> dict | None:
    """
    Look up an ingredient by name on PubChem.

    Returns dict with: smiles, mw, log_p, hbd, hba, tpsa
    or None if not found.
    """
    encoded = urllib.parse.quote(inci_name)

    # Step 1: Search by name to get CID
    search_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/cids/JSON"
    )
    search_data = _get_json(search_url)
    if not search_data:
        return None

    cids = search_data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None

    cid = cids[0]

    # Step 2: Get properties
    props_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
        f"property/CanonicalSMILES,MolecularWeight,XLogP,HBondDonorCount,"
        f"HBondAcceptorCount,TPSA/JSON"
    )
    props_data = _get_json(props_url)
    if not props_data:
        return None

    props_list = props_data.get("PropertyTable", {}).get("Properties", [])
    if not props_list:
        return None

    p = props_list[0]
    return {
        "smiles": p.get("CanonicalSMILES"),
        "mw": p.get("MolecularWeight"),
        "log_p": p.get("XLogP"),
        "hbd": p.get("HBondDonorCount"),
        "hba": p.get("HBondAcceptorCount"),
        "tpsa": p.get("TPSA"),
        "pubchem_cid": cid,
    }
