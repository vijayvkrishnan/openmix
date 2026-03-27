"""
Local cache management for resolved ingredients.

Two caches:
  - Seed cache: ships with the package (~2K common ingredients)
  - User cache: grows from PubChem lookups (~/.openmix/ingredient_cache.json)
"""

from __future__ import annotations

import json
from pathlib import Path

SEED_DATA_PATH = Path(__file__).parent / "seed_ingredients.json"
USER_CACHE_DIR = Path.home() / ".openmix"
USER_CACHE_PATH = USER_CACHE_DIR / "ingredient_cache.json"


def load_seed_cache() -> dict[str, dict]:
    """Load the bundled seed ingredient data."""
    if not SEED_DATA_PATH.exists():
        return {}
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Key by uppercased INCI name
    return {k.upper().strip(): v for k, v in data.items()}


def load_user_cache() -> dict[str, dict]:
    """Load the user's local ingredient cache."""
    if not USER_CACHE_PATH.exists():
        return {}
    try:
        with open(USER_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_to_user_cache(key: str, data: dict):
    """Save a resolved ingredient to the user's local cache."""
    cache = load_user_cache()
    cache[key] = data
    USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
