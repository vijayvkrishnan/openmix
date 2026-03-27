"""
Shampoo Stability Dataset — the first FormulaBench task.

812 shampoo formulations with binary stability labels. 18 ingredients
(12 surfactants + 4 conditioning polymers + 2 thickeners) in water base.

Source: Velho et al. 2024, Scientific Data (Nature)
        DOI: 10.1038/s41597-024-03573-w
        Data: Figshare DOI 10.6084/m9.figshare.c.7132624.v1
License: CC-BY-4.0

Prior work:
- Bigan & Dufour 2024 (MDPI Cosmetics) used LLaMA models on this dataset
- AUROC ~0.7 with 20 training samples, improving with model size
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from openmix.schema import Formula


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Trade name -> INCI name mapping (from BASF Surfactants Information CSV)
TRADE_TO_INCI: dict[str, str] = {
    "Texapon SB 3 KC": "Disodium Laureth Sulfosuccinate",
    "Plantapon ACG 50": "Disodium Cocoyl Glutamate",
    "Plantapon LC 7": "Laureth-7 Citrate",
    "Plantacare 818": "Coco-Glucoside",
    "Plantacare 2000": "Decyl Glucoside",
    "Dehyton MC": "Sodium Cocoamphoacetate",
    "Dehyton PK 45": "Cocamidopropyl Betaine",
    "Dehyton ML": "Sodium Lauroamphoacetate",
    "Dehyton AB 30": "Coco-Betaine",
    "Plantapon Amino SCG-L": "Sodium Cocoyl Glutamate",
    "Plantapon Amino KG-L": "Potassium Cocoyl Glycinate",
    "Dehyquart A-CA": "Cetrimonium Chloride",
    # Conditioning polymers
    "Luviquat Excellence": "Polyquaternium-68",
    "Dehyquart CC6": "Polyquaternium-6",
    "Dehyquart CC7 Benz": "Polyquaternium-7",
    "Salcare Super 7": "Polyquaternium-7",  # crosslinked variant
    # Thickeners
    "Arlypon F": "PEG-40 Hydrogenated Castor Oil",
    "Arlypon TT": "PEG-200 Hydrogenated Glyceryl Palmate",
}

# SMILES from the surfactants CSV
TRADE_TO_SMILES: dict[str, str] = {
    "Texapon SB 3 KC": "CCCCCCCCCCCCOCCOCCOCCOC(=O)C(CC(=O)[O-])S(=O)(=O)[O-].[Na+].[Na+]",
    "Plantapon ACG 50": "CCCCCCCCCCCC(=O)NC(CCC(=O)[O-])C(=O)[O-].[Na+].[Na+]",
    "Plantapon LC 7": "OC(CC(O)(CC(O)=O)C(OCCOCCOCCOCCOCCOCCOCCOCCCCCCCCCCCC)=O)=O",
    "Plantacare 818": "CCCCCCCCCCCCCCOC1OC(CO)C(O)C(O)C1O",
    "Plantacare 2000": "CCCCCCCCCCOC1OC(CO)C(O)C(O)C1O",
    "Dehyton MC": "CCCCCCCCCCCC(=O)NCCN(CCO)CC(=O)[O-].[Na+]",
    "Dehyton PK 45": "CCCCCCCCCCCC(NCCC[N+](C)(C)CC([O-])=O)=O",
    "Dehyton ML": "CCCCCCCCCCCC(NCCN(CC([O-])=O)CCO)=O.[Na+]",
    "Dehyton AB 30": "CCCCCCCCCCCC[N+](C)(C)CC([O-])=O.[Na+]",
    "Plantapon Amino SCG-L": "[O-]C(CCC(NC(CCCCCCCCCCCCCCC)=O)C([O-])=O)=O.[Na+].[Na+]",
    "Plantapon Amino KG-L": "CCCCCCCCCCCC(NCC([O-])=O)=O.[K+]",
    "Dehyquart A-CA": "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Cl-]",
}

# Surfactant type classification
TRADE_TO_TYPE: dict[str, str] = {
    "Texapon SB 3 KC": "anionic",
    "Plantapon ACG 50": "anionic",
    "Plantapon LC 7": "anionic",
    "Plantacare 818": "nonionic",
    "Plantacare 2000": "nonionic",
    "Dehyton MC": "amphoteric",
    "Dehyton PK 45": "amphoteric",
    "Dehyton ML": "amphoteric",
    "Dehyton AB 30": "amphoteric",
    "Plantapon Amino SCG-L": "anionic",
    "Plantapon Amino KG-L": "anionic",
    "Dehyquart A-CA": "cationic",
    "Luviquat Excellence": "cationic_polymer",
    "Dehyquart CC6": "cationic_polymer",
    "Dehyquart CC7 Benz": "cationic_polymer",
    "Salcare Super 7": "cationic_polymer",
    "Arlypon F": "nonionic_thickener",
    "Arlypon TT": "nonionic_thickener",
}

INGREDIENT_COLS = list(TRADE_TO_INCI.keys())
META_COLS = {"Stability_Test", "Turbidity_NTU", "Turbidity_Error",
             "Viscosity", "Rheology_Type", "Rheology_Data", "ID"}


@dataclass
class ShampooRecord:
    """A single shampoo formulation with stability label."""
    id: int
    ingredients: dict[str, float]  # trade_name -> percentage
    stable: bool
    turbidity_ntu: Optional[float] = None
    viscosity: Optional[str] = None
    rheology_type: Optional[str] = None

    def to_formula(self) -> Formula:
        """Convert to an OpenMix Formula using INCI names."""
        ings = []
        for trade_name, pct in self.ingredients.items():
            if pct > 0:
                inci = TRADE_TO_INCI.get(trade_name, trade_name)
                smiles = TRADE_TO_SMILES.get(trade_name)
                surf_type = TRADE_TO_TYPE.get(trade_name, "unknown")
                ings.append({
                    "inci_name": inci,
                    "percentage": pct,
                    "function": surf_type,
                    "smiles": smiles,
                })

        # Add water to reach 100%
        total = sum(i["percentage"] for i in ings)
        if total < 100:
            ings.append({
                "inci_name": "Water",
                "percentage": round(100 - total, 2),
                "function": "solvent",
            })

        return Formula(
            name=f"Shampoo #{self.id}",
            ingredients=ings,
            category="skincare",
            product_type="shampoo",
        )

    @property
    def feature_vector(self) -> list[float]:
        """Return the 18 ingredient percentages as a flat vector."""
        return [self.ingredients.get(col, 0.0) for col in INGREDIENT_COLS]


class ShampooStability:
    """
    Shampoo stability benchmark dataset.

    812 formulations, 294 stable (36.2%), 518 unstable (63.8%).
    18 ingredient features (percentages), binary stability label.

    Usage:
        from openmix.benchmarks import ShampooStability

        ds = ShampooStability()
        print(f"{len(ds)} formulations, {ds.n_stable} stable")

        # Get train/test split
        train, test = ds.split_random(test_size=0.15, seed=42)

        # As numpy arrays for ML
        X_train, y_train = ds.to_arrays(train)
    """

    def __init__(self, data_path: str | Path | None = None):
        self._data_path = Path(data_path) if data_path else DATA_DIR / "raw" / "LiquidFormulationsDataset_2023.json"
        self._records: list[ShampooRecord] = []
        self._load()

    def _load(self):
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._data_path}. "
                f"Download from: https://doi.org/10.6084/m9.figshare.25451878.v1"
            )

        with open(self._data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for record in raw:
            ingredients = {}
            for col in INGREDIENT_COLS:
                val = record.get(col, 0)
                if isinstance(val, (int, float)) and val > 0:
                    ingredients[col] = float(val)

            turbidity = record.get("Turbidity_NTU")
            if turbidity == "NA" or turbidity is None:
                turbidity = None
            else:
                turbidity = float(turbidity)

            viscosity = record.get("Viscosity")
            if viscosity == "NA":
                viscosity = None

            rheology_type = record.get("Rheology_Type")
            if rheology_type == "NA":
                rheology_type = None

            self._records.append(ShampooRecord(
                id=record["ID"],
                ingredients=ingredients,
                stable=bool(record["Stability_Test"]),
                turbidity_ntu=turbidity,
                viscosity=viscosity,
                rheology_type=rheology_type,
            ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> ShampooRecord:
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> list[ShampooRecord]:
        return self._records

    @property
    def n_stable(self) -> int:
        return sum(1 for r in self._records if r.stable)

    @property
    def n_unstable(self) -> int:
        return sum(1 for r in self._records if not r.stable)

    @property
    def labels(self) -> list[int]:
        """Binary labels: 1=stable, 0=unstable."""
        return [1 if r.stable else 0 for r in self._records]

    def split_random(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[ShampooRecord], list[ShampooRecord], list[ShampooRecord]]:
        """
        Split A: random stratified split.

        Returns (train, val, test) lists of ShampooRecord.
        """
        rng = np.random.RandomState(seed)

        stable_idx = [i for i, r in enumerate(self._records) if r.stable]
        unstable_idx = [i for i, r in enumerate(self._records) if not r.stable]

        rng.shuffle(stable_idx)
        rng.shuffle(unstable_idx)

        def _split_indices(indices):
            n = len(indices)
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            return indices[:-(n_test + n_val)], indices[-(n_test + n_val):-n_test], indices[-n_test:]

        s_train, s_val, s_test = _split_indices(stable_idx)
        u_train, u_val, u_test = _split_indices(unstable_idx)

        train = [self._records[i] for i in sorted(s_train + u_train)]
        val = [self._records[i] for i in sorted(s_val + u_val)]
        test = [self._records[i] for i in sorted(s_test + u_test)]

        return train, val, test

    def to_arrays(
        self,
        records: list[ShampooRecord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert records to numpy arrays for ML.

        Returns (X, y) where X is (n, 18) ingredient percentages
        and y is (n,) binary stability labels.
        """
        recs = records or self._records
        X = np.array([r.feature_vector for r in recs], dtype=np.float32)
        y = np.array([1 if r.stable else 0 for r in recs], dtype=np.int32)
        return X, y

    def __str__(self) -> str:
        return (
            f"ShampooStability: {len(self)} formulations "
            f"({self.n_stable} stable, {self.n_unstable} unstable)"
        )
