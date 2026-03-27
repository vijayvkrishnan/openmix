"""
Pharmaceutical Formulation Solubility Dataset — FormulaBench Task 2.

251 drug delivery formulations with 6 excipients and measured drug
solubility (mg/mL). Predicting how excipient composition affects drug
solubility is fundamental to pharmaceutical formulation design.

Source: CheMixHub (chemcognition-lab/chemixhub)
        Rajaonson et al. 2025, arXiv 2506.12231
License: MIT
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openmix.schema import Formula


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Compound metadata
COMPOUNDS = {
    0: {"name": "Polysorbate 20", "inci": "Polysorbate 20", "smiles": "CCCCCCCCCCCC(=O)OCCOCC(OCCO)C1OCC(OCCO)C1OCCO", "type": "nonionic_surfactant"},
    1: {"name": "Polysorbate 80", "inci": "Polysorbate 80", "smiles": "CCCCCCCC/C=C/CCCCCCCC(=O)OCCOCC(OCCO)C1OC(CC1OCCO)OCCO", "type": "nonionic_surfactant"},
    2: {"name": "Poloxamer 188", "inci": "Poloxamer 188", "smiles": "CC1CO1.C2CO2", "type": "nonionic_surfactant"},
    3: {"name": "DMSO", "inci": "Dimethyl Sulfoxide", "smiles": "C[S](C)=O", "type": "solvent"},
    4: {"name": "Propylene Glycol", "inci": "Propylene Glycol", "smiles": "CC(O)CO", "type": "cosolvent"},
    5: {"name": "Water", "inci": "Water", "smiles": "O", "type": "solvent"},
}

COMPOUND_NAMES = [COMPOUNDS[i]["name"] for i in range(6)]


@dataclass
class PharmaSolubilityRecord:
    """A single pharmaceutical formulation with measured solubility."""
    id: int
    mole_fractions: dict[int, float]  # compound_id -> mole fraction
    solubility: float  # mg/mL

    def to_formula(self) -> Formula:
        """Convert to an OpenMix Formula using INCI names."""
        ings = []
        for cid, frac in self.mole_fractions.items():
            if frac > 0:
                info = COMPOUNDS[cid]
                ings.append({
                    "inci_name": info["inci"],
                    "percentage": round(frac * 100, 2),
                    "function": info["type"],
                    "smiles": info["smiles"],
                })

        return Formula(
            name=f"Pharma #{self.id}",
            ingredients=ings,
            category="pharma",
        )

    @property
    def feature_vector(self) -> list[float]:
        """6 mole fractions as a flat vector."""
        return [self.mole_fractions.get(i, 0.0) for i in range(6)]


class PharmaSolubility:
    """
    Pharmaceutical formulation solubility benchmark dataset.

    251 formulations, 6 excipients, continuous solubility target (mg/mL).
    Task: predict drug solubility from excipient composition.

    Source: CheMixHub (Rajaonson et al. 2025)

    Usage:
        from openmix.benchmarks.pharma_solubility import PharmaSolubility

        ds = PharmaSolubility()
        print(f"{len(ds)} formulations")

        train, val, test = ds.split_random()
        X_train, y_train = ds.to_arrays(train)
    """

    def __init__(self, data_path: str | Path | None = None):
        self._data_path = Path(data_path) if data_path else DATA_DIR / "raw" / "chemixhub_medicine_formulations.csv"
        self._records: list[PharmaSolubilityRecord] = []
        self._load()

    def _load(self):
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._data_path}. "
                f"Download from: https://github.com/chemcognition-lab/chemixhub"
            )

        with open(self._data_path, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                cmp_ids = ast.literal_eval(row["cmp_ids"])
                fractions = ast.literal_eval(row["cmp_mole_fractions"])

                mole_fractions = {}
                for cid, frac in zip(cmp_ids, fractions):
                    mole_fractions[cid] = float(frac)

                self._records.append(PharmaSolubilityRecord(
                    id=i + 1,
                    mole_fractions=mole_fractions,
                    solubility=float(row["value"]),
                ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> PharmaSolubilityRecord:
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> list[PharmaSolubilityRecord]:
        return self._records

    @property
    def solubilities(self) -> list[float]:
        return [r.solubility for r in self._records]

    def split_random(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[PharmaSolubilityRecord], list[PharmaSolubilityRecord], list[PharmaSolubilityRecord]]:
        """Random split. Returns (train, val, test)."""
        rng = np.random.RandomState(seed)
        indices = list(range(len(self._records)))
        rng.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return (
            [self._records[i] for i in sorted(train_idx)],
            [self._records[i] for i in sorted(val_idx)],
            [self._records[i] for i in sorted(test_idx)],
        )

    def to_arrays(
        self,
        records: list[PharmaSolubilityRecord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy. X = (n, 6) mole fractions, y = (n,) solubility."""
        recs = records or self._records
        X = np.array([r.feature_vector for r in recs], dtype=np.float32)
        y = np.array([r.solubility for r in recs], dtype=np.float32)
        return X, y

    def __str__(self) -> str:
        sols = self.solubilities
        return (
            f"PharmaSolubility: {len(self)} formulations, "
            f"solubility range {min(sols):.2f}-{max(sols):.2f} mg/mL"
        )
