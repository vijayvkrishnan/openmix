"""
MixtureSolDB — Binary Solvent Mixture Solubility Benchmark.

175,166 experimental solubility values for 810 organic compounds in 750
binary solvent mixtures across temperatures 252-383 K. Derived from
1,115 peer-reviewed sources. 28% of solutes are FDA-approved drugs.

The task: predict how well a compound dissolves in a specific binary
solvent mixture at a given temperature. This requires understanding
molecular interactions between three components (solute + two solvents)
— exactly the kind of mixture reasoning OpenMix is built for.

Source: Vasconcelos et al. 2026, Nature Scientific Data
        DOI: 10.1038/s41597-026-07047-z
        Data: Zenodo DOI 10.5281/zenodo.18660057
License: CC-BY 4.0
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


@dataclass
class MixtureSolRecord:
    """A single solubility measurement in a binary solvent mixture."""
    solute_smiles: str
    solvent1_smiles: str
    solvent2_smiles: str
    solvent1_name: str
    solvent2_name: str
    solvent1_fraction: float
    solvent2_fraction: float
    temperature_k: float
    log_solubility: float  # LogS (mole fraction)
    compound_name: Optional[str] = None
    cas: Optional[str] = None
    fda_approved: bool = False
    is_pure_endpoint: bool = False


class MixtureSolubility:
    """
    Binary solvent mixture solubility benchmark.

    175K records, 810 solutes, 135 solvents, 750 solvent pairs.
    Task: predict LogS (mole fraction) from molecular structure
    and mixture composition.

    Usage:
        from openmix.benchmarks.mixture_solubility import MixtureSolubility

        ds = MixtureSolubility()
        print(ds)

        train, val, test = ds.split_random()
        X, y = ds.to_molecular_arrays(train)
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        binary_only: bool = True,
        max_records: int | None = None,
    ):
        self._data_path = (
            Path(data_path) if data_path
            else DATA_DIR / "raw" / "MixtureSolDB.csv"
        )
        self._binary_only = binary_only
        self._max_records = max_records
        self._records: list[MixtureSolRecord] = []
        self._load()

    def _load(self):
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._data_path}. "
                f"Download from: https://zenodo.org/records/18660057"
            )

        with open(self._data_path, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if self._max_records and i >= self._max_records:
                    break

                is_pure = row.get("IsPureSolventEndpoint") == "True"
                if self._binary_only and is_pure:
                    continue

                log_s = row.get("LogS(mole_fraction)")
                if not log_s or log_s == "NA":
                    continue

                self._records.append(MixtureSolRecord(
                    solute_smiles=row["SMILES_Solute"],
                    solvent1_smiles=row["SMILES_Solvent1"],
                    solvent2_smiles=row["SMILES_Solvent2"],
                    solvent1_name=row["Solvent1"],
                    solvent2_name=row["Solvent2"],
                    solvent1_fraction=float(row["Fraction_Solvent1"]),
                    solvent2_fraction=float(row["Fraction_Solvent2"]),
                    temperature_k=float(row["Temperature_K"]),
                    log_solubility=float(log_s),
                    compound_name=row.get("Compound_Name"),
                    cas=row.get("CAS"),
                    fda_approved=row.get("FDA_Approved") == "True",
                    is_pure_endpoint=is_pure,
                ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> MixtureSolRecord:
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> list[MixtureSolRecord]:
        return self._records

    @property
    def unique_solutes(self) -> int:
        return len(set(r.solute_smiles for r in self._records))

    @property
    def unique_solvents(self) -> int:
        return len(
            set(r.solvent1_smiles for r in self._records)
            | set(r.solvent2_smiles for r in self._records)
        )

    def split_random(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[MixtureSolRecord], list[MixtureSolRecord], list[MixtureSolRecord]]:
        """Random split. Returns (train, val, test)."""
        rng = np.random.RandomState(seed)
        indices = list(range(len(self._records)))
        rng.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))

        return (
            [self._records[i] for i in indices[n_test + n_val:]],
            [self._records[i] for i in indices[n_test:n_test + n_val]],
            [self._records[i] for i in indices[:n_test]],
        )

    def split_leave_solutes_out(
        self,
        n_held_out: int = 80,
        seed: int = 42,
    ) -> tuple[list[MixtureSolRecord], list[MixtureSolRecord]]:
        """
        Hold out entire solutes the model has never seen.
        This tests molecular generalization: can the model predict
        solubility for a completely new compound?
        """
        rng = np.random.RandomState(seed)
        all_solutes = list(set(r.solute_smiles for r in self._records))
        rng.shuffle(all_solutes)

        held_out = set(all_solutes[:n_held_out])

        train = [r for r in self._records if r.solute_smiles not in held_out]
        test = [r for r in self._records if r.solute_smiles in held_out]
        return train, test

    def to_composition_arrays(
        self, records: list[MixtureSolRecord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tier 1: Composition-only features.
        [solvent1_fraction, solvent2_fraction, temperature]
        """
        recs = records or self._records
        X = np.array([
            [r.solvent1_fraction, r.solvent2_fraction, r.temperature_k]
            for r in recs
        ], dtype=np.float32)
        y = np.array([r.log_solubility for r in recs], dtype=np.float32)
        return X, y

    def __str__(self) -> str:
        return (
            f"MixtureSolubility: {len(self)} records, "
            f"{self.unique_solutes} solutes, "
            f"{self.unique_solvents} solvents"
        )
