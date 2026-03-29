"""
Drug-Excipient Compatibility Dataset — FormulaBench Task 3.

4,248 drug-excipient pairs with binary compatibility labels and
PubChem fingerprints (881 bits per compound). The task: predict
whether a drug and excipient are compatible based on molecular
structure. This is critical for pharmaceutical formulation design —
incompatible drug-excipient combinations cause degradation, loss
of potency, or adverse reactions.

Source: Patel et al. 2023, International Journal of Pharmaceutics
        "DE-INTERACT: A machine-learning-based predictive tool for
        the drug-excipient interaction study during product development"
        DOI: 10.1016/j.ijpharm.2023.122839
        Data: https://github.com/nghiencuuthuoc/Drug-Excipient-Interaction-Prediction
License: Research use (published with code on GitHub)

Dataset characteristics:
- 4,248 instances (3,549 compatible + 699 incompatible)
- 470 unique drugs (PubChem CIDs), 266 unique excipients (PubChem CIDs)
- Features: 881 PubChem fingerprint bits per drug + 881 per excipient = 1,762 bits
- No SMILES or compound names in the raw data — only PubChem CIDs
  (names/SMILES can be resolved via PubChem API from the CIDs)

Limitations:
- Heavily imbalanced: 83.5% compatible vs 16.5% incompatible
- No SMILES in raw CSV — only CIDs and pre-computed PubChem fingerprints
- Drug/excipient names require PubChem CID resolution (network call)
- Fingerprints are PubChem CACTVS substructure keys (881 bits), not
  interpretable chemical descriptors
- Some duplicate drug-excipient CID pairs exist with same label
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Number of PubChem fingerprint bits per compound
FP_BITS = 881

# Total fingerprint columns: drug FP (881) + excipient FP (881)
TOTAL_FP_COLS = FP_BITS * 2

# Column layout in DEL.csv:
# Col 0: Drugs (PubChem CID)
# Col 1: Excipients (PubChem CID)
# Cols 2-882: Drug PubChem fingerprint bits (PubchemFP0..PubchemFP880)
# Cols 883-1763: Excipient PubChem fingerprint bits (PubchemFP0..PubchemFP880)
# Col 1764: Output Value ("Compatible" or "Imcompatible")

DOWNLOAD_URL = (
    "https://github.com/nghiencuuthuoc/Drug-Excipient-Interaction-Prediction"
)


@dataclass
class DrugExcipientRecord:
    """A single drug-excipient compatibility observation."""
    id: int
    drug_cid: int  # PubChem Compound ID
    excipient_cid: int  # PubChem Compound ID
    compatible: bool
    drug_fp: list[int] = field(repr=False)  # 881 PubChem fingerprint bits
    excipient_fp: list[int] = field(repr=False)  # 881 PubChem fingerprint bits
    drug_name: Optional[str] = None  # Resolved from CID (not in raw data)
    excipient_name: Optional[str] = None
    drug_smiles: Optional[str] = None  # Resolved from CID (not in raw data)
    excipient_smiles: Optional[str] = None

    @property
    def feature_vector(self) -> list[int]:
        """Return the 1762 fingerprint bits as a flat vector (drug + excipient)."""
        return self.drug_fp + self.excipient_fp


class DrugExcipientCompatibility:
    """
    Drug-excipient compatibility benchmark dataset.

    4,248 drug-excipient pairs, 3,549 compatible (83.5%), 699 incompatible (16.5%).
    470 unique drugs, 266 unique excipients.
    1,762 fingerprint features (881 per compound).

    This is a binary classification task on highly imbalanced data with
    pre-computed molecular fingerprints as features.

    Source: Patel et al. 2023, Int. J. Pharmaceutics (DE-INTERACT)

    Usage:
        from openmix.benchmarks import DrugExcipientCompatibility

        ds = DrugExcipientCompatibility()
        print(f"{len(ds)} pairs, {ds.n_compatible} compatible")

        # Random split (stratified)
        train, val, test = ds.split_random()

        # Leave-drugs-out (generalization to unseen drugs)
        train, test = ds.split_leave_drugs_out()

        # As numpy arrays for ML
        X_train, y_train = ds.to_arrays(train)
    """

    def __init__(self, data_path: str | Path | None = None):
        self._data_path = (
            Path(data_path) if data_path
            else DATA_DIR / "raw" / "DEL.csv"
        )
        self._records: list[DrugExcipientRecord] = []
        self._load()

    def _load(self):
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._data_path}. "
                f"Download DEL.csv from: {DOWNLOAD_URL}"
            )

        with open(self._data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Validate expected structure
            n_cols = len(header)
            if n_cols != 2 + TOTAL_FP_COLS + 1:
                raise ValueError(
                    f"Expected {2 + TOTAL_FP_COLS + 1} columns, got {n_cols}. "
                    f"The DEL.csv format may have changed."
                )

            for i, row in enumerate(reader):
                if len(row) != n_cols:
                    continue  # Skip malformed rows

                drug_cid = int(row[0])
                excipient_cid = int(row[1])

                # Drug fingerprint: columns 2 through 882 (881 bits)
                drug_fp = [int(row[j]) for j in range(2, 2 + FP_BITS)]

                # Excipient fingerprint: columns 883 through 1763 (881 bits)
                excipient_fp = [
                    int(row[j]) for j in range(2 + FP_BITS, 2 + TOTAL_FP_COLS)
                ]

                label = row[-1].strip()
                # Original data uses "Imcompatible" (typo) and "Compatible"
                compatible = label == "Compatible"

                self._records.append(DrugExcipientRecord(
                    id=i + 1,
                    drug_cid=drug_cid,
                    excipient_cid=excipient_cid,
                    compatible=compatible,
                    drug_fp=drug_fp,
                    excipient_fp=excipient_fp,
                ))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DrugExcipientRecord:
        return self._records[idx]

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> list[DrugExcipientRecord]:
        return self._records

    @property
    def n_compatible(self) -> int:
        return sum(1 for r in self._records if r.compatible)

    @property
    def n_incompatible(self) -> int:
        return sum(1 for r in self._records if not r.compatible)

    @property
    def labels(self) -> list[int]:
        """Binary labels: 1=compatible, 0=incompatible."""
        return [1 if r.compatible else 0 for r in self._records]

    @property
    def unique_drugs(self) -> int:
        return len(set(r.drug_cid for r in self._records))

    @property
    def unique_excipients(self) -> int:
        return len(set(r.excipient_cid for r in self._records))

    def split_random(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> tuple[
        list[DrugExcipientRecord],
        list[DrugExcipientRecord],
        list[DrugExcipientRecord],
    ]:
        """
        Stratified random split preserving class balance.

        Returns (train, val, test) lists of DrugExcipientRecord.
        """
        rng = np.random.RandomState(seed)

        compat_idx = [i for i, r in enumerate(self._records) if r.compatible]
        incompat_idx = [i for i, r in enumerate(self._records) if not r.compatible]

        rng.shuffle(compat_idx)
        rng.shuffle(incompat_idx)

        def _split_indices(
            indices: list[int],
        ) -> tuple[list[int], list[int], list[int]]:
            n = len(indices)
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            return (
                indices[: -(n_test + n_val)],
                indices[-(n_test + n_val): -n_test],
                indices[-n_test:],
            )

        c_train, c_val, c_test = _split_indices(compat_idx)
        i_train, i_val, i_test = _split_indices(incompat_idx)

        train = [self._records[i] for i in sorted(c_train + i_train)]
        val = [self._records[i] for i in sorted(c_val + i_val)]
        test = [self._records[i] for i in sorted(c_test + i_test)]

        return train, val, test

    def split_leave_drugs_out(
        self,
        n_held_out: int | None = None,
        seed: int = 42,
    ) -> tuple[list[DrugExcipientRecord], list[DrugExcipientRecord]]:
        """
        Hold out entire drugs the model has never seen.

        Tests molecular generalization: can we predict compatibility
        for a drug not in the training set?

        If n_held_out is None, holds out ~10% of unique drugs.
        """
        rng = np.random.RandomState(seed)
        all_drugs = list(set(r.drug_cid for r in self._records))
        rng.shuffle(all_drugs)

        if n_held_out is None:
            n_held_out = max(10, len(all_drugs) // 10)

        held_out = set(all_drugs[:n_held_out])

        train = [r for r in self._records if r.drug_cid not in held_out]
        test = [r for r in self._records if r.drug_cid in held_out]

        return train, test

    def to_arrays(
        self,
        records: list[DrugExcipientRecord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert records to numpy arrays for ML.

        Returns (X, y) where X is (n, 1762) fingerprint bits
        and y is (n,) binary compatibility labels.
        """
        recs = records or self._records
        X = np.array([r.feature_vector for r in recs], dtype=np.float32)
        y = np.array([1 if r.compatible else 0 for r in recs], dtype=np.int32)
        return X, y

    def to_fingerprint_arrays(
        self,
        records: list[DrugExcipientRecord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return separate drug and excipient fingerprint arrays.

        Returns (X_drug, X_excipient, y) where:
        - X_drug is (n, 881) drug fingerprint bits
        - X_excipient is (n, 881) excipient fingerprint bits
        - y is (n,) binary labels
        """
        recs = records or self._records
        X_drug = np.array([r.drug_fp for r in recs], dtype=np.float32)
        X_excipient = np.array([r.excipient_fp for r in recs], dtype=np.float32)
        y = np.array([1 if r.compatible else 0 for r in recs], dtype=np.int32)
        return X_drug, X_excipient, y

    def __str__(self) -> str:
        return (
            f"DrugExcipientCompatibility: {len(self)} pairs "
            f"({self.n_compatible} compatible, {self.n_incompatible} incompatible), "
            f"{self.unique_drugs} drugs, {self.unique_excipients} excipients"
        )
