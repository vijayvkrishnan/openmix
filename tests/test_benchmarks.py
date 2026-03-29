"""Tests for FormulaBench datasets."""

import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.mark.skipif(
    not (DATA_DIR / "LiquidFormulationsDataset_2023.json").exists(),
    reason="Shampoo dataset not downloaded"
)
class TestShampooStability:

    def test_loads(self):
        from openmix.benchmarks import ShampooStability
        ds = ShampooStability()
        assert len(ds) == 812
        assert ds.n_stable == 294
        assert ds.n_unstable == 518

    def test_split(self):
        from openmix.benchmarks import ShampooStability
        ds = ShampooStability()
        train, val, test = ds.split_random()
        assert len(train) + len(val) + len(test) == len(ds)

    def test_to_formula(self):
        from openmix.benchmarks import ShampooStability
        ds = ShampooStability()
        formula = ds[0].to_formula()
        assert formula.total_percentage > 99
        assert len(formula.ingredients) >= 2

    def test_to_arrays(self):
        from openmix.benchmarks import ShampooStability
        ds = ShampooStability()
        X, y = ds.to_arrays()
        assert X.shape == (812, 18)
        assert y.shape == (812,)


@pytest.mark.skipif(
    not (DATA_DIR / "chemixhub_medicine_formulations.csv").exists(),
    reason="Pharma dataset not downloaded"
)
class TestPharmaSolubility:

    def test_loads(self):
        from openmix.benchmarks import PharmaSolubility
        ds = PharmaSolubility()
        assert len(ds) == 251

    def test_solubility_range(self):
        from openmix.benchmarks import PharmaSolubility
        ds = PharmaSolubility()
        sols = ds.solubilities
        assert min(sols) > 0
        assert max(sols) < 20

    def test_split(self):
        from openmix.benchmarks import PharmaSolubility
        ds = PharmaSolubility()
        train, val, test = ds.split_random()
        assert len(train) + len(val) + len(test) == len(ds)

    def test_to_formula(self):
        from openmix.benchmarks import PharmaSolubility
        ds = PharmaSolubility()
        formula = ds[0].to_formula()
        assert len(formula.ingredients) >= 2

    def test_to_arrays(self):
        from openmix.benchmarks import PharmaSolubility
        ds = PharmaSolubility()
        X, y = ds.to_arrays()
        assert X.shape == (251, 6)
        assert y.shape == (251,)


@pytest.mark.skipif(
    not (DATA_DIR / "DEL.csv").exists(),
    reason="Drug-excipient dataset not downloaded"
)
class TestDrugExcipientCompatibility:

    def test_loads(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        assert len(ds) > 4000
        assert ds.n_compatible > ds.n_incompatible

    def test_class_balance(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        assert ds.n_compatible + ds.n_incompatible == len(ds)
        # Expect ~83% compatible
        ratio = ds.n_compatible / len(ds)
        assert 0.7 < ratio < 0.95

    def test_unique_compounds(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        assert ds.unique_drugs > 100
        assert ds.unique_excipients > 100

    def test_split_random(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        train, val, test = ds.split_random()
        assert len(train) + len(val) + len(test) == len(ds)
        # Check stratification preserved
        assert any(r.compatible for r in test)
        assert any(not r.compatible for r in test)

    def test_split_leave_drugs_out(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        train, test = ds.split_leave_drugs_out()
        assert len(train) + len(test) == len(ds)
        # Verify no drug overlap
        train_drugs = set(r.drug_cid for r in train)
        test_drugs = set(r.drug_cid for r in test)
        assert train_drugs.isdisjoint(test_drugs)

    def test_to_arrays(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        X, y = ds.to_arrays()
        assert X.shape == (len(ds), 1762)
        assert y.shape == (len(ds),)
        # Fingerprint bits should be 0 or 1
        assert set(np.unique(X).astype(int)).issubset({0, 1})

    def test_to_fingerprint_arrays(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        X_drug, X_exc, y = ds.to_fingerprint_arrays()
        assert X_drug.shape == (len(ds), 881)
        assert X_exc.shape == (len(ds), 881)
        assert y.shape == (len(ds),)

    def test_record_fingerprint_length(self):
        from openmix.benchmarks import DrugExcipientCompatibility
        ds = DrugExcipientCompatibility()
        record = ds[0]
        assert len(record.drug_fp) == 881
        assert len(record.excipient_fp) == 881
        assert len(record.feature_vector) == 1762
