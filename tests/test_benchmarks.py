"""Tests for FormulaBench datasets."""

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
