"""Tests for constraint enforcement."""

from openmix import Formula
from openmix.constraints import check_constraints


def test_all_constraints_pass():
    formula = Formula(
        ingredients=[
            ("Water", 70.0),
            ("Ascorbic Acid", 15.0),
            ("Glycerin", 15.0),
        ],
        target_ph=3.0,
    )
    result = check_constraints(
        formula,
        required_ingredients=[{"name": "Ascorbic Acid", "min_pct": 10, "max_pct": 20}],
        constraints={"total_percentage": 100, "target_ph": [2.5, 3.5]},
    )
    assert result.passed


def test_missing_required_ingredient():
    formula = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    result = check_constraints(
        formula,
        required_ingredients=[{"name": "Ascorbic Acid", "min_pct": 10, "max_pct": 20}],
    )
    assert not result.passed
    assert any("not found" in v.message for v in result.violations)


def test_required_below_minimum():
    formula = Formula(
        ingredients=[("Water", 92.0), ("Ascorbic Acid", 3.0), ("Glycerin", 5.0)],
    )
    result = check_constraints(
        formula,
        required_ingredients=[{"name": "Ascorbic Acid", "min_pct": 10, "max_pct": 20}],
    )
    assert not result.passed
    assert any("below minimum" in v.message for v in result.violations)


def test_required_above_maximum():
    formula = Formula(
        ingredients=[("Water", 55.0), ("Ascorbic Acid", 25.0), ("Glycerin", 20.0)],
    )
    result = check_constraints(
        formula,
        required_ingredients=[{"name": "Ascorbic Acid", "min_pct": 10, "max_pct": 20}],
    )
    assert not result.passed
    assert any("exceeds maximum" in v.message for v in result.violations)


def test_max_ingredients_exceeded():
    formula = Formula(
        ingredients=[("A", 20.0), ("B", 20.0), ("C", 20.0), ("D", 20.0), ("E", 20.0)],
    )
    result = check_constraints(formula, constraints={"max_ingredients": 3})
    assert not result.passed
    assert any("exceeds max" in v.message for v in result.violations)


def test_ph_out_of_range():
    formula = Formula(
        ingredients=[("Water", 100.0)],
        target_ph=7.0,
    )
    result = check_constraints(formula, constraints={"target_ph": [2.5, 3.5]})
    assert not result.passed
    assert any("pH" in v.message for v in result.violations)


def test_total_percentage_off():
    formula = Formula(ingredients=[("Water", 80.0), ("Glycerin", 5.0)])
    result = check_constraints(formula, constraints={"total_percentage": 100})
    assert not result.passed
    assert any("Total" in v.message for v in result.violations)
