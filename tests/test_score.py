"""Tests for the stability scoring function."""

from openmix import Formula, score


def test_clean_formula_scores_high():
    """A well-formulated moisturizer should score well."""
    formula = Formula(
        name="Simple Moisturizer",
        ingredients=[
            ("Water", 75.0),
            ("Glycerin", 5.0),
            ("Cetearyl Alcohol", 4.0),
            ("Caprylic/Capric Triglyceride", 10.0),
            ("Phenoxyethanol", 1.0),
            ("Polysorbate 60", 3.0),
            ("Xanthan Gum", 0.5),
            ("Tocopherol", 0.5),
            ("Citric Acid", 0.2),
            ("Sodium Hydroxide", 0.1),
            ("Disodium EDTA", 0.1),
        ],
        target_ph=5.5,
        category="skincare",
    )
    result = score(formula)
    # Should be well above 70
    assert result.total >= 70, f"Clean formula scored only {result.total}"
    assert result.compatibility >= 30, "No incompatibilities should exist"
    assert result.formula_integrity >= 8, "Percentages should be good"


def test_incompatible_formula_scores_low():
    """BPO + Retinol should tank the compatibility score."""
    formula = Formula(
        ingredients=[
            ("Benzoyl Peroxide", 2.5),
            ("Retinol", 1.0),
            ("Water", 96.5),
        ],
    )
    result = score(formula)
    assert result.compatibility == 0, "Hard rule violation should zero compatibility"
    assert result.total < 50, f"Incompatible formula scored {result.total}"


def test_score_is_deterministic():
    """Same formula must always get the same score."""
    formula = Formula(
        ingredients=[
            ("Water", 80.0),
            ("Glycerin", 10.0),
            ("Phenoxyethanol", 1.0),
            ("Niacinamide", 5.0),
            ("Xanthan Gum", 0.5),
        ],
        target_ph=5.5,
    )
    s1 = score(formula)
    s2 = score(formula)
    assert s1.total == s2.total, "Score must be deterministic"


def test_score_decomposition():
    """Sub-scores should sum to total."""
    formula = Formula(
        ingredients=[
            ("Water", 85.0),
            ("Glycerin", 10.0),
            ("Phenoxyethanol", 1.0),
            ("Niacinamide", 4.0),
        ],
        target_ph=5.5,
    )
    result = score(formula)
    expected_total = (
        result.compatibility
        + result.ph_suitability
        + result.emulsion_balance
        + result.formula_integrity
        + result.system_completeness
    )
    assert abs(result.total - expected_total) < 0.2, "Sub-scores should sum to total"


def test_ph_conflict_reduces_score():
    """Ascorbic Acid at high pH should reduce pH suitability."""
    good_ph = Formula(
        ingredients=[("Ascorbic Acid", 15.0), ("Water", 85.0)],
        target_ph=3.0,
    )
    bad_ph = Formula(
        ingredients=[("Ascorbic Acid", 15.0), ("Water", 85.0)],
        target_ph=7.0,
    )
    good_result = score(good_ph)
    bad_result = score(bad_ph)
    assert good_result.ph_suitability > bad_result.ph_suitability


def test_preservative_bonus():
    """Formula with preservative should score higher on completeness."""
    with_preserv = Formula(
        ingredients=[("Water", 90.0), ("Glycerin", 9.0), ("Phenoxyethanol", 1.0)],
    )
    without_preserv = Formula(
        ingredients=[("Water", 90.0), ("Glycerin", 10.0)],
    )
    assert score(with_preserv).system_completeness > score(without_preserv).system_completeness


def test_percentage_integrity():
    """Bad percentages should reduce integrity score."""
    good = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    bad = Formula(ingredients=[("Water", 80.0), ("Glycerin", 5.0)])
    assert score(good).formula_integrity > score(bad).formula_integrity


def test_score_string_output():
    """Score should have a readable string representation."""
    formula = Formula(
        ingredients=[("Water", 90.0), ("Glycerin", 10.0)],
        target_ph=5.5,
    )
    result = score(formula)
    output = str(result)
    assert "Stability Score" in output
    assert "/100" in output
