"""Tests for the OpenMix validation engine."""

from openmix import Formula, validate
from openmix.knowledge.loader import load_knowledge


# ---------------------------------------------------------------------------
# Hard Rules — must fire in ALL modes
# ---------------------------------------------------------------------------

def test_bleach_ammonia_fires_in_all_modes():
    """Lethal combination must be caught in every mode, including discovery."""
    formula = Formula(
        ingredients=[
            ("Sodium Hypochlorite", 5.0),
            ("Ammonia", 3.0),
            ("Water", 92.0),
        ],
    )
    for mode in ("safety", "formulation", "discovery"):
        report = validate(formula, mode=mode)
        assert report.errors > 0, f"Bleach + Ammonia not caught in {mode} mode"
        assert any("chloramine" in i.message.lower() or "toxic" in i.message.lower()
                    for i in report.issues)


def test_bpo_retinol_hard_rule():
    """BPO + Retinol is a hard rule — complete deactivation."""
    formula = Formula(
        ingredients=[
            ("Benzoyl Peroxide", 2.5),
            ("Retinol", 1.0),
            ("Water", 96.5),
        ],
    )
    report = validate(formula)
    assert report.errors > 0
    assert not report.passed


def test_benzene_formation():
    """Ascorbic Acid + Sodium Benzoate = benzene risk. Hard rule."""
    formula = Formula(
        ingredients=[
            ("Ascorbic Acid", 0.5),
            ("Sodium Benzoate", 0.1),
            ("Water", 99.4),
        ],
        category="beverage",
    )
    report = validate(formula)
    assert report.errors > 0
    assert any("benzene" in i.message.lower() for i in report.issues)


def test_probiotic_preservative_hard():
    """Probiotics + antimicrobial preservative is a hard rule."""
    formula = Formula(
        ingredients=[
            ("Lactobacillus Acidophilus", 10.0),
            ("Potassium Sorbate", 0.1),
            ("Maltodextrin", 89.9),
        ],
        category="supplement",
    )
    report = validate(formula)
    assert report.errors > 0


# ---------------------------------------------------------------------------
# Soft Rules — mode-dependent behavior
# ---------------------------------------------------------------------------

def test_niacinamide_vitamin_c_safety_mode():
    """In safety mode, niacinamide + vitamin C should warn."""
    formula = Formula(
        ingredients=[
            ("Niacinamide", 10.0),
            ("Ascorbic Acid", 15.0),
            ("Water", 75.0),
        ],
        target_ph=3.0,
        category="skincare",
    )
    report = validate(formula, mode="safety")
    assert report.warnings > 0


def test_niacinamide_vitamin_c_discovery_mode():
    """In discovery mode, niacinamide + vitamin C should be ignored (debated interaction)."""
    formula = Formula(
        ingredients=[
            ("Niacinamide", 10.0),
            ("Ascorbic Acid", 15.0),
            ("Water", 75.0),
        ],
        target_ph=3.0,
        category="skincare",
    )
    report = validate(formula, mode="discovery")
    # Should NOT have warnings for this debated interaction
    niac_issues = [i for i in report.issues
                   if i.check == "compatibility"
                   and i.severity in ("error", "warning")
                   and i.ingredient and "NIACINAMIDE" in i.ingredient]
    assert len(niac_issues) == 0


def test_retinol_glycolic_acid_soft():
    """Retinol + Glycolic Acid is a soft rule — mode affects severity."""
    formula = Formula(
        ingredients=[
            ("Retinol", 1.0),
            ("Glycolic Acid", 8.0),
            ("Water", 91.0),
        ],
        category="skincare",
    )
    safety_report = validate(formula, mode="safety")
    discovery_report = validate(formula, mode="discovery")

    # Safety mode should flag this
    safety_compat = [i for i in safety_report.issues if i.check == "compatibility"]
    assert len(safety_compat) > 0

    # Discovery mode should have softer or no warnings
    discovery_warnings = [i for i in discovery_report.issues
                          if i.check == "compatibility" and i.severity == "warning"]
    safety_warnings = [i for i in safety_report.issues
                       if i.check == "compatibility" and i.severity == "warning"]
    assert len(discovery_warnings) <= len(safety_warnings)


def test_supplement_calcium_iron():
    """Calcium + Iron absorption competition is a soft rule."""
    formula = Formula(
        name="Multi-Mineral",
        ingredients=[
            ("Calcium Carbonate", 40.0),
            ("Ferrous Sulfate", 5.0),
            ("Maltodextrin", 55.0),
        ],
        category="supplement",
    )
    report = validate(formula)
    assert report.warnings > 0
    assert any("absorption" in i.message.lower() or "calcium" in i.message.lower()
               for i in report.issues)


# ---------------------------------------------------------------------------
# Conditions — rules only fire when conditions are met
# ---------------------------------------------------------------------------

def test_niacinamide_vitamin_c_high_ph_no_fire():
    """At high pH, niacinamide + vitamin C condition is not met."""
    formula = Formula(
        ingredients=[
            ("Niacinamide", 10.0),
            ("Ascorbic Acid", 15.0),
            ("Water", 75.0),
        ],
        target_ph=6.0,  # Above the 3.5 threshold
        category="skincare",
    )
    report = validate(formula, mode="safety")
    niac_issues = [i for i in report.issues
                   if i.check == "compatibility"
                   and "NIACINAMIDE" in (i.ingredient or "")
                   and "ASCORBIC" in (i.ingredient_b or "")]
    assert len(niac_issues) == 0


# ---------------------------------------------------------------------------
# Coverage Warnings
# ---------------------------------------------------------------------------

def test_pharma_coverage_warning():
    """Pharma formulas should get a coverage warning."""
    formula = Formula(
        ingredients=[
            ("Tacrolimus", 0.1),
            ("Water", 99.9),
        ],
        category="pharma",
    )
    report = validate(formula)
    coverage_issues = [i for i in report.issues if i.check == "coverage"]
    assert len(coverage_issues) > 0
    assert any("coverage" in i.message.lower() for i in coverage_issues)


def test_skincare_no_coverage_warning():
    """Skincare should NOT get a coverage warning (well-covered domain)."""
    formula = Formula(
        ingredients=[
            ("Water", 90.0),
            ("Glycerin", 10.0),
        ],
        category="skincare",
    )
    report = validate(formula)
    coverage_warnings = [i for i in report.issues
                         if i.check == "coverage" and i.severity == "warning"]
    assert len(coverage_warnings) == 0


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------

def test_formula_over_100_percent():
    """Formula exceeding 100% should be flagged."""
    formula = Formula(
        ingredients=[
            ("Water", 80.0),
            ("Glycerin", 25.0),
        ],
    )
    report = validate(formula)
    assert report.errors > 0


def test_formula_under_100_percent():
    """Formula well under 100% should warn."""
    formula = Formula(
        ingredients=[
            ("Retinol", 1.0),
            ("Glycerin", 5.0),
        ],
    )
    report = validate(formula)
    assert report.warnings > 0


def test_duplicate_ingredients():
    """Duplicate ingredients should be flagged."""
    formula = Formula(
        ingredients=[
            ("Glycerin", 5.0),
            ("Water", 85.0),
            ("Glycerin", 5.0),
        ],
    )
    report = validate(formula)
    assert any("duplicate" in i.message.lower() for i in report.issues)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_tuple_ingredient_shorthand():
    """Formulas should accept (name, percentage) tuples."""
    formula = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    assert len(formula.ingredients) == 2
    assert formula.ingredients[0].inci_name == "Water"


def test_report_string_output():
    """Report should have a readable string representation."""
    formula = Formula(
        name="Test Formula",
        ingredients=[("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5)],
    )
    report = validate(formula)
    output = str(report)
    assert "Test Formula" in output


def test_hlb_info_for_oils():
    """Formulas with known oils should get HLB information."""
    formula = Formula(
        ingredients=[
            ("Water", 70.0),
            ("Coconut Oil", 15.0),
            ("Squalane", 5.0),
            ("Polysorbate 80", 5.0),
            ("Glycerin", 5.0),
        ],
    )
    report = validate(formula)
    hlb_issues = [i for i in report.issues if i.check == "hlb"]
    assert len(hlb_issues) > 0


# ---------------------------------------------------------------------------
# Knowledge Base Integrity
# ---------------------------------------------------------------------------

def test_knowledge_loads():
    """Knowledge base should load without errors."""
    kb = load_knowledge()
    assert len(kb.interaction_rules) > 0
    assert len(kb.oil_hlb) > 0
    assert len(kb.aliases) > 0


def test_hard_rules_exist():
    """There should be hard rules for dangerous combinations."""
    kb = load_knowledge()
    assert len(kb.hard_rules) >= 15, "Expected at least 15 hard (unconditional) rules"


def test_soft_rules_have_confidence():
    """All soft rules should have a confidence score."""
    kb = load_knowledge()
    for rule in kb.soft_rules:
        assert 0 <= rule.confidence <= 1.0, f"Bad confidence for {rule.a} + {rule.b}: {rule.confidence}"


def test_all_rules_have_sources():
    """All rules should cite a source."""
    kb = load_knowledge()
    for rule in kb.interaction_rules:
        assert rule.source, f"Missing source for {rule.a} + {rule.b}"


def test_beverage_casein_acid():
    """Casein + Citric Acid in beverages should error (hard rule)."""
    formula = Formula(
        ingredients=[
            ("Casein", 5.0),
            ("Citric Acid", 2.0),
            ("Water", 93.0),
        ],
        category="beverage",
    )
    report = validate(formula)
    assert report.errors > 0


def test_clean_formula_passes():
    """A simple, well-formulated moisturizer should pass."""
    formula = Formula(
        name="Simple Moisturizer",
        ingredients=[
            ("Water", 75.0),
            ("Glycerin", 5.0),
            ("Cetearyl Alcohol", 4.0),
            ("Caprylic/Capric Triglyceride", 10.0),
            ("Phenoxyethanol", 1.0),
            ("Xanthan Gum", 0.5),
        ],
        category="skincare",
    )
    report = validate(formula)
    assert report.errors == 0
    assert report.overall_score >= 80
