"""Tests for the physics observation engine."""

import sys
from unittest.mock import patch
from openmix import Formula
from openmix.observe import (
    observe,
    Observation,
    Violation,
    FormulationObservation,
)
from openmix.resolver.resolve import ResolvedIngredient

# Use sys.modules for unambiguous module reference (avoids name collision
# between openmix.observe the module and observe the function on Python 3.10)
_observe_mod = sys.modules["openmix.observe"]


# ---------------------------------------------------------------------------
# Helper: mock resolver so tests don't hit PubChem or require seed cache
# ---------------------------------------------------------------------------

def _mock_resolve(inci_name: str) -> ResolvedIngredient:
    """Deterministic mock resolver for testing."""
    db = {
        "WATER": ResolvedIngredient(inci_name="Water", resolved=False),
        "GLYCERIN": ResolvedIngredient(
            inci_name="Glycerin", smiles="OCC(O)CO",
            molecular_weight=92.09, log_p=-1.76, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "NIACINAMIDE": ResolvedIngredient(
            inci_name="Niacinamide", smiles="c1ccc(c(c1)C(=O)N)N",
            molecular_weight=122.12, log_p=-0.35, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "RETINOL": ResolvedIngredient(
            inci_name="Retinol", smiles="CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C",
            molecular_weight=286.45, log_p=5.68, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "PHENOXYETHANOL": ResolvedIngredient(
            inci_name="Phenoxyethanol", smiles="C1=CC=C(C=C1)OCCO",
            molecular_weight=138.16, log_p=1.16, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "CETEARYL ALCOHOL": ResolvedIngredient(
            inci_name="Cetearyl Alcohol", smiles="CCCCCCCCCCCCCCCCO",
            molecular_weight=242.44, log_p=6.83, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "CAPRYLIC/CAPRIC TRIGLYCERIDE": ResolvedIngredient(
            inci_name="Caprylic/Capric Triglyceride",
            molecular_weight=470.0, log_p=8.0, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "SODIUM LAURYL SULFATE": ResolvedIngredient(
            inci_name="Sodium Lauryl Sulfate",
            smiles="CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
            molecular_weight=288.38, log_p=1.6, resolved=True, source="test",
            charge_type="anionic",
        ),
        "CETRIMONIUM CHLORIDE": ResolvedIngredient(
            inci_name="Cetrimonium Chloride",
            smiles="CCCCCCCCCCCCCCCC[N+](C)(C)C.[Cl-]",
            molecular_weight=320.0, log_p=3.0, resolved=True, source="test",
            charge_type="cationic",
        ),
        "XANTHAN GUM": ResolvedIngredient(
            inci_name="Xanthan Gum", resolved=False,
        ),
        "ASCORBIC ACID": ResolvedIngredient(
            inci_name="Ascorbic Acid", smiles="OC(=O)C1OC(=O)C(O)=C1O",
            molecular_weight=176.12, log_p=-1.85, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "BENZOYL PEROXIDE": ResolvedIngredient(
            inci_name="Benzoyl Peroxide",
            molecular_weight=242.23, log_p=2.8, resolved=True, source="test",
            charge_type="nonionic",
        ),
        "CYCLOSPORINE A": ResolvedIngredient(
            inci_name="Cyclosporine A",
            molecular_weight=1202.61, log_p=2.9, resolved=True, source="test",
            charge_type="nonionic",
        ),
    }
    return db.get(inci_name.upper().strip(),
                  ResolvedIngredient(inci_name=inci_name, resolved=False))


def _observe(formula: Formula) -> FormulationObservation:
    """Run observe() with mocked resolver."""
    with patch.object(_observe_mod, "resolve", side_effect=_mock_resolve):
        return observe(formula)


# ---------------------------------------------------------------------------
# Basic observation structure
# ---------------------------------------------------------------------------

def test_observe_returns_formulation_observation():
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    assert isinstance(obs, FormulationObservation)
    assert isinstance(obs.observations, list)
    assert isinstance(obs.violations, list)


def test_observe_sets_formula_name():
    f = Formula(name="Test Serum", ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    assert obs.formula_name == "Test Serum"


def test_observe_resolution_rate():
    """Water is unresolved in mock, Glycerin is resolved -> 50% rate."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    assert obs.resolution_rate == 0.5


def test_observe_all_resolved():
    """Formula with all resolved ingredients -> 100% rate."""
    f = Formula(ingredients=[("Glycerin", 50.0), ("Niacinamide", 50.0)])
    obs = _observe(f)
    assert obs.resolution_rate == 1.0


# ---------------------------------------------------------------------------
# Structural observations
# ---------------------------------------------------------------------------

def test_structural_total_ok():
    """Formula totaling 100% should show confirmed."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    structural = [o for o in obs.observations
                  if o.category == "structural" and o.subject == "formula"
                  and "Total" in o.observed]
    assert len(structural) == 1
    assert structural[0].agreement == "confirmed"


def test_structural_total_over():
    """Formula over 101% should flag discrepancy."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 15.0)])
    obs = _observe(f)
    structural = [o for o in obs.observations
                  if o.category == "structural" and "Total" in o.observed]
    assert len(structural) == 1
    assert structural[0].agreement == "discrepancy"


def test_structural_total_under():
    """Formula under 99% should flag discrepancy."""
    f = Formula(ingredients=[("Water", 50.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    structural = [o for o in obs.observations
                  if o.category == "structural" and "Total" in o.observed]
    assert structural[0].agreement == "discrepancy"


def test_structural_preservative_missing():
    """Water-based formula without preservative -> discrepancy."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    preserv = [o for o in obs.observations
               if o.category == "structural" and "preservative" in o.observed.lower()]
    assert len(preserv) == 1
    assert preserv[0].agreement == "discrepancy"


def test_structural_preservative_present():
    """Water-based formula with preservative should NOT flag preservative concern."""
    f = Formula(ingredients=[
        ("Water", 89.0), ("Glycerin", 10.0), ("Phenoxyethanol", 1.0),
    ])
    obs = _observe(f)
    preserv = [o for o in obs.observations
               if o.category == "structural" and "preservative" in o.observed.lower()]
    assert len(preserv) == 0


def test_structural_duplicate_ingredient():
    """Duplicate ingredients should be flagged."""
    f = Formula(ingredients=[
        ("Glycerin", 5.0), ("Water", 85.0), ("Glycerin", 10.0),
    ])
    obs = _observe(f)
    dupes = [o for o in obs.observations
             if o.category == "structural" and "more than once" in o.observed.lower()]
    assert len(dupes) >= 1


# ---------------------------------------------------------------------------
# Molecular observations
# ---------------------------------------------------------------------------

def test_molecular_hydrophobic_warning():
    """High LogP ingredient at >2% should produce an observation."""
    f = Formula(ingredients=[
        ("Water", 80.0), ("Retinol", 6.0), ("Glycerin", 14.0),
    ])
    obs = _observe(f)
    mol = [o for o in obs.observations
           if o.category == "molecular" and "Retinol" in o.subject]
    assert len(mol) >= 1
    # At 6% with LogP 5.68, should be discrepancy (>5%)
    assert mol[0].agreement == "discrepancy"


def test_molecular_hydrophobic_uncertain():
    """High LogP at 3% (>2 but <5) should be uncertain, not discrepancy."""
    f = Formula(ingredients=[
        ("Water", 87.0), ("Retinol", 3.0), ("Glycerin", 10.0),
    ])
    obs = _observe(f)
    mol = [o for o in obs.observations
           if o.category == "molecular" and "Retinol" in o.subject]
    assert len(mol) >= 1
    assert mol[0].agreement == "uncertain"


def test_molecular_hydrophilic_confirmed():
    """Very negative LogP (< -2.0) should confirm hydrophilicity."""
    # Mock an ingredient with LogP well below -2.0
    hyper_hydrophilic = ResolvedIngredient(
        inci_name="Mannitol", smiles="OCC(O)C(O)C(O)C(O)CO",
        molecular_weight=182.17, log_p=-3.1, resolved=True, source="test",
        charge_type="nonionic",
    )

    def mock_with_mannitol(inci_name):
        if inci_name.upper().strip() == "MANNITOL":
            return hyper_hydrophilic
        return _mock_resolve(inci_name)

    f = Formula(ingredients=[("Mannitol", 10.0), ("Water", 90.0)])
    with patch.object(_observe_mod, "resolve", side_effect=mock_with_mannitol):
        obs = observe(f)
    mol = [o for o in obs.observations
           if o.category == "molecular" and "hydrophilic" in o.observed.lower()]
    assert len(mol) == 1
    assert mol[0].agreement == "confirmed"


def test_molecular_large_molecule_flagged():
    """MW > 500 should produce an observation."""
    f = Formula(ingredients=[
        ("Cyclosporine A", 5.0), ("Water", 95.0),
    ])
    obs = _observe(f)
    mol = [o for o in obs.observations
           if o.category == "molecular" and "Cyclosporine A" in o.subject
           and "MW" in o.observed]
    assert len(mol) == 1
    assert mol[0].agreement == "uncertain"
    assert "Lipinski" in mol[0].expected or "500" in mol[0].expected


# ---------------------------------------------------------------------------
# Charge observations
# ---------------------------------------------------------------------------

def test_charge_conflict_detected():
    """Anionic + cationic at meaningful % should flag discrepancy."""
    f = Formula(ingredients=[
        ("Sodium Lauryl Sulfate", 10.0),
        ("Cetrimonium Chloride", 5.0),
        ("Water", 85.0),
    ])
    obs = _observe(f)
    charge = [o for o in obs.observations if o.category == "charge"]
    assert len(charge) >= 1
    assert charge[0].agreement == "discrepancy"


def test_charge_uniform_confirmed():
    """Single charge type should confirm no conflict."""
    f = Formula(ingredients=[
        ("Sodium Lauryl Sulfate", 10.0),
        ("Water", 90.0),
    ])
    obs = _observe(f)
    charge = [o for o in obs.observations if o.category == "charge"]
    assert len(charge) >= 1
    assert charge[0].agreement == "confirmed"


# ---------------------------------------------------------------------------
# Knowledge base violations
# ---------------------------------------------------------------------------

def test_hard_violation_bpo_retinol():
    """BPO + Retinol should produce a hard violation."""
    f = Formula(ingredients=[
        ("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5),
    ])
    obs = _observe(f)
    assert obs.hard_violations > 0
    hard = [v for v in obs.violations if v.severity == "hard"]
    assert any("RETINOL" in v.ingredients[0] or "RETINOL" in v.ingredients[1]
               for v in hard)


def test_soft_violation_retinol_glycolic():
    """Retinol + Glycolic Acid should produce a soft violation."""
    f = Formula(ingredients=[
        ("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0),
    ])
    obs = _observe(f)
    assert obs.soft_violations > 0


# ---------------------------------------------------------------------------
# Phase observations
# ---------------------------------------------------------------------------

def test_phase_hydrophobic_detected():
    """Oil-phase ingredients should be flagged for emulsification."""
    f = Formula(ingredients=[
        ("Water", 70.0),
        ("Caprylic/Capric Triglyceride", 15.0),
        ("Cetearyl Alcohol", 10.0),
        ("Glycerin", 5.0),
    ])
    obs = _observe(f)
    phase = [o for o in obs.observations if o.category == "phase"]
    assert len(phase) >= 1
    assert "Hydrophobic" in phase[0].observed or "hydrophobic" in phase[0].observed.lower()


# ---------------------------------------------------------------------------
# Concern count & concern score
# ---------------------------------------------------------------------------

def test_clean_formula_low_concerns():
    """Well-formulated moisturizer should have low concern count."""
    f = Formula(ingredients=[
        ("Water", 75.0), ("Glycerin", 5.0),
        ("Niacinamide", 4.0), ("Phenoxyethanol", 1.0),
        ("Ascorbic Acid", 5.0),
    ])
    obs = _observe(f)
    # No hard violations expected
    assert obs.hard_violations == 0
    # Under 99% total is the main structural concern
    assert obs.concern_count < 30


def test_hard_violation_high_concern_count():
    """Hard violations should dominate concern count."""
    f = Formula(ingredients=[
        ("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5),
    ])
    obs = _observe(f)
    assert obs.concern_count >= 10  # at least one hard violation * 10


def test_concern_score_inverse_of_concerns():
    """concern_score should be 100 for zero-concern formula, lower for problematic ones."""
    clean = Formula(ingredients=[("Glycerin", 50.0), ("Niacinamide", 50.0)])
    dirty = Formula(ingredients=[
        ("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5),
    ])
    clean_obs = _observe(clean)
    dirty_obs = _observe(dirty)
    assert clean_obs.concern_score >= dirty_obs.concern_score


# ---------------------------------------------------------------------------
# String output
# ---------------------------------------------------------------------------

def test_str_output_readable():
    """__str__ should produce readable output with key sections."""
    f = Formula(
        name="Test Serum",
        ingredients=[
            ("Water", 80.0), ("Retinol", 6.0), ("Glycerin", 14.0),
        ],
    )
    obs = _observe(f)
    output = str(obs)
    assert "Test Serum" in output
    assert "Resolved" in output
    assert "Concern count" in output


def test_str_shows_violations():
    """String output should show violations when present."""
    f = Formula(ingredients=[
        ("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5),
    ])
    obs = _observe(f)
    output = str(obs)
    assert "HARD" in output or "Violations" in output


def test_str_shows_discrepancies():
    """String output should mark discrepancies with !."""
    f = Formula(ingredients=[
        ("Water", 80.0), ("Retinol", 6.0), ("Glycerin", 14.0),
    ])
    obs = _observe(f)
    output = str(obs)
    assert "[!]" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_formula():
    """Empty formula should not crash."""
    f = Formula(ingredients=[])
    obs = _observe(f)
    assert isinstance(obs, FormulationObservation)
    assert obs.resolution_rate == 0


def test_single_ingredient():
    """Single ingredient formula should work."""
    f = Formula(ingredients=[("Glycerin", 100.0)])
    obs = _observe(f)
    assert obs.resolution_rate == 1.0
    assert obs.hard_violations == 0


def test_unresolved_ingredients_still_checked():
    """Unresolved ingredients should still get structural checks."""
    f = Formula(ingredients=[
        ("Water", 50.0), ("Xanthan Gum", 50.0),
    ])
    obs = _observe(f)
    # Both unresolved in mock (Water explicitly unresolved, Xanthan Gum unresolved)
    assert obs.resolution_rate == 0.0
    # Should still get structural observations
    structural = [o for o in obs.observations if o.category == "structural"]
    assert len(structural) >= 1


# ---------------------------------------------------------------------------
# Dual Modes: Engineering vs Discovery
# ---------------------------------------------------------------------------

def _observe_mode(formula: Formula, mode: str) -> FormulationObservation:
    """Run observe() with mocked resolver in a specific mode."""
    with patch.object(_observe_mod, "resolve", side_effect=_mock_resolve):
        return observe(formula, mode=mode)


def test_engineering_mode_default():
    """observe() without mode arg should default to engineering."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe(f)
    assert obs.mode == "engineering"


def test_mode_stored_on_result():
    """FormulationObservation.mode should reflect what was passed."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    eng = _observe_mode(f, "engineering")
    disc = _observe_mode(f, "discovery")
    assert eng.mode == "engineering"
    assert disc.mode == "discovery"


def test_engineering_concern_count_matches_current():
    """Engineering mode concern_count should include soft violations + concerns."""
    f = Formula(ingredients=[
        ("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0),
    ])
    obs = _observe_mode(f, "engineering")
    # Retinol + Glycolic is soft violation -> adds to concern_count
    assert obs.soft_violations > 0
    assert obs.concern_count > 0


def test_discovery_mode_soft_violations_not_concerns():
    """In discovery mode, soft violations should NOT add to concern_count."""
    f = Formula(ingredients=[
        ("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0),
    ])
    eng = _observe_mode(f, "engineering")
    disc = _observe_mode(f, "discovery")

    # Both should detect the same soft violations
    assert eng.soft_violations == disc.soft_violations
    assert eng.soft_violations > 0

    # But discovery mode doesn't count them as concerns
    assert disc.concern_count < eng.concern_count


def test_discovery_mode_hard_violations_still_count():
    """Safety is non-negotiable — hard violations count in BOTH modes."""
    f = Formula(ingredients=[
        ("Benzoyl Peroxide", 2.5), ("Retinol", 1.0), ("Water", 96.5),
    ])
    eng = _observe_mode(f, "engineering")
    disc = _observe_mode(f, "discovery")

    assert eng.hard_violations > 0
    assert disc.hard_violations > 0
    # Both modes penalize hard violations equally
    assert disc.concern_count >= 10  # hard * 10


def test_discovery_mode_signals_property():
    """signals property should list soft violations."""
    f = Formula(ingredients=[
        ("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0),
    ])
    obs = _observe_mode(f, "discovery")
    assert len(obs.signals) > 0
    assert all(s.severity == "soft" for s in obs.signals)


def test_discovery_mode_discoveries_property():
    """discoveries should surface low-confidence discrepancies."""
    # Retinol at 3% triggers an uncertain molecular observation (conf 0.5)
    f = Formula(ingredients=[
        ("Water", 87.0), ("Retinol", 3.0), ("Glycerin", 10.0),
    ])
    obs = _observe_mode(f, "discovery")
    # The uncertain observation has confidence 0.5, below 0.7 threshold
    # But it's "uncertain" not "discrepancy", so it shouldn't be in discoveries
    # Let's use a formula that produces actual low-confidence discrepancies
    f2 = Formula(ingredients=[
        ("Water", 80.0), ("Retinol", 6.0), ("Glycerin", 14.0),
    ])
    obs2 = _observe_mode(f2, "discovery")
    # Retinol at 6% with LogP 5.68 → discrepancy with confidence 0.5 (< 0.7)
    assert len(obs2.discoveries) >= 1
    assert all(d.confidence < 0.7 for d in obs2.discoveries)


def test_discovery_str_output():
    """Discovery mode __str__ should show signals/knowledge gaps, not concern count."""
    f = Formula(ingredients=[
        ("Retinol", 1.0), ("Glycolic Acid", 8.0), ("Water", 91.0),
    ])
    obs = _observe_mode(f, "discovery")
    output = str(obs)
    assert "discovery" in output.lower()
    assert "Signals" in output or "signals" in output
    assert "Knowledge gaps" in output or "knowledge gaps" in output


def test_engineering_str_output():
    """Engineering mode __str__ should show concern count."""
    f = Formula(ingredients=[("Water", 90.0), ("Glycerin", 10.0)])
    obs = _observe_mode(f, "engineering")
    output = str(obs)
    assert "engineering" in output.lower()
    assert "Concern count" in output
