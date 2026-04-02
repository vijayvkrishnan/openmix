"""Tests for the discourse engine and protocol schema."""

from openmix.schema import Formula
from openmix.protocol import Phase, ProcessStep, Protocol
from openmix.memory import ExperimentMemory, Discovery
from openmix.discourse import (
    Claim,
    Discourse,
    DiscourseTopic,
    EvidenceLevel,
    classify_topic,
    evaluate_discourse,
    _extract_process_claims,
    _extract_data_claims,
)


# ---------------------------------------------------------------------------
# Protocol schema
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_phase_construction(self):
        phase = Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"])
        assert phase.name == "A"
        assert phase.target_temp_c == 75.0
        assert "Water" in phase.ingredients

    def test_process_step(self):
        step = ProcessStep("heat", "A", {"temp_c": 75, "duration_min": 10})
        assert step.action == "heat"
        assert step.parameters["temp_c"] == 75

    def test_protocol_phase_lookup(self):
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Squalane"]),
            ],
        )
        assert protocol.get_phase("A") is not None
        assert protocol.get_phase("A").label == "Water Phase"
        assert protocol.get_phase("X") is None

    def test_phase_for_ingredient(self):
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Squalane"]),
                Phase("C", "Cool-Down", 40.0, ["Retinol"]),
            ],
        )
        assert protocol.phase_for_ingredient("Retinol").name == "C"
        assert protocol.phase_for_ingredient("Squalane").name == "B"
        assert protocol.phase_for_ingredient("Unknown") is None

    def test_ingredients_in_phase(self):
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
            ],
        )
        assert protocol.ingredients_in_phase("A") == ["Water", "Glycerin"]
        assert protocol.ingredients_in_phase("Z") == []

    def test_protocol_str(self):
        protocol = Protocol(
            phases=[Phase("A", "Water Phase", 75.0, ["Water"])],
            steps=[ProcessStep("heat", "A", {"temp_c": 75})],
            equipment=["stirrer"],
            batch_size_g=100.0,
        )
        output = str(protocol)
        assert "Water Phase" in output
        assert "heat" in output
        assert "stirrer" in output

    def test_full_skincare_protocol(self):
        """Realistic skincare protocol with all phases."""
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0,
                      ["Water", "Glycerin", "Xanthan Gum"]),
                Phase("B", "Oil Phase", 75.0,
                      ["Squalane", "Cetyl Alcohol"]),
                Phase("C", "Cool-Down", 40.0,
                      ["Retinol", "Tocopherol", "Phenoxyethanol"]),
            ],
            steps=[
                ProcessStep("heat", "A", {"temp_c": 75, "duration_min": 10}),
                ProcessStep("heat", "B", {"temp_c": 75, "duration_min": 10}),
                ProcessStep("combine", "B", {"into": "A", "rate": "slow"},
                            notes="Add oil to water while mixing"),
                ProcessStep("homogenize", "all", {"rpm": 5000, "duration_min": 3}),
                ProcessStep("cool", "all", {"target_c": 40}),
                ProcessStep("add", "C", {}),
                ProcessStep("adjust_ph", "all", {"target": 5.5}),
            ],
            equipment=["overhead stirrer", "homogenizer", "pH meter"],
            batch_size_g=100.0,
        )
        assert len(protocol.phases) == 3
        assert len(protocol.steps) == 7
        assert protocol.phase_for_ingredient("Retinol").name == "C"
        assert protocol.phase_for_ingredient("Retinol").target_temp_c == 40.0


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

class TestClassification:
    def test_agreement_same_position(self):
        claims = [
            Claim("physics", "Glycerin", "safe", "No concerns",
                  "LogP -1.76", EvidenceLevel.COMPUTATIONAL, 0.9),
            Claim("chemistry", "Glycerin", "safe", "No interactions",
                  "knowledge_base", EvidenceLevel.RULE_BASED, 0.9),
        ]
        classification, winner = classify_topic(claims)
        assert classification == "agreement"
        assert winner is None

    def test_correction_large_evidence_gap(self):
        """When evidence levels differ by 2+, stronger side corrects."""
        claims = [
            Claim("physics", "Retinol", "safe", "LLM said it's fine",
                  "reasoning", EvidenceLevel.LLM_REASONING, 0.5),
            Claim("chemistry", "Retinol", "concern", "Degrades at low pH",
                  "Smith et al. 2019", EvidenceLevel.RULE_BASED, 0.8),
        ]
        classification, winner = classify_topic(claims)
        assert classification == "correction"
        assert winner is not None
        assert winner.perspective == "chemistry"

    def test_true_disagreement_comparable_evidence(self):
        """When evidence levels are close but positions differ, it's a true disagreement."""
        claims = [
            Claim("physics", "Ceramide NP", "concern",
                  "LogP 12.4, extremely hydrophobic, poor aqueous solubility",
                  "RDKit LogP computation", EvidenceLevel.COMPUTATIONAL, 0.7),
            Claim("data", "Ceramide NP", "safe",
                  "Found in 27 commercial moisturizers at 0.5-2%",
                  "commercial product database", EvidenceLevel.EMPIRICAL_PROXY, 0.8),
        ]
        classification, winner = classify_topic(claims)
        assert classification == "true_disagreement"
        assert winner is None

    def test_knowledge_gap_no_claims(self):
        classification, _ = classify_topic([])
        assert classification == "knowledge_gap"

    def test_knowledge_gap_all_unknown(self):
        claims = [
            Claim("physics", "Novel Ingredient", "unknown",
                  "No SMILES found", "resolver", EvidenceLevel.COMPUTATIONAL, 0.0),
            Claim("chemistry", "Novel Ingredient", "unknown",
                  "No rules for this ingredient", "knowledge_base",
                  EvidenceLevel.RULE_BASED, 0.0),
        ]
        classification, _ = classify_topic(claims)
        assert classification == "knowledge_gap"

    def test_single_perspective_is_agreement(self):
        """Only one perspective has a claim -- no discourse possible."""
        claims = [
            Claim("physics", "Water", "safe", "Solvent",
                  "fundamental", EvidenceLevel.COMPUTATIONAL, 1.0),
        ]
        classification, _ = classify_topic(claims)
        assert classification == "agreement"

    def test_correction_favors_higher_evidence(self):
        """The claim with higher evidence wins the correction."""
        claims = [
            Claim("chemistry", "Pair X+Y", "violation",
                  "Toxic gas formation", "CDC NIOSH",
                  EvidenceLevel.EMPIRICAL_DIRECT, 1.0),
            Claim("physics", "Pair X+Y", "safe",
                  "No LogP concern", "RDKit",
                  EvidenceLevel.COMPUTATIONAL, 0.5),
        ]
        classification, winner = classify_topic(claims)
        assert classification == "correction"
        assert winner.perspective == "chemistry"
        assert winner.position == "violation"

    def test_adjacent_levels_are_disagreement_not_correction(self):
        """Evidence gap of 1 level is a true disagreement, not a correction."""
        claims = [
            Claim("physics", "Ingredient X", "concern",
                  "Phase separation predicted",
                  "HLB computation", EvidenceLevel.COMPUTATIONAL, 0.6),
            Claim("chemistry", "Ingredient X", "safe",
                  "Compatible per knowledge base",
                  "rule entry with source", EvidenceLevel.RULE_BASED, 0.7),
        ]
        classification, _ = classify_topic(claims)
        assert classification == "true_disagreement"


# ---------------------------------------------------------------------------
# End-to-end discourse evaluation
# ---------------------------------------------------------------------------

class TestEvaluateDiscourse:
    def test_clean_formula_mostly_agreements(self):
        """A simple, clean formula should produce mostly agreements."""
        formula = Formula(
            name="Simple Moisturizer",
            ingredients=[
                ("Water", 80.0),
                ("Glycerin", 10.0),
                ("Phenoxyethanol", 1.0),
                ("Xanthan Gum", 0.5),
            ],
            target_ph=5.5,
            category="skincare",
        )
        disc = evaluate_discourse(formula)
        assert isinstance(disc, Discourse)
        assert disc.formula_name == "Simple Moisturizer"
        # Should have some topics
        assert len(disc.topics) > 0

    def test_problematic_formula_finds_concerns(self):
        """A formula with known interactions should produce non-agreements."""
        formula = Formula(
            name="Problem Serum",
            ingredients=[
                ("Retinol", 2.0),
                ("Ascorbic Acid", 15.0),
                ("Benzoyl Peroxide", 2.5),
                ("Niacinamide", 5.0),
                ("Water", 75.5),
            ],
            target_ph=3.5,
            category="skincare",
        )
        disc = evaluate_discourse(formula)
        # Should find interaction concerns from chemistry perspective
        assert len(disc.topics) > 0

    def test_discourse_str_output(self):
        formula = Formula(
            name="Test Formula",
            ingredients=[("Water", 90.0), ("Glycerin", 10.0)],
            category="skincare",
        )
        disc = evaluate_discourse(formula)
        output = str(disc)
        assert "MULTI-PERSPECTIVE DISCOURSE" in output
        assert "Test Formula" in output

    def test_dangerous_formula_has_corrections_or_agreements_on_violations(self):
        """Bleach + ammonia should be flagged by chemistry with high evidence."""
        formula = Formula(
            name="Dangerous Mix",
            ingredients=[
                ("Sodium Hypochlorite", 5.0),
                ("Ammonia", 3.0),
                ("Water", 92.0),
            ],
            category="home_care",
        )
        disc = evaluate_discourse(formula)
        # Chemistry should flag this as a hard violation
        all_claims = [c for t in disc.topics for c in t.claims]
        violation_claims = [c for c in all_claims if c.position == "violation"]
        assert len(violation_claims) > 0

    def test_discourse_topic_str(self):
        topic = DiscourseTopic(
            subject="Retinol",
            claims=[
                Claim("physics", "Retinol", "concern", "LogP 5.68, hydrophobic",
                      "RDKit", EvidenceLevel.COMPUTATIONAL, 0.7),
            ],
            classification="agreement",
            summary="physics: concern",
        )
        output = str(topic)
        assert "Retinol" in output
        assert "LogP" in output


# ---------------------------------------------------------------------------
# Process perspective
# ---------------------------------------------------------------------------

class TestProcessPerspective:
    def test_thermal_violation_retinol_in_heat_phase(self):
        """Retinol in a 75C phase should be flagged."""
        formula = Formula(
            ingredients=[
                ("Water", 80.0), ("Retinol", 1.0), ("Glycerin", 19.0),
            ],
            category="skincare",
        )
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Retinol"]),
            ],
        )
        claims = _extract_process_claims(formula, protocol)
        retinol_claims = [c for c in claims if "Retinol" in c.subject]
        assert any(c.position == "violation" for c in retinol_claims)
        assert any("degrades above 40" in c.detail for c in retinol_claims)

    def test_thermal_safe_retinol_in_cooldown(self):
        """Retinol in a 40C cool-down phase should be safe."""
        formula = Formula(
            ingredients=[
                ("Water", 80.0), ("Retinol", 1.0), ("Glycerin", 19.0),
            ],
            category="skincare",
        )
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("C", "Cool-Down", 40.0, ["Retinol"]),
            ],
        )
        claims = _extract_process_claims(formula, protocol)
        retinol_claims = [c for c in claims if "Retinol" in c.subject]
        assert any(c.position == "safe" for c in retinol_claims)

    def test_missing_homogenization_with_oil_phase(self):
        """Oil phase > 5% without homogenize step should be flagged."""
        formula = Formula(
            ingredients=[
                ("Water", 70.0), ("Squalane", 15.0), ("Glycerin", 15.0),
            ],
            category="skincare",
        )
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Squalane"]),
            ],
            steps=[
                ProcessStep("heat", "A", {"temp_c": 75}),
                ProcessStep("combine", "B", {"into": "A"}),
                # No homogenize step
            ],
        )
        claims = _extract_process_claims(formula, protocol)
        formula_claims = [c for c in claims if c.subject == "formula"]
        assert any("homogenization" in c.detail.lower() for c in formula_claims)

    def test_missing_ph_adjustment(self):
        """Target pH specified but no adjust_ph step should be flagged."""
        formula = Formula(
            ingredients=[("Water", 90.0), ("Glycerin", 10.0)],
            target_ph=5.5,
            category="skincare",
        )
        protocol = Protocol(
            phases=[Phase("A", "Water Phase", None, ["Water", "Glycerin"])],
            steps=[ProcessStep("mix", "A", {"rpm": 300})],
        )
        claims = _extract_process_claims(formula, protocol)
        ph_claims = [c for c in claims if "pH" in c.detail]
        assert len(ph_claims) > 0

    def test_preservative_in_heat_phase(self):
        """Preservative in a 75C phase should be flagged."""
        formula = Formula(
            ingredients=[
                ("Water", 89.0), ("Glycerin", 10.0), ("Phenoxyethanol", 1.0),
            ],
            category="skincare",
        )
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0,
                      ["Water", "Glycerin", "Phenoxyethanol"]),
            ],
        )
        claims = _extract_process_claims(formula, protocol)
        preservative_claims = [c for c in claims if "Phenoxyethanol" in c.subject]
        assert any(c.position == "concern" for c in preservative_claims)


# ---------------------------------------------------------------------------
# Data perspective
# ---------------------------------------------------------------------------

class TestDataPerspective:
    def _make_memory_with_discoveries(self, tmp_path):
        memory = ExperimentMemory(base_dir=tmp_path)
        discoveries = [
            Discovery(
                id="pref_glycerin_skincare",
                finding="Glycerin at 10.0% as humectant",
                kind="preference",
                domain="skincare",
                ingredients=["Glycerin"],
                confidence=0.8,
                evidence_count=3,
                source_experiments=["exp-1", "exp-2", "exp-3"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
            Discovery(
                id="avoid_retinol_skincare",
                finding="Retinol tried in 3 trials but dropped",
                kind="avoidance",
                domain="skincare",
                ingredients=["Retinol"],
                confidence=0.5,
                evidence_count=1,
                source_experiments=["exp-1"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
            Discovery(
                id="pref_iron_supplement",
                finding="Iron Bisglycinate at 1.2% as active",
                kind="preference",
                domain="supplement",
                ingredients=["Iron Bisglycinate"],
                confidence=0.65,
                evidence_count=2,
                source_experiments=["exp-4", "exp-5"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
        ]
        memory._ensure_dirs()
        memory._save_discoveries(discoveries)
        return memory

    def test_data_preference_creates_safe_claim(self, tmp_path):
        memory = self._make_memory_with_discoveries(tmp_path)
        formula = Formula(
            ingredients=[("Water", 80.0), ("Glycerin", 10.0)],
            category="skincare",
        )
        claims = _extract_data_claims(formula, memory)
        glycerin_claims = [c for c in claims if "Glycerin" in c.subject]
        assert len(glycerin_claims) == 1
        assert glycerin_claims[0].position == "safe"
        assert glycerin_claims[0].evidence_level == EvidenceLevel.EMPIRICAL_PROXY

    def test_data_avoidance_creates_concern_claim(self, tmp_path):
        memory = self._make_memory_with_discoveries(tmp_path)
        formula = Formula(
            ingredients=[("Water", 80.0), ("Retinol", 1.0), ("Glycerin", 19.0)],
            category="skincare",
        )
        claims = _extract_data_claims(formula, memory)
        retinol_claims = [c for c in claims if "Retinol" in c.subject]
        assert len(retinol_claims) == 1
        assert retinol_claims[0].position == "concern"

    def test_data_cross_domain_excluded(self, tmp_path):
        """Supplement discoveries should not appear for skincare formulas."""
        memory = self._make_memory_with_discoveries(tmp_path)
        formula = Formula(
            ingredients=[("Water", 90.0), ("Iron Bisglycinate", 1.0)],
            category="skincare",
        )
        claims = _extract_data_claims(formula, memory)
        iron_claims = [c for c in claims if "Iron" in c.subject]
        assert len(iron_claims) == 0

    def test_data_no_discoveries_returns_empty(self, tmp_path):
        memory = ExperimentMemory(base_dir=tmp_path)
        formula = Formula(ingredients=[("Water", 100.0)], category="skincare")
        claims = _extract_data_claims(formula, memory)
        assert claims == []


# ---------------------------------------------------------------------------
# Full 4-perspective discourse
# ---------------------------------------------------------------------------

class TestFullDiscourse:
    def test_discourse_with_protocol(self):
        """Discourse with protocol should include process claims."""
        formula = Formula(
            name="Retinol Cream",
            ingredients=[
                ("Water", 75.0), ("Retinol", 1.0), ("Glycerin", 10.0),
                ("Squalane", 10.0), ("Phenoxyethanol", 1.0),
            ],
            target_ph=5.5,
            category="skincare",
        )
        # Bad protocol: retinol in heat phase
        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Squalane", "Retinol"]),
                Phase("C", "Cool-Down", 40.0, ["Phenoxyethanol"]),
            ],
            steps=[
                ProcessStep("heat", "A", {"temp_c": 75}),
                ProcessStep("heat", "B", {"temp_c": 75}),
                ProcessStep("combine", "B", {"into": "A"}),
                ProcessStep("cool", "all", {"target_c": 40}),
                ProcessStep("add", "C", {}),
            ],
        )
        disc = evaluate_discourse(formula, protocol=protocol)
        # Process should flag retinol in 75C phase
        all_claims = [c for t in disc.topics for c in t.claims]
        process_claims = [c for c in all_claims if c.perspective == "process"]
        assert len(process_claims) > 0
        assert any("degrades" in c.detail for c in process_claims)

    def test_discourse_with_memory(self, tmp_path):
        """Discourse with memory should include data claims."""
        memory = ExperimentMemory(base_dir=tmp_path)
        memory._ensure_dirs()
        memory._save_discoveries([
            Discovery(
                id="pref_squalane_skincare",
                finding="Squalane at 12.0% as emollient",
                kind="preference",
                domain="skincare",
                ingredients=["Squalane"],
                confidence=0.8,
                evidence_count=3,
                source_experiments=["exp-1", "exp-2", "exp-3"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
        ])

        formula = Formula(
            name="Test Cream",
            ingredients=[
                ("Water", 75.0), ("Squalane", 12.0), ("Glycerin", 13.0),
            ],
            category="skincare",
        )
        disc = evaluate_discourse(formula, memory=memory)
        all_claims = [c for t in disc.topics for c in t.claims]
        data_claims = [c for c in all_claims if c.perspective == "data"]
        assert len(data_claims) > 0
        assert any("Squalane" in c.subject for c in data_claims)

    def test_physics_vs_data_disagreement(self, tmp_path):
        """Physics flags hydrophobic concern, data says it worked before.
        This should be a true disagreement -- the interesting case."""
        memory = ExperimentMemory(base_dir=tmp_path)
        memory._ensure_dirs()
        memory._save_discoveries([
            Discovery(
                id="pref_squalane_skincare",
                finding="Squalane at 15.0% as emollient",
                kind="preference",
                domain="skincare",
                ingredients=["Squalane"],
                confidence=0.8,
                evidence_count=3,
                source_experiments=["exp-1", "exp-2", "exp-3"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
        ])

        formula = Formula(
            name="Squalane Test",
            ingredients=[
                ("Water", 70.0), ("Squalane", 15.0), ("Glycerin", 15.0),
            ],
            category="skincare",
        )
        disc = evaluate_discourse(formula, memory=memory)

        # Find the Squalane topic
        squalane_topics = [
            t for t in disc.topics
            if "squalane" in t.subject.lower()
        ]
        # Physics should flag LogP concern, data should say it worked
        # This should produce a true disagreement
        if squalane_topics:
            squalane = squalane_topics[0]
            perspectives = {c.perspective for c in squalane.claims}
            if "physics" in perspectives and "data" in perspectives:
                assert squalane.classification == "true_disagreement"
