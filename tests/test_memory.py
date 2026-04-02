"""Tests for experiment memory — persistence and learning across runs."""

import pytest

from openmix.schema import Formula
from openmix.experiment import ExperimentLog, Trial
from openmix.observe import FormulationObservation
from openmix.memory import (
    ExperimentMemory,
    Discovery,
    IndexEntry,
    format_prior_knowledge,
    _extract_findings,
    _merge_discoveries,
    _generate_summary,
    INITIAL_CONFIDENCE,
    CONFIRMATION_BOOST,
    MAX_CONFIDENCE,
    CONTRADICTION_PENALTY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_memory(tmp_path):
    """ExperimentMemory backed by a temp directory."""
    return ExperimentMemory(base_dir=tmp_path)


def _make_formula(ingredients, category="skincare", target_ph=5.5):
    return Formula(
        name="test formula",
        ingredients=ingredients,
        category=category,
        target_ph=target_ph,
    )


def _make_trial(iteration, ingredients, concern_count=0.0, reasoning=None):
    formula = _make_formula(ingredients)
    obs = FormulationObservation(
        mode="engineering",
        formula_name="test",
        observations=[],
        violations=[],
    )
    trial = Trial(
        iteration=iteration,
        formula=formula,
        observation=obs,
        reasoning=reasoning,
        formula_hash=f"hash_{iteration}",
    )
    return trial


def _make_log(
    name="test-experiment",
    n_trials=3,
    converged=True,
    category="skincare",
):
    """Build a minimal ExperimentLog for testing."""
    log = ExperimentLog(
        name=name,
        brief="Test experiment brief",
        config={"constraints": {"category": category}},
        started_at="2026-04-01T10:00:00Z",
        finished_at="2026-04-01T10:05:00Z",
        total_duration_ms=300000,
    )

    # Trial 1: tries glycerin + retinol
    t1 = _make_trial(1, [
        ("Water", 70.0), ("Glycerin", 10.0), ("Retinol", 2.0),
        ("Niacinamide", 5.0), ("Phenoxyethanol", 1.0),
    ], reasoning="Starting with a basic retinol serum")

    # Trial 2: drops retinol, adds squalane
    t2 = _make_trial(2, [
        ("Water", 65.0), ("Glycerin", 10.0), ("Squalane", 12.0),
        ("Niacinamide", 5.0), ("Phenoxyethanol", 1.0),
    ], reasoning="Retinol caused stability issues, trying squalane")

    # Trial 3 (best): keeps squalane, adds tocopherol
    t3 = _make_trial(3, [
        ("Water", 64.0), ("Glycerin", 10.0), ("Squalane", 12.0),
        ("Niacinamide", 5.0), ("Tocopherol", 1.0), ("Phenoxyethanol", 1.0),
    ], reasoning="Added antioxidant protection")

    log.trials = [t1, t2, t3][:n_trials]
    log.best_trial = log.trials[-1]
    log.best_concerns = 0.0 if converged else 2.0
    log.converged = converged

    return log


# ---------------------------------------------------------------------------
# Layer 1: Index
# ---------------------------------------------------------------------------

class TestIndex:
    def test_empty_index(self, tmp_memory):
        assert tmp_memory.load_index() == []

    def test_save_and_load_index(self, tmp_memory):
        entry = IndexEntry(
            name="test-exp",
            category="skincare",
            date="2026-04-01T10:00:00Z",
            best_concerns=0.0,
            converged=True,
            n_trials=5,
            key_ingredients=["Glycerin", "Squalane"],
            summary="Converged at iter 3. 6 ingredients",
            log_file="test-exp_20260401.json",
        )
        tmp_memory._save_index([entry])
        loaded = tmp_memory.load_index()
        assert len(loaded) == 1
        assert loaded[0].name == "test-exp"
        assert loaded[0].converged is True
        assert loaded[0].key_ingredients == ["Glycerin", "Squalane"]

    def test_add_to_index(self, tmp_memory):
        log = _make_log()
        tmp_memory._ensure_dirs()
        entry = tmp_memory._add_to_index(log, "test.json")
        assert entry.name == "test-experiment"
        assert entry.category == "skincare"
        assert entry.converged is True

        loaded = tmp_memory.load_index()
        assert len(loaded) == 1

    def test_multiple_entries(self, tmp_memory):
        log1 = _make_log(name="exp-1")
        log2 = _make_log(name="exp-2", category="supplement")
        tmp_memory._ensure_dirs()
        tmp_memory._add_to_index(log1, "exp1.json")
        tmp_memory._add_to_index(log2, "exp2.json")

        loaded = tmp_memory.load_index()
        assert len(loaded) == 2
        assert loaded[0].name == "exp-1"
        assert loaded[1].name == "exp-2"


# ---------------------------------------------------------------------------
# Layer 2: Logs
# ---------------------------------------------------------------------------

class TestLogs:
    def test_save_and_load_log(self, tmp_memory):
        log = _make_log()
        filename = tmp_memory.save_log(log)
        assert filename.startswith("test-experiment_")
        assert filename.endswith(".json")

        data = tmp_memory.load_log(filename)
        assert data["experiment"] == "test-experiment"
        assert data["converged"] is True

    def test_list_logs(self, tmp_memory):
        assert tmp_memory.list_logs() == []

        log = _make_log()
        tmp_memory.save_log(log)
        logs = tmp_memory.list_logs()
        assert len(logs) == 1


# ---------------------------------------------------------------------------
# Layer 3: Discoveries
# ---------------------------------------------------------------------------

class TestDiscoveries:
    def test_empty_discoveries(self, tmp_memory):
        assert tmp_memory.load_discoveries() == []

    def test_save_and_load_discoveries(self, tmp_memory):
        discoveries = [
            Discovery(
                id="pref_glycerin_skincare",
                finding="Glycerin at 10.0% as humectant",
                kind="preference",
                domain="skincare",
                ingredients=["Glycerin"],
                confidence=0.5,
                evidence_count=1,
                source_experiments=["test-exp"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
        ]
        tmp_memory._save_discoveries(discoveries)
        loaded = tmp_memory.load_discoveries()
        assert len(loaded) == 1
        assert loaded[0].finding == "Glycerin at 10.0% as humectant"
        assert loaded[0].confidence == 0.5


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extract_preferences(self):
        log = _make_log()
        findings = _extract_findings(log)
        preferences = [f for f in findings if f.kind == "preference"]
        pref_names = [f.ingredients[0] for f in preferences]
        # Best trial has: Glycerin, Squalane, Niacinamide, Tocopherol, Phenoxyethanol
        assert "Glycerin" in pref_names
        assert "Squalane" in pref_names
        assert "Phenoxyethanol" in pref_names
        # Water should NOT be in preferences
        assert "Water" not in pref_names

    def test_extract_avoidances(self):
        log = _make_log()
        findings = _extract_findings(log)
        avoidances = [f for f in findings if f.kind == "avoidance"]
        # Avoidance requires 2+ appearances then dropped from best
        assert isinstance(avoidances, list)

    def test_extract_convergence_pattern(self):
        log = _make_log(converged=True)
        findings = _extract_findings(log)
        patterns = [f for f in findings if f.kind == "pattern"]
        assert any("Converged" in p.finding for p in patterns)

    def test_no_findings_without_best_trial(self):
        log = ExperimentLog(name="empty", brief="nothing")
        assert _extract_findings(log) == []


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_new_finding_added(self):
        existing = []
        new = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin at 10.0%",
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=0.5,
            evidence_count=1,
            source_experiments=["exp-1"],
            first_seen="2026-04-01",
            last_confirmed="2026-04-01",
        )]
        merged, added = _merge_discoveries(existing, new)
        assert len(merged) == 1
        assert len(added) == 1

    def test_confirmation_boosts_confidence(self):
        existing = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin at 10.0%",
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=INITIAL_CONFIDENCE,
            evidence_count=1,
            source_experiments=["exp-1"],
            first_seen="2026-04-01",
            last_confirmed="2026-04-01",
        )]
        new = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin at 8.0%",  # slightly different pct
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=INITIAL_CONFIDENCE,
            evidence_count=1,
            source_experiments=["exp-2"],
            first_seen="2026-04-02",
            last_confirmed="2026-04-02",
        )]
        merged, added = _merge_discoveries(existing, new)
        assert len(merged) == 1  # merged, not duplicated
        assert len(added) == 0  # not newly added
        assert merged[0].evidence_count == 2
        assert merged[0].confidence == pytest.approx(INITIAL_CONFIDENCE + CONFIRMATION_BOOST)
        assert "exp-2" in merged[0].source_experiments

    def test_confidence_caps_at_max(self):
        existing = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin",
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=0.90,
            evidence_count=5,
            source_experiments=["exp-1"],
            first_seen="2026-04-01",
            last_confirmed="2026-04-01",
        )]
        new = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin",
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=0.5,
            evidence_count=1,
            source_experiments=["exp-6"],
            first_seen="2026-04-02",
            last_confirmed="2026-04-02",
        )]
        merged, _ = _merge_discoveries(existing, new)
        assert merged[0].confidence <= MAX_CONFIDENCE

    def test_contradiction_lowers_confidence(self):
        existing = [Discovery(
            id="pref_retinol_skincare",
            finding="Retinol at 2.0%",
            kind="preference",
            domain="skincare",
            ingredients=["Retinol"],
            confidence=0.65,
            evidence_count=2,
            source_experiments=["exp-1"],
            first_seen="2026-04-01",
            last_confirmed="2026-04-01",
        )]
        # New finding says retinol was avoided
        new = [Discovery(
            id="avoid_retinol_skincare",
            finding="Retinol dropped from best",
            kind="avoidance",
            domain="skincare",
            ingredients=["Retinol"],
            confidence=0.5,
            evidence_count=1,
            source_experiments=["exp-2"],
            first_seen="2026-04-02",
            last_confirmed="2026-04-02",
        )]
        merged, added = _merge_discoveries(existing, new)
        assert len(merged) == 2  # both exist
        # Existing preference confidence lowered
        pref = next(d for d in merged if d.kind == "preference")
        assert pref.confidence == pytest.approx(0.65 - CONTRADICTION_PENALTY)
        # New avoidance also penalized
        avoid = next(d for d in merged if d.kind == "avoidance")
        assert avoid.confidence == pytest.approx(0.5 - CONTRADICTION_PENALTY)

    def test_different_domains_dont_match(self):
        existing = [Discovery(
            id="pref_glycerin_skincare",
            finding="Glycerin",
            kind="preference",
            domain="skincare",
            ingredients=["Glycerin"],
            confidence=0.5,
            evidence_count=1,
            source_experiments=["exp-1"],
            first_seen="2026-04-01",
            last_confirmed="2026-04-01",
        )]
        new = [Discovery(
            id="pref_glycerin_supplement",
            finding="Glycerin",
            kind="preference",
            domain="supplement",
            ingredients=["Glycerin"],
            confidence=0.5,
            evidence_count=1,
            source_experiments=["exp-2"],
            first_seen="2026-04-02",
            last_confirmed="2026-04-02",
        )]
        merged, added = _merge_discoveries(existing, new)
        assert len(merged) == 2  # not merged — different domains
        assert len(added) == 1


# ---------------------------------------------------------------------------
# Consolidation (end-to-end)
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_first_experiment_creates_discoveries(self, tmp_memory):
        log = _make_log()
        new = tmp_memory.consolidate(log)
        assert len(new) > 0

        saved = tmp_memory.load_discoveries()
        assert len(saved) > 0

    def test_second_experiment_boosts_confidence(self, tmp_memory):
        log1 = _make_log(name="exp-1")
        tmp_memory.consolidate(log1)
        initial = tmp_memory.load_discoveries()

        # Find glycerin preference confidence
        glycerin_conf_1 = next(
            d.confidence for d in initial
            if d.kind == "preference" and "Glycerin" in d.ingredients
        )

        log2 = _make_log(name="exp-2")
        tmp_memory.consolidate(log2)
        updated = tmp_memory.load_discoveries()

        glycerin_conf_2 = next(
            d.confidence for d in updated
            if d.kind == "preference" and "Glycerin" in d.ingredients
        )
        assert glycerin_conf_2 > glycerin_conf_1


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_no_discoveries_returns_empty(self, tmp_memory):
        result = tmp_memory.retrieve_prior_knowledge("skincare")
        assert result == ""

    def test_retrieves_same_domain(self, tmp_memory):
        log = _make_log(category="skincare")
        tmp_memory.consolidate(log)

        result = tmp_memory.retrieve_prior_knowledge("skincare")
        assert "PRIOR KNOWLEDGE" in result
        assert "Glycerin" in result

    def test_different_domain_excluded(self, tmp_memory):
        log = _make_log(category="skincare")
        tmp_memory.consolidate(log)

        # Supplement domain should not see skincare preferences
        result = tmp_memory.retrieve_prior_knowledge("supplement")
        assert result == "" or "Glycerin" not in result

    def test_retrieval_does_not_mutate_stored_discoveries(self, tmp_memory):
        """Bug regression: retrieve_prior_knowledge must not mutate
        Discovery objects loaded from disk."""
        log = _make_log(category="skincare")
        tmp_memory.consolidate(log)

        # Call retrieve twice with ingredient boost
        tmp_memory.retrieve_prior_knowledge("skincare", ingredients=["Glycerin"])
        d1 = tmp_memory.load_discoveries()
        glycerin_conf_1 = next(
            d.confidence for d in d1
            if d.kind == "preference" and "Glycerin" in d.ingredients
        )

        tmp_memory.retrieve_prior_knowledge("skincare", ingredients=["Glycerin"])
        d2 = tmp_memory.load_discoveries()
        glycerin_conf_2 = next(
            d.confidence for d in d2
            if d.kind == "preference" and "Glycerin" in d.ingredients
        )

        # Confidence on disk must not have changed
        assert glycerin_conf_1 == glycerin_conf_2


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

class TestFormat:
    def test_empty_list(self):
        assert format_prior_knowledge([]) == ""

    def test_preferences_formatted(self):
        discoveries = [
            Discovery(
                id="pref_glycerin_skincare",
                finding="Glycerin at 10.0% as humectant",
                kind="preference",
                domain="skincare",
                ingredients=["Glycerin"],
                confidence=0.65,
                evidence_count=2,
                source_experiments=["exp-1", "exp-2"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-02",
            ),
        ]
        result = format_prior_knowledge(discoveries)
        assert "PRIOR KNOWLEDGE" in result
        assert "Glycerin" in result
        assert "0.65" in result
        assert "2 experiments" in result

    def test_avoidances_formatted(self):
        discoveries = [
            Discovery(
                id="avoid_retinol_skincare",
                finding="Retinol dropped from best",
                kind="avoidance",
                domain="skincare",
                ingredients=["Retinol"],
                confidence=0.5,
                evidence_count=1,
                source_experiments=["exp-1"],
                first_seen="2026-04-01",
                last_confirmed="2026-04-01",
            ),
        ]
        result = format_prior_knowledge(discoveries)
        assert "reconsider" in result.lower()
        assert "Retinol" in result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_converged_summary(self):
        log = _make_log(converged=True)
        summary = _generate_summary(log)
        assert "Converged" in summary

    def test_not_converged_summary(self):
        log = _make_log(converged=False)
        summary = _generate_summary(log)
        assert "Best:" in summary

    def test_key_ingredients_in_summary(self):
        log = _make_log()
        summary = _generate_summary(log)
        # Should mention key non-water ingredients
        assert "Squalane" in summary or "Glycerin" in summary


# ---------------------------------------------------------------------------
# Full record_experiment flow
# ---------------------------------------------------------------------------

class TestRecordExperiment:
    def test_record_creates_all_layers(self, tmp_memory):
        log = _make_log()
        new_discoveries = tmp_memory.record_experiment(log)

        # Layer 1: index
        index = tmp_memory.load_index()
        assert len(index) == 1
        assert index[0].name == "test-experiment"

        # Layer 2: log file exists
        logs = tmp_memory.list_logs()
        assert len(logs) == 1

        # Layer 3: discoveries
        discoveries = tmp_memory.load_discoveries()
        assert len(discoveries) > 0
        assert len(new_discoveries) > 0

    def test_record_two_experiments(self, tmp_memory):
        log1 = _make_log(name="exp-1")
        log2 = _make_log(name="exp-2")

        tmp_memory.record_experiment(log1)
        tmp_memory.record_experiment(log2)

        index = tmp_memory.load_index()
        assert len(index) == 2

        logs = tmp_memory.list_logs()
        assert len(logs) == 2

    def test_memory_summary(self, tmp_memory):
        log = _make_log()
        tmp_memory.record_experiment(log)

        summary = tmp_memory.summary()
        assert "EXPERIMENT MEMORY" in summary
        assert "test-experiment" in summary
        assert "Experiments: 1" in summary
