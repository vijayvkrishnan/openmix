"""Tests for experiment configuration and infrastructure."""

from pathlib import Path

from openmix.experiment import ExperimentConfig


EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


def test_load_vitamin_c_config():
    """Should parse the vitamin C experiment YAML."""
    config = ExperimentConfig.from_file(EXPERIMENTS_DIR / "vitamin_c_stability.yaml")
    assert config.name == "vitamin-c-stability"
    assert len(config.required_ingredients) == 1
    assert config.required_ingredients[0]["name"] == "Ascorbic Acid"
    assert len(config.available_ingredients) > 10
    assert config.max_iterations == 30
    assert config.target_score == 95.0


def test_load_multivitamin_config():
    """Should parse the multivitamin experiment YAML."""
    config = ExperimentConfig.from_file(EXPERIMENTS_DIR / "multivitamin_gummy.yaml")
    assert config.name == "multivitamin-gummy"
    assert len(config.required_ingredients) == 2
    assert config.constraints.get("max_ingredients") == 18
    assert config.mode == "formulation"


def test_load_mrna_config():
    """Should parse the mRNA LNP experiment YAML."""
    config = ExperimentConfig.from_file(EXPERIMENTS_DIR / "mrna_lipid_nanoparticle.yaml")
    assert config.name == "mrna-lnp-stability"
    assert config.mode == "discovery"
    assert config.llm_config is not None
    assert config.llm_config["provider"] == "anthropic"


def test_config_has_llm_section():
    """Experiment configs should support LLM provider configuration."""
    config = ExperimentConfig.from_file(EXPERIMENTS_DIR / "vitamin_c_stability.yaml")
    assert config.llm_config is not None
    assert "provider" in config.llm_config
    assert "model" in config.llm_config
