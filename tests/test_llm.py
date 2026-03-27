"""Tests for LLM provider abstraction."""

from openmix.llm import create_provider


def test_create_anthropic_provider():
    """Should create AnthropicProvider from config."""
    # Don't actually connect — just verify config parsing
    try:
        provider = create_provider({
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
        })
        assert provider.name == "anthropic/claude-sonnet-4-20250514"
    except ImportError:
        pass  # anthropic not installed in CI


def test_create_ollama_provider():
    """Should create OllamaProvider from config."""
    try:
        provider = create_provider({
            "provider": "ollama",
            "model": "llama3.1",
            "base_url": "http://localhost:11434/v1",
        })
        assert provider.name == "ollama/llama3.1"
    except ImportError:
        pass  # openai not installed


def test_unknown_provider_raises():
    """Unknown provider should raise ValueError."""
    try:
        create_provider({"provider": "nonexistent"})
        assert False, "Should have raised"
    except (ValueError, ImportError):
        pass
