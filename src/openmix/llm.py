"""
LLM provider abstraction. Bring your own model.

Supports:
  - Anthropic (Claude)
  - OpenAI (GPT)
  - Any OpenAI-compatible API (Ollama, Together, Groq, vLLM, LM Studio)

Configure in experiment YAML:
  llm:
    provider: anthropic          # or: openai, ollama, custom
    model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY   # env var name (never hardcode keys)

Or pass at runtime:
  exp = Experiment.from_file("exp.yaml", llm=OpenAIProvider(model="gpt-4o"))
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, system: str, messages: list[dict]) -> str:
        """Send messages and return the text response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def generate(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI API and any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def generate(self, system: str, messages: list[dict]) -> str:
        msgs = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=msgs,
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama local models via OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        max_tokens: int = 4096,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def generate(self, system: str, messages: list[dict]) -> str:
        msgs = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=msgs,
        )
        return response.choices[0].message.content


def create_provider(config: dict) -> LLMProvider:
    """
    Create an LLM provider from a config dict.

    Config format (from experiment YAML):
        provider: anthropic
        model: claude-sonnet-4-20250514
        api_key_env: ANTHROPIC_API_KEY
        base_url: null
    """
    provider_name = config.get("provider", "anthropic").lower()
    model = config.get("model", "")
    api_key_env = config.get("api_key_env", "")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    base_url = config.get("base_url")

    if provider_name == "anthropic":
        return AnthropicProvider(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
        )
    elif provider_name in ("openai", "together", "groq"):
        return OpenAIProvider(
            model=model or "gpt-4o",
            api_key=api_key,
            base_url=base_url,
        )
    elif provider_name == "ollama":
        return OllamaProvider(
            model=model or "llama3.1",
            base_url=base_url or "http://localhost:11434/v1",
        )
    elif provider_name == "custom":
        return OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported: anthropic, openai, ollama, together, groq, custom"
        )
