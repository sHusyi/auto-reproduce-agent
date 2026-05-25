"""LLM configuration — provider-agnostic, config-driven."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for an LLM provider.

    Loads from environment variables by default, but every field can be
    overridden programmatically.

    Usage:
        # From env (default)
        config = LLMConfig.from_env()

        # Explicit
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com/v1",
        )
    """

    provider: str  # "deepseek", "openai", "anthropic"
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

    @classmethod
    def from_env(cls, provider: str = "deepseek") -> "LLMConfig":
        """Create config from environment variables for the given provider."""
        if provider == "deepseek":
            return cls(
                provider="deepseek",
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            )
        elif provider == "openai":
            return cls(
                provider="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )
        elif provider == "anthropic":
            return cls(
                provider="anthropic",
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'deepseek', 'openai', or 'anthropic'.")

    @classmethod
    def from_dotenv(cls, dotenv_path: str | Path = ".env", provider: str = "deepseek") -> "LLMConfig":
        """Load .env file then create config from environment."""
        dotenv_path = Path(dotenv_path)
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = value
        return cls.from_env(provider)
