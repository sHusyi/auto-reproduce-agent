"""LLM Factory — creates ChatModel instances from config.

Supports DeepSeek, OpenAI, and Anthropic through a single interface.
New providers are added by extending the factory, not modifying callers.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.llm.config import LLMConfig


class LLMFactory:
    """Creates LangChain-compatible chat models from config.

    Usage:
        config = LLMConfig.from_env("deepseek")
        llm = LLMFactory.create(config)
        response = llm.invoke("Hello")
    """

    @staticmethod
    def create(config: LLMConfig, callbacks: list | None = None) -> BaseChatModel:
        """Create a chat model from the given config.

        Args:
            config: LLM configuration.
            callbacks: Optional list of LangChain callback handlers (e.g., MetricsCallback).
        """
        if config.provider == "deepseek":
            llm = LLMFactory._create_openai_compatible(config)
        elif config.provider == "openai":
            llm = LLMFactory._create_openai(config)
        elif config.provider == "anthropic":
            llm = LLMFactory._create_anthropic(config)
        else:
            raise ValueError(
                f"Unknown provider: {config.provider}. "
                f"Supported: deepseek, openai, anthropic."
            )

        if callbacks:
            llm = llm.with_config({"callbacks": callbacks})
        return llm

    @staticmethod
    def _create_openai_compatible(config: LLMConfig) -> BaseChatModel:
        """Create a ChatOpenAI pointed at a compatible API (e.g. DeepSeek)."""
        from langchain_openai import ChatOpenAI

        kwargs: dict = dict(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url

        return ChatOpenAI(**kwargs)

    @staticmethod
    def _create_openai(config: LLMConfig) -> BaseChatModel:
        """Create a standard OpenAI Chat model."""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def _create_anthropic(config: LLMConfig) -> BaseChatModel:
        """Create an Anthropic Chat model."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
