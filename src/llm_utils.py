"""LLM utilities for provider and model handling.

This module contains utility functions for resolving LLM providers,
building chat parameters, and other LLM-related operations.
"""

from collections.abc import Sequence
from typing import Any, cast

from .config_manager import ConfigValue


def get_llm_provider_model(channel_id: int, channel_models: dict[int, str], default_model: str) -> tuple[str, str]:
    """Resolve the LLM provider and model for a specific channel.

    :param channel_id: The Discord channel ID that triggered the request.
    :param channel_models: A dictionary mapping channel IDs to provider/model strings.
    :param default_model: The default provider/model string to use as a fallback.
    :return: A tuple containing (provider name, model name).
    :raises ValueError: If the provider/model string does not contain a '/' separator.
    """
    provider_slash_model = channel_models.get(channel_id, default_model)

    if "/" not in provider_slash_model:
        error = f"Invalid model format: '{provider_slash_model}'. Expected 'provider/model'."
        raise ValueError(error)

    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    return provider, model


def build_chat_params(
    model: str,
    messages: Sequence[dict[str, Any]],
    provider: str,
    provider_config: ConfigValue,
    llm_models_config: dict[str, ConfigValue],
) -> dict[str, Any]:
    """Construct the parameters for the Litellm completion call.

    :param model: The model identifier.
    :param messages: The list of message payloads.
    :param provider: The provider identifier.
    :param provider_config: The configuration dictionary for the provider.
    :param llm_models_config: The dictionary of model-specific overrides.
    :return: A dictionary of parameters for the API call.
    :raises TypeError: If provider_config is not a dictionary.
    """
    if not isinstance(provider_config, dict):
        error = f"Provider config must be a dict, got {type(provider_config)}"
        raise TypeError(error)

    raw_overrides = llm_models_config.get(f"{provider}/{model}")
    model_overrides: dict[str, object] = raw_overrides if isinstance(raw_overrides, dict) else {}

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    raw_extra_body = provider_config.get("extra_body")
    extra_body_base: dict[str, Any] = cast("dict[str, Any]", raw_extra_body) if isinstance(raw_extra_body, dict) else {}

    # Merge provider-level extra_body with model-specific overrides
    extra_body: dict[str, Any] = extra_body_base | model_overrides

    return {
        "model": f"{provider}/{model}",
        "messages": messages,
        "api_key": provider_config.get("api_key"),
        "api_base": provider_config.get("base_url"),
        "stream": False,
        "extra_headers": extra_headers if isinstance(extra_headers, dict) else None,
        "extra_query": extra_query if isinstance(extra_query, dict) else None,
        "extra_body": extra_body,
    }
