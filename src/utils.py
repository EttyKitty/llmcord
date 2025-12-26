"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import discord
import httpx
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionMessageParam,
)

from .config import ConfigValue, config_manager

REGEX_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
REGEX_MULTI_SPACE = re.compile(r" {2,}")
REGEX_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\r?\n|$)")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
TYPOGRAPHY_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",  # noqa: RUF001
        "’": "'",  # noqa: RUF001
        "—": "-",
        "…": "...",
    },
)

logger = logging.getLogger(__name__)


@dataclass
class MsgNode:
    """Represents a single message node in the conversation history.

    :param text: The text content of the message.
    :param images: A list of image content parts for OpenAI API.
    :param role: The role of the message sender ('user' or 'assistant').
    :param user_id: The Discord user ID of the sender.
    :param user_display_name: The display name of the sender.
    :param has_bad_attachments: Indicates if the message had unsupported attachments.
    :param fetch_parent_failed: Indicates if the parent message could not be retrieved.
    :param parent_msg: The Discord message object of the parent node.
    :param lock: An async lock to manage concurrent access to this node.
    """

    text: str | None = None
    images: list[ChatCompletionContentPartImageParam] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: int | None = None
    user_display_name: str | None = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: discord.Message | None = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def clean_response(text: str) -> str:
    """Sanitizes the response text by normalizing typography and whitespace.

    This function replaces typographic quotes/dashes, reduces excessive
    newlines and spaces, and trims trailing whitespace.

    :param text: The raw text string to clean.
    :return: The cleaned and normalized string.
    """
    text = text.translate(TYPOGRAPHY_MAP)
    text = REGEX_MULTI_SPACE.sub(" ", text)
    text = REGEX_EXCESSIVE_NEWLINES.sub("\n\n", text)
    text = REGEX_TRAILING_WHITESPACE.sub("", text)
    return text.strip()


def get_llm_specials(provider: str, model: str) -> tuple[bool, bool]:
    """Resolve the LLM flags based on configuration.

    :param provider: The LLM provider identifier.
    :param model: The model identifier.
    :return: A tuple containing accept_images flag, and accept_usernames flag.
    """
    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    return accept_images, accept_usernames


def extract_chunk_content(chunk: ChatCompletionChunk) -> str:
    """Extract text content from a streaming chunk.

    Handles standard strings and multimodal list/object structures.

    :param chunk: The chunk object from the API response.
    :return: The extracted text content.
    """
    if not chunk.choices:
        return ""

    choice = chunk.choices[0]
    delta_content = choice.delta.content or ""

    # Handle potential list content (Mistral/Multimodal quirks)
    if isinstance(delta_content, list):
        logger.debug("Multimodal content detected...")
        parts_text = ""
        for part in delta_content:
            if isinstance(part, str):
                parts_text += part
            elif isinstance(part, dict):
                parts_text += part.get("text", "")
            elif hasattr(part, "text"):
                parts_text += part.text
        return parts_text

    return str(delta_content)


def update_content_buffer(buffer: list[str], new_content: str, max_len: int) -> None:
    """Append content to the buffer, creating new chunks if max length is exceeded.

    :param buffer: The list of message content chunks.
    :param new_content: The new text to append.
    :param max_len: The maximum length of a single message.
    """
    if not buffer or len(buffer[-1]) + len(new_content) > max_len:
        buffer.append("")
    buffer[-1] += new_content


def process_response_text(text: str, *, sanitize: bool) -> str:
    """Sanitize and process the final response text.

    :param text: The raw response text.
    :param sanitize: Whether to apply the clean_response utility.
    :return: The processed text.
    """
    final_text = text
    if sanitize:
        logger.debug("Sanitizing text...")
        final_text = clean_response(final_text)

    if "<think>" in final_text:
        logger.debug("Removing <think> block...")
        final_text = REGEX_THINK_BLOCK.sub("", final_text).strip()

    return final_text


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


def get_openai_client(
    provider_config: ConfigValue,
    client_cache: dict[str, AsyncOpenAI],
    httpx_client: httpx.AsyncClient,
) -> AsyncOpenAI:
    """Retrieve or initialize an OpenAI-compatible client for a specific provider.

    :param provider_config: The configuration dictionary for the provider.
    :param client_cache: The dictionary storing existing AsyncOpenAI instances.
    :param httpx_client: The shared httpx.AsyncClient to use for requests.
    :return: An AsyncOpenAI client instance.
    :raises TypeError: If provider_config is not a dictionary.
    """
    if not isinstance(provider_config, dict):
        error = f"Provider config must be a dict, got {type(provider_config)}"
        raise TypeError(error)

    base_url = provider_config["base_url"]
    if base_url not in client_cache:
        client_cache[base_url] = AsyncOpenAI(
            base_url=base_url,
            api_key=provider_config.get("api_key", "sk-no-key-required"),
            http_client=httpx_client,
        )

    return client_cache[base_url]


def get_provider_config(
    provider: str,
    providers_config: dict[str, ConfigValue],
    client_cache: dict[str, AsyncOpenAI],
    httpx_client: httpx.AsyncClient,
) -> tuple[ConfigValue, AsyncOpenAI]:
    """Resolve the provider configuration and its associated OpenAI client.

    :param provider: The name of the provider to resolve.
    :param providers_config: The dictionary of all provider configurations.
    :param client_cache: The dictionary storing existing AsyncOpenAI instances.
    :param httpx_client: The shared httpx.AsyncClient to use for requests.
    :raises KeyError: If the provider is not found in providers_config.
    """
    if provider not in providers_config:
        error = f"Provider '{provider}' not found in configuration."
        raise KeyError(error)

    provider_config = providers_config[provider]
    openai_client = get_openai_client(provider_config, client_cache, httpx_client)

    return provider_config, openai_client


def build_chat_params(
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    provider: str,
    provider_config: ConfigValue,
    llm_models_config: dict[str, ConfigValue],
) -> dict[str, Any]:
    """Construct the parameters for the OpenAI chat completion call.

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
    extra_body_base = raw_extra_body if isinstance(raw_extra_body, dict) else {}

    # Merge provider-level extra_body with model-specific overrides
    extra_body: dict[str, object] = extra_body_base | model_overrides

    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "extra_headers": extra_headers if isinstance(extra_headers, dict) else None,
        "extra_query": extra_query if isinstance(extra_query, dict) else None,
        "extra_body": extra_body,
    }


def is_admin(user_id: int) -> bool:
    """Check if a user has admin permissions.

    :param user_id: The Discord user ID to check.
    :return: True if the user is an admin, False otherwise.
    """
    return user_id in config_manager.config.discord.permissions.users.admin_ids
