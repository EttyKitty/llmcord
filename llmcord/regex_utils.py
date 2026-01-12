"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import re
from datetime import datetime, timezone
from functools import lru_cache

import discord
from loguru import logger

from .logging_utils_ import time_performance

REGEX_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
REGEX_MULTI_SPACE = re.compile(r" {2,}")
REGEX_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\r?\n|$)")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
REGEX_USER_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
TYPOGRAPHY_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",  # noqa: RUF001
        "’": "'",  # noqa: RUF001
        "—": " - ",
        "…": "...",
    },
)
REGEX_BRACKETED_PREFIX = re.compile(r"^\[.*?\][\s:]+", re.IGNORECASE)
REGEX_GENERIC_ASSISTANT_PREFIX = re.compile(r"^(Assistant|AI|System)[\s:]+", re.IGNORECASE)


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


@time_performance("Response cleaning")
def process_response_text(text: str, *, sanitize: bool, bot_name: str) -> str:
    """Sanitize and process the final response text.

    :param text: The raw response text.
    :param sanitize: Whether to apply the clean_response utility.
    :param bot_name: The bot's display name for prefix removal.
    :return: The processed text.
    """
    final_text = text

    if sanitize:
        logger.debug("Sanitizing text...")
        final_text = clean_response(final_text)

    if "<think>" in final_text:
        logger.debug("Removing <think> block...")
        final_text = REGEX_THINK_BLOCK.sub("", final_text).strip()

    return remove_prefixes(final_text, bot_name)


def replace_placeholders(text: str, msg: discord.Message, bot_user: discord.ClientUser, model: str, provider: str) -> str:
    """Replace dynamic placeholders in prompt strings.

    :param text: The text containing placeholders to replace.
    :param msg: The Discord message providing context.
    :param bot_user: The bot's client user object.
    :param model: The LLM model identifier.
    :param provider: The LLM provider identifier.
    :return: The text with all placeholders replaced.
    """
    now = datetime.now(timezone.utc)
    user_roles = getattr(msg.author, "roles", [])
    user_roles_str = ", ".join([role.name for role in user_roles if role.name != "@everyone"]) or ""
    guild_emojis = getattr(msg.guild, "emojis", [])
    guild_emojis_str = ", ".join([str(emoji) for emoji in guild_emojis]) or ""

    placeholders: dict[str, str] = {
        "{date}": now.strftime("%B %d %Y"),
        "{time}": now.strftime("%H:%M:%S %Z%z"),
        "{bot_name}": bot_user.display_name,
        "{bot_id}": str(bot_user.id),
        "{model}": model,
        "{provider}": provider,
        "{user_display_name}": msg.author.display_name,
        "{user_id}": str(msg.author.id),
        "{user_roles}": user_roles_str,
        "{guild_name}": msg.guild.name if msg.guild else "Direct Messages",
        "{guild_description}": msg.guild.description or "" if msg.guild else "",
        "{guild_emojis}": guild_emojis_str,
        "{channel_name}": getattr(msg.channel, "name", ""),
        "{channel_topic}": getattr(msg.channel, "topic", ""),
        "{channel_nsfw}": str(getattr(msg.channel, "nsfw", False)),
    }

    return text.format_map(placeholders).strip()


def sanitize_symbols(username: str) -> str:
    """Remove non-alphanumeric characters from username, keeping underscores and hyphens.

    :param username: The username to sanitize.
    :return: The sanitized username containing only [a-zA-Z0-9_-].
    """
    return REGEX_USER_NAME_SANITIZER.sub("", username)


@lru_cache(maxsize=128)
def _get_prefix_patterns(bot_name: str) -> list[re.Pattern[str]]:
    """Get compiled regex patterns for the given bot name."""
    name_esc = re.escape(bot_name)
    return [
        re.compile(rf"^\[.*?\]\s*{name_esc}(?:\(\d+\))?[\s:]+", re.IGNORECASE),
        re.compile(rf"^{name_esc}(?:\(\d+\))?[\s:]+", re.IGNORECASE),
        re.compile(rf"^{name_esc}[\s:]+", re.IGNORECASE),
        REGEX_BRACKETED_PREFIX,
        REGEX_GENERIC_ASSISTANT_PREFIX,
    ]


def remove_prefixes(text: str, bot_name: str) -> str:
    """Remove common prefixes from the start of the LLM response.

    Strips headers like timestamps, bot name identifiers, and generic
    assistant prefixes that some LLMs prepend to responses.

    :param text: The response text to clean.
    :param bot_name: The bot's display name to match in prefixes.
    :return: The text with leading prefixes removed.
    """
    text = text.strip()
    patterns = _get_prefix_patterns(bot_name)

    for pattern in patterns:
        text = pattern.sub("", text).strip()

    return text
