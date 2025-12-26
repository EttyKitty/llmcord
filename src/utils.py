"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal

import discord
from openai.types.chat import ChatCompletionContentPartImageParam

from .config import PermissionsConfig

REGEX_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
REGEX_MULTI_SPACE = re.compile(r" {2,}")
REGEX_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\r?\n|$)")

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


def is_message_allowed(msg: discord.Message, permissions: PermissionsConfig, *, allow_dms: bool) -> bool:
    """Check if the message author and channel are allowed based on configuration.

    :param msg: The Discord message to check.
    :param permissions: The permissions configuration object.
    :param allow_dms: Whether the bot is allowed to respond in Direct Messages.
    :return: True if allowed, False otherwise.
    """
    is_dm = msg.channel.type == discord.ChannelType.private

    # 1. Admin Bypass
    if msg.author.id in permissions.users.admin_ids:
        return True

    # 2. User/Role Validation
    role_ids = {role.id for role in getattr(msg.author, "roles", ())}
    allowed_users = permissions.users.allowed_ids
    blocked_users = permissions.users.blocked_ids
    allowed_roles = permissions.roles.allowed_ids
    blocked_roles = permissions.roles.blocked_ids

    # Determine if the user is "good" (allowed by default or explicitly)
    allow_all_users = not allowed_users if is_dm else (not allowed_users and not allowed_roles)
    is_good_user = allow_all_users or msg.author.id in allowed_users or not role_ids.isdisjoint(allowed_roles)

    # Determine if the user is "bad" (not good or explicitly blocked)
    is_bad_user = not is_good_user or msg.author.id in blocked_users or not role_ids.isdisjoint(blocked_roles)

    if is_bad_user:
        return False

    # 3. Channel Validation
    # Collect current channel ID, parent ID (for threads), and category ID
    channel_ids = {
        msg.channel.id,
        getattr(msg.channel, "parent_id", None),
        getattr(msg.channel, "category_id", None),
    }
    channel_ids.discard(None)  # Remove None values

    allowed_channels = permissions.channels.allowed_ids
    blocked_channels = permissions.channels.blocked_ids

    # Determine if the channel is "good"
    is_good_channel = allow_dms if is_dm else (not allowed_channels or not channel_ids.isdisjoint(allowed_channels))

    # Determine if the channel is "bad"
    is_bad_channel = not is_good_channel or not channel_ids.isdisjoint(blocked_channels)

    return not is_bad_channel
