"""Data models for LLMCord.

This module contains dataclasses and types used throughout the application.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class MsgNode:
    """Represents a single message node in the conversation history.

    :param text: The text content of the message.
    :param images: A list of image content parts for OpenAI API.
    :param role: The role of the message sender ('user' or 'assistant').
    :param user_id: The Discord user ID of the sender.
    :param user_display_name: The display name of the sender.
    :param has_bad_attachments: Indicates if the message had unsupported attachments.
    :param lock: An async lock to manage concurrent access to this node.
    """

    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[assignment] # dataclasses field() has incomplete type stubs, remove when fixed upstream

    role: Literal["user", "assistant"] = "assistant"
    user_id: int | None = None
    user_display_name: str | None = None

    has_bad_attachments: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
