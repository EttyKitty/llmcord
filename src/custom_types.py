"""Type aliases and custom types for the LLMCord project."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import discord


@dataclass
class MessageNode:
    """Represents a single message node in the conversation history.

    :param text: The text content of the message.
    :param images: A list of image content parts for OpenAI API.
    :param role: The role of the message sender ('user' or 'assistant').
    :param user_id: The Discord user ID of the sender.
    :param user_display_name: The display name of the sender.
    :param has_bad_attachments: Indicates if the message had unsupported attachments.
    :param lock: An async lock to manage concurrent access to this node.
    """

    created_at: discord.datetime | None = None

    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[assignment] # dataclasses field() has incomplete type stubs, remove when fixed upstream

    role: Literal["user", "assistant"] = "assistant"
    user_id: int | None = None
    user_display_name: str | None = None

    has_bad_attachments: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True, slots=True)
class MessagePayloadParams:
    """Parameters for message payload creation."""

    max_text: int
    max_images: int
    prefix_users: bool
    accept_images: bool
    accept_usernames: bool


@dataclass(frozen=True, slots=True)
class BuildMessagesParams:
    """Parameters for building messages payload."""

    message_history: list[discord.Message]
    message_nodes: dict[int, MessageNode]
    pre_history: str | None
    post_history: str | None
    model: str
    provider: str


# Common type aliases for better readability and maintainability
MessageCache = dict[int, discord.Message]
MessageNodeCache = dict[int, MessageNode]
ConfigDict = dict[str, Any]
StringDict = dict[str, str]
AnyDict = dict[str, Any]
AnyList = list[Any]

# Discord-specific type aliases
MessageList = list[discord.Message]
ChannelIDList = list[int]
UserIDList = list[int]
RoleIDList = list[int]

# Configuration type aliases
ProviderConfig = dict[str, Any]
ModelConfig = dict[str, Any]

# Message processing type aliases
ImageContent = dict[str, Any]
ImageList = list[ImageContent]

# LLM service type aliases
ChatParams = dict[str, Any]
ToolDefinition = dict[str, Any]
ToolList = list[ToolDefinition]

# Embed and component type aliases
EmbedFieldList = list[str | None]
ComponentContentList = list[str]

# Message utility type aliases
MessageContent = str | list[dict[str, Any]]
MessagePartList = list[dict[str, Any]]
