"""Module for handling message processing and payload construction.

This module provides functionality for constructing LLM payloads from Discord messages,
handling attachments, and managing message nodes for conversation history.
"""

import asyncio
import json
from base64 import b64encode
from dataclasses import dataclass, field
from typing import Any, Literal

import discord
import httpx
import litellm
from discord import Message
from litellm import utils as litellm_utils
from loguru import logger

from .config_manager import RootConfig, config_manager
from .discord_utils import fetch_history
from .llm_tools import get_tool_definitions
from .regex_utils import replace_placeholders, sanitize_symbols

# Constants
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
MAX_MESSAGE_NODES = 500


@dataclass(frozen=True, slots=True)
class MessagePayloadParams:
    """Parameters for message payload creation."""

    max_text: int
    max_images: int
    prefix_users: bool
    accept_images: bool
    accept_usernames: bool


@dataclass
class MessageNode:
    """Represents a single message node in the conversation history.

    :param created_at:
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
    images: list[dict[str, str | dict[str, str]]] | None = None
    role: Literal["user", "assistant"] | None = None
    user_id: int | None = None
    user_display_name: str | None = None
    has_bad_attachments: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


MessageNodeCache = dict[int, MessageNode]


class MessageService:
    """Manage message processing and LLM payload construction.

    This class handles the conversion of Discord messages into structured LLM payloads,
    manages message nodes for conversation history, and processes attachments.
    """

    @property
    def config(self) -> RootConfig:
        """Get the current configuration.

        :return: The root configuration object.
        """
        return config_manager.config

    def __init__(self, user: discord.ClientUser, httpx_client: httpx.AsyncClient) -> None:
        """Initialize the MessageService with configuration and dependencies.

        :param config: Root configuration object.
        :param user: The bot's Discord user object.
        :param httpx_client: HTTP client for downloading attachments.
        """
        self.user = user
        self.httpx_client = httpx_client
        self.message_nodes: MessageNodeCache = {}

    def prune_msg_nodes(self) -> None:
        """Prune message nodes to maintain maximum capacity.

        Efficiently remove oldest message nodes using dict insertion order (Python 3.7+).
        """
        while len(self.message_nodes) > MAX_MESSAGE_NODES:
            # Pop the oldest item (first inserted)
            self.message_nodes.pop(next(iter(self.message_nodes)))

    async def _ensure_node(self, msg: discord.Message) -> None:
        node = self.message_nodes.setdefault(msg.id, MessageNode())
        if node.text is not None:
            return

        async with node.lock:
            if node.text is not None:  # Double-check pattern
                return

            # 1. Basic Metadata
            node.text = self._get_text_content(msg)
            node.role = "assistant" if msg.author == self.user else "user"
            node.user_id = msg.author.id
            node.created_at = msg.created_at
            node.user_display_name = msg.author.display_name if not msg.guild else (msg.guild.get_member(msg.author.id) or msg.author).display_name

            # 2. Process Attachments
            valid_attachments = [a for a in msg.attachments if self._is_supported(a)]
            node.has_bad_attachments = len(msg.attachments) > len(valid_attachments)

            if not valid_attachments:
                return

            results = await asyncio.gather(*[self._download_attachment(self.httpx_client, a) for a in valid_attachments])

            node.images = []
            for attachment, content in zip(valid_attachments, results, strict=True):
                if content is None:
                    node.has_bad_attachments = True
                    continue

                ctype = attachment.content_type or ""
                if ctype.startswith("text"):
                    node.text += f"\n{content.decode('utf-8', errors='ignore')}"
                elif ctype.startswith("image"):
                    b64 = b64encode(content).decode()
                    node.images.append({"type": "image_url", "image_url": {"url": f"data:{ctype};base64,{b64}"}})

    def _is_supported(self, a: discord.Attachment) -> bool:
        return any(a.content_type.startswith(t) for t in ("text", "image")) if a.content_type else False

    async def construct_llm_payload(self, message: discord.Message) -> dict[str, Any]:
        """Construct the LLM payload from the given message."""
        # 1. Context Resolution
        provider, model, vision = self._parse_model_str(message)
        history = await self._fetch_history(message)

        # 2. Node Syncing
        await asyncio.gather(*[self._ensure_node(m) for m in history])

        # 3. Payload Construction
        params = self._get_payload_params(provider, model, force_vision=vision)
        messages = self._assemble_chat_history(history, params)

        # 4. Prompt Injection
        messages = self._inject_system_prompts(messages, message, model, provider)

        # 5. Model-Specific Config
        return self._finalize_payload(messages, provider, model)

    def _assemble_chat_history(self, history: list[Message], params: MessagePayloadParams) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for msg in history:
            node = self.message_nodes.get(msg.id)
            if not node or not node.text:
                continue

            # Text Processing
            text = node.text[: params.max_text]
            if params.prefix_users and not params.accept_usernames and node.role in ("user", "assistant"):
                text = f"[{node.created_at:%Y-%m-%d %H:%M}] {node.user_display_name}({node.user_id}): {text}"

            # Structure
            content_data: str | list[dict[str, Any]]
            content_data = [{"type": "text", "text": text}, *node.images[: params.max_images]] if params.accept_images and node.images else text

            msg_dict: dict[str, Any] = {"role": node.role, "content": content_data}
            if params.accept_usernames and node.role == "user":
                msg_dict["name"] = sanitize_symbols(node.user_display_name or "")[:64]

            payload.append(msg_dict)

        return payload[::-1]  # Chronological

    def _finalize_payload(self, messages: list[dict[str, Any]], provider: str, model: str) -> dict[str, Any]:
        provider_config = self.config.llm.providers.get(provider, {})
        full_model = f"{provider}/{model}"
        model_config = self.config.llm.models.get(full_model, {})

        tools = []
        tool_overhead = 0

        if litellm.supports_function_calling(model=full_model) and self.config.chat.use_tools:
            tools = get_tool_definitions()
            tool_string = json.dumps(tools)
            tool_overhead = (len(tool_string) // 3) + 100

        effective_max_tokens = max(0, self.config.chat.max_input_tokens - tool_overhead)

        # Trim tokens
        trimmed_result = litellm_utils.trim_messages(messages, model=full_model, max_tokens=effective_max_tokens, trim_ratio=1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        messages = trimmed_result if isinstance(trimmed_result, list) else messages  # pyright: ignore[reportUnknownVariableType]

        payload: dict[str, Any] = {
            "model": full_model if provider in litellm.provider_list else f"openai/{model}",
            "messages": messages,
            "api_key": provider_config.get("api_key") if isinstance(provider_config, dict) else None,
            "api_base": provider_config.get("base_url") if isinstance(provider_config, dict) else None,
            "extra_body": model_config.get("extra_body") if isinstance(model_config, dict) else None,
            "extra_headers": model_config.get("extra_headers") if isinstance(model_config, dict) else None,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        return payload

    def _parse_model_str(self, msg: Message) -> tuple[str, str, bool]:
        raw = self.config.chat.channel_models.get(msg.channel.id, self.config.chat.default_model)

        has_vision_override = raw.endswith(":vision")
        clean_raw = raw.removesuffix(":vision")

        if "/" not in clean_raw:
            error = f"Invalid model string: {raw}"
            raise ValueError(error)

        provider, model = clean_raw.split("/", 1)
        return provider, model, has_vision_override

    async def _fetch_history(self, message: Message) -> list[Message]:
        use_ctx = self.config.chat.use_channel_context
        if use_ctx and self.config.chat.force_reply_chains and message.reference:
            use_ctx = False
        return await fetch_history(message, self.config.chat.max_messages, use_channel_context=use_ctx, bot_user=self.user)

    def _get_payload_params(self, provider: str, model: str, *, force_vision: bool) -> MessagePayloadParams:
        return MessagePayloadParams(
            max_text=self.config.chat.max_text,
            max_images=self.config.chat.max_images,
            prefix_users=self.config.chat.prefix_users,
            accept_images=force_vision or litellm.supports_vision(model, custom_llm_provider=provider),
            accept_usernames=any(provider.startswith(p) for p in PROVIDERS_SUPPORTING_USERNAMES),
        )

    def _inject_system_prompts(self, messages: list[dict[str, Any]], msg: Message, model: str, provider: str) -> list[dict[str, Any]]:
        pre = replace_placeholders(self.config.prompts.pre_history, msg, self.user, model, provider)
        post = replace_placeholders(self.config.prompts.post_history, msg, self.user, model, provider)
        if pre:
            messages.insert(0, {"role": "system", "content": pre})
        if post:
            messages.append({"role": "system", "content": post})
        return messages

    @staticmethod
    def _get_text_content(msg: discord.Message) -> str:
        parts: list[str] = []

        content = msg.content.lstrip()
        if content:
            parts.append(content)

        for embed in msg.embeds:
            if embed.title:
                parts.append(embed.title)
            if embed.description:
                parts.append(embed.description)
            if embed.footer and embed.footer.text:
                parts.append(embed.footer.text)

        for component in msg.components:
            if component.type == discord.ComponentType.text_display:
                content = getattr(component, "content", "")
                if content:
                    parts.append(content)

        return "\n".join(parts)

    @staticmethod
    async def _download_attachment(client: httpx.AsyncClient, attachment: discord.Attachment) -> bytes | None:
        try:
            resp = await client.get(attachment.url, timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to download {attachment.filename}: HTTP {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Failed to download {attachment.filename}: {e}")
            return None
        else:
            return resp.content
