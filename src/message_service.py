"""Message Processor for handling Discord message operations.

This module provides a service for processing Discord messages,
including validation, response sending, and node management.
"""

import asyncio
import logging
import os
from base64 import b64encode
from collections.abc import Sequence
from typing import Any, cast

import discord
import httpx
import litellm
from discord import Message
from litellm import utils as litellm_utils

from .config_manager import ConfigValue, RootConfig, config_manager
from .custom_types import BuildMessagesParams, MessageNode, MessagePayloadParams
from .discord_utils import fetch_history
from .llm_tools import tool_manager
from .regex_utils import replace_placeholders, sanitize_symbols
from .time_utils import time_performance

PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
MAX_MESSAGE_NODES: int = 500

logger = logging.getLogger(__name__)
config = config_manager.config
os.environ["LITELLM_LOG"] = "ERROR"
litellm.telemetry = False
litellm.modify_params = True


class MessageService:
    """Service for processing Discord messages and managing responses."""

    def __init__(self, config: RootConfig, user: discord.ClientUser, httpx_client: httpx.AsyncClient) -> None:
        """Initialize the message processor.

        :param config: The application configuration.
        :param bot_user: The bot's user object.
        """
        self.config = config
        self.httpx_client = httpx_client
        self.user = user
        self.message_nodes: dict[int, MessageNode] = {}

    def release_node_locks(self, msgs: list[discord.Message], full_content: str) -> None:
        """Update and release locks on message nodes.

        :param msgs: The messages that were sent.
        :param full_content: The complete response content.
        """
        for msg in msgs:
            node = self.message_nodes.get(msg.id)
            if node:
                node.text = full_content
                if hasattr(node.lock, "locked") and node.lock.locked():
                    node.lock.release()

    def prune_msg_nodes(self) -> None:
        """Prune old message nodes to prevent memory leaks."""
        if len(self.message_nodes) <= MAX_MESSAGE_NODES:
            return

        to_remove_count = len(self.message_nodes) - MAX_MESSAGE_NODES
        logger.debug("Pruning %d old MsgNodes...", to_remove_count)

        sorted_ids = sorted(self.message_nodes.keys())
        ids_to_delete = sorted_ids[:to_remove_count]

        for msg_id in ids_to_delete:
            self.message_nodes.pop(msg_id, None)

        logger.debug("Successfully pruned %d nodes!", len(ids_to_delete))

    async def _init_msg_node(self, msg: discord.Message) -> None:
        """Initialize a MessageNode with granular debug logging."""
        msg_id = msg.id
        # start_time = time.perf_counter()

        # logger.debug("[%s] Starting _init_msg_node", msg_id)
        node = self.message_nodes.setdefault(msg_id, MessageNode())

        if node.text is not None:
            # logger.debug("[%s] Node already initialized, skipping", msg_id)
            return

        # logger.debug("[%s] Attempting to acquire lock...", msg_id)
        # lock_start = time.perf_counter()

        async with node.lock:
            # logger.debug("[%s] Lock acquired in %.4fs", msg_id, time.perf_counter() - lock_start)

            if node.text is not None:
                # logger.debug("[%s] Node initialized by another task while waiting for lock", msg_id)
                return

            # 1. Text Extraction
            # logger.debug("[%s] Extracting text from content/embeds/components", msg_id)
            text_parts = [msg.content.lstrip()] if msg.content.lstrip() else []
            text_parts.extend(get_embed_text(embed) for embed in msg.embeds)
            text_parts.extend(get_component_text(c) for c in msg.components)

            # 2. Attachment Handling
            to_download = [a for a in msg.attachments if is_supported_attachment(a)]
            if to_download:
                # logger.debug("[%s] Found %d attachments to download", msg_id, len(to_download))
                # download_start = time.perf_counter()

                try:
                    # This is the most likely place for a hang
                    downloads = await asyncio.gather(*(download_attachment(self.httpx_client, a) for a in to_download))
                    # logger.debug("[%s] Downloads completed in %.4fs", msg_id, time.perf_counter() - download_start)
                except Exception:
                    downloads = []
            else:
                downloads = []

            # 3. Processing Results
            # logger.debug("[%s] Processing %d download results", msg_id, len(downloads))
            node.images = []
            for attachment, resp in downloads:
                if resp is None:
                    # logger.warning("[%s] Failed download for attachment: %s", msg_id, attachment.filename)
                    node.has_bad_attachments = True
                    continue

                content_type = attachment.content_type or ""
                if content_type.startswith("text"):
                    # Potential hang if file is massive
                    # logger.debug("[%s] Reading text file: %s", msg_id, attachment.filename)
                    text_parts.append(resp.text)
                elif content_type.startswith("image"):
                    # Potential hang (CPU bound) if image is massive
                    # logger.debug("[%s] Encoding image: %s", msg_id, attachment.filename)
                    base64_data = b64encode(resp.content).decode()
                    node.images.append({"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_data}"}})

            # 4. Metadata Assignment
            # logger.debug("[%s] Finalizing node metadata", msg_id)
            node.text = "\n".join(filter(None, text_parts))
            node.role = "assistant" if msg.author == self.user else "user"
            node.user_id = msg.author.id
            node.created_at = msg.created_at

            # Guild member resolution
            author = msg.author
            if msg.guild and not isinstance(author, discord.Member):
                node.user_display_name = (msg.guild.get_member(author.id) or author).display_name
            else:
                node.user_display_name = author.display_name

            node.has_bad_attachments |= len(msg.attachments) > len(to_download)

        # logger.debug("[%s] _init_msg_node finished. Total time: %.4fs", msg_id, time.perf_counter() - start_time)

    @time_performance("Message nodes initialization")
    async def _initialize_message_nodes(self, message_history: list[discord.Message]) -> None:
        """Initialize message nodes for the message history.

        :param message_history: List of messages to process.
        """
        tasks = [asyncio.wait_for(self._init_msg_node(msg), timeout=10.0) for msg in message_history]

        # return_exceptions=True ensures that if one message hangs/fails,
        # the others still complete.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log failures for debugging
        for i, result in enumerate(results):
            if isinstance(result, asyncio.TimeoutError):
                logger.warning("Message %s timed out during initialization (10s limit)", message_history[i].id)
            elif isinstance(result, Exception):
                logger.error("Message %s failed initialization: %s", message_history[i].id, result)

    async def construct_llm_payload(
        self,
        message: discord.Message,
        bot_user: discord.ClientUser,
    ) -> dict[str, Any]:
        """Prepare the LLM request parameters for a message.

        :param message: The Discord message to process.
        :param bot_user: The bot's user object.
        :return: Chat parameters for the LLM call.
        """
        provider, model = self._get_provider_and_model(message)
        message_history = await self._prepare_message_history(message, bot_user)
        await self._initialize_message_nodes(message_history)

        build_params = await self._create_build_parameters(message, message_history, self.message_nodes, model, provider, bot_user)
        payload_params = self._create_payload_params(build_params)
        content_messages = self._build_messages_payload(build_params.message_history, build_params.message_nodes, payload_params)
        self._insert_prompts(build_params, content_messages)
        messages_payload = self._trim_messages_if_needed(content_messages, build_params)

        payload = self._create_chat_parameters(model, messages_payload, provider)

        payload["tools"] = tool_manager.get_tool_definitions()
        payload["tool_choice"] = "auto"

        return payload

    async def _create_build_parameters(
        self,
        message: discord.Message,
        message_history: list[discord.Message],
        message_nodes: dict[int, Any],
        model: str,
        provider: str,
        bot_user: discord.ClientUser,
    ) -> BuildMessagesParams:
        """Create build parameters for message processing.

        :param message: The Discord message.
        :param message_history: List of messages in the history.
        :param message_nodes: Dictionary of message nodes.
        :param model: The LLM model to use.
        :param provider: The LLM provider.
        :param bot_user: The bot's user object.
        :return: BuildMessagesParams object.
        """
        pre_history, post_history = self._generate_system_prompts(message, model, provider, bot_user)
        return BuildMessagesParams(
            message_history=message_history,
            message_nodes=message_nodes,
            pre_history=pre_history,
            post_history=post_history,
            model=model,
            provider=provider,
        )

    def _create_chat_parameters(
        self,
        model: str,
        messages_payload: list[dict[str, Any]],
        provider: str,
    ) -> dict[str, Any]:
        """Create the final chat parameters for the LLM call.

        :param model: The LLM model to use.
        :param messages_payload: The processed messages payload.
        :param provider: The LLM provider.
        :return: Dictionary of chat parameters.
        """
        provider_config = self._get_provider_config(provider)
        return build_chat_params(
            model=model,
            messages=messages_payload,
            provider=provider,
            provider_config=provider_config,
            llm_models_config=self.config.llm.models,
        )

    async def _prepare_message_history(self, message: discord.Message, bot_user: discord.ClientUser) -> list[discord.Message]:
        """Prepare the message history for context."""
        config = self.config
        use_channel_context = config.chat.use_channel_context
        if use_channel_context and config.chat.force_reply_chains and message.reference:
            use_channel_context = False

        return await fetch_history(
            message=message,
            max_messages=config.chat.max_messages,
            use_channel_context=use_channel_context,
            bot_user=bot_user,
        )

    def _generate_system_prompts(self, message: discord.Message, model: str, provider: str, bot_user: discord.ClientUser) -> tuple[str | None, str | None]:
        """Generate system prompts."""
        pre_history = replace_placeholders(self.config.prompts.pre_history, message, bot_user, model, provider)
        post_history = replace_placeholders(self.config.prompts.post_history, message, bot_user, model, provider)
        return pre_history, post_history

    def _create_payload_params(self, build_params: BuildMessagesParams) -> MessagePayloadParams:
        """Create payload parameters based on model capabilities."""
        accept_images = litellm.supports_vision(build_params.model, custom_llm_provider=build_params.provider)
        accept_usernames = self._provider_supports_usernames(build_params.provider)

        return MessagePayloadParams(
            max_text=self.config.chat.max_text,
            max_images=self.config.chat.max_images,
            prefix_users=self.config.chat.prefix_users,
            accept_images=accept_images,
            accept_usernames=accept_usernames,
        )

    def _provider_supports_usernames(self, provider: str) -> bool:
        """Check if provider supports usernames in messages."""
        return any(provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    def _insert_prompts(
        self,
        build_params: BuildMessagesParams,
        content_messages: list[dict[str, Any]],
    ) -> None:
        """Assemble the complete messages payload with system prompts."""
        if build_params.pre_history:
            content_messages.insert(0, {"role": "system", "content": build_params.pre_history})
        if build_params.post_history:
            content_messages.append({"role": "system", "content": build_params.post_history})

    def _trim_messages_if_needed(
        self,
        messages_payload: list[dict[str, Any]],
        build_params: BuildMessagesParams,
    ) -> list[dict[str, Any]]:
        """Trim messages payload if it exceeds token limit."""
        full_model_name = f"{build_params.provider}/{build_params.model}"
        trimmed_result = litellm_utils.trim_messages(  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream
            messages_payload,
            model=full_model_name,
            max_tokens=self.config.chat.max_input_tokens,
        )
        # Cast to expected type since litellm's annotations are incomplete
        return cast("list[dict[str, Any]]", trimmed_result) if isinstance(trimmed_result, list) else messages_payload

    def _get_provider_config(self, provider: str) -> ConfigValue:
        """Get provider configuration."""
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            error = f"Provider '{provider}' not found in configuration."
            raise KeyError(error)
        return provider_config

    def _get_llm_provider_model(self, channel_id: int, channel_models: dict[int, str], default_model: str) -> tuple[str, str]:
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

    def _get_provider_and_model(self, message: discord.Message) -> tuple[str, str]:
        """Get the LLM provider and model for a message.

        :param message: The Discord message.
        :return: Tuple of (provider, model).
        """
        return self._get_llm_provider_model(
            channel_id=message.channel.id,
            channel_models=self.config.chat.channel_models,
            default_model=self.config.chat.default_model,
        )

    def _build_messages_payload(self, message_history: list[Message], message_nodes: dict[int, MessageNode], params: MessagePayloadParams) -> list[dict[str, Any]]:
        """Build the messages payload for the LLM.

        :param message_history: The history of messages to process.
        :param message_nodes: The dictionary of processed message nodes.
        :param params: Parameters for payload creation.
        :return: A list of messages for the payload.
        """
        messages_payload: list[dict[str, Any]] = []

        for msg in message_history:
            node = message_nodes.get(msg.id)
            if not node or node.text is None:
                logger.debug("Empty or missing message node found, skipping...")
                continue

            message_payload = create_message_payload(node=node, params=params)

            if message_payload:
                messages_payload.append(message_payload)

        return messages_payload[::-1]


def get_embed_text(embed: discord.Embed) -> str:
    """Extract text from an embed.

    :param embed: The Discord embed to process.
    :return: A string containing the title, description, and footer text.
    """
    fields: list[str | None] = [embed.title, embed.description, getattr(embed.footer, "text", None)]
    return "\n".join(filter(None, fields))


def get_component_text(component: discord.Component) -> str:
    """Extract text from a component.

    :param component: The Discord component to process.
    :return: The text content if the component is a TextDisplay, otherwise an empty string.
    """
    return getattr(component, "content", "") if component.type == discord.ComponentType.text_display else ""


def is_supported_attachment(attachment: discord.Attachment) -> bool:
    """Check if attachment type is supported.

    :param attachment: The Discord attachment to check.
    :return: True if the attachment is a text or image file, False otherwise.
    """
    return any(attachment.content_type.startswith(t) for t in ("text", "image")) if attachment.content_type else False


async def download_attachment(httpx_client: httpx.AsyncClient, attachment: discord.Attachment) -> tuple[discord.Attachment, httpx.Response | None]:
    """Download a Discord attachment using the provided HTTP client.

    :param httpx_client: The shared HTTPX client instance.
    :param attachment: The Discord attachment to download.
    :return: A tuple containing the original attachment and the HTTP response (or None if failed).
    """
    try:
        resp = await httpx_client.get(attachment.url)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as error:
        logger.warning("Failed to download attachment %s (%s): %s", attachment.filename, attachment.url, error)
        return attachment, None
    else:
        return attachment, resp


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


def create_message_payload(
    node: MessageNode,
    params: MessagePayloadParams,
) -> dict[str, Any] | None:
    """Create a single message payload from a MessageNode.

    :param node: The message node to process.
    :param params: Parameters for payload creation.
    :return: The message payload or None if empty.
    """
    formatted_text = node.text[: params.max_text] if node.text else ""

    if params.prefix_users and not params.accept_usernames and node.user_display_name and node.role in {"user", "assistant"}:
        formatted_text = f"[{node.created_at:%Y-%m-%d %H:%M}] {node.user_display_name}({node.user_id}): {formatted_text}"

    images_to_add: list[dict[str, Any]] = node.images[: params.max_images] if params.accept_images else []

    content: str | list[dict[str, Any]]
    if images_to_add:
        parts: list[dict[str, Any]] = []
        if formatted_text:
            parts.append({"type": "text", "text": formatted_text})
        parts.extend(images_to_add)
        content = parts
    else:
        content = formatted_text

    if not content:
        return None

    message_payload: dict[str, Any]
    if node.role == "user":
        user_payload: dict[str, Any] = {"role": "user", "content": content}
        if params.accept_usernames and node.user_id:
            user_payload["name"] = sanitize_symbols(node.user_display_name or "")[:64] or str(node.user_id)
        message_payload = user_payload
    else:
        message_payload = {"role": "assistant", "content": formatted_text}

    return message_payload
