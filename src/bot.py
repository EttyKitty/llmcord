"""LLMCord Bot Main Module.

This module contains the core implementation of the LLMCord Discord bot.
It handles the bot's lifecycle, event processing, message validation,
context construction, and interaction with various LLM providers using
OpenAI-compatible APIs.
"""

import asyncio
import logging
import os
import re
import threading
import time
from base64 import b64encode
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import discord
import httpx
import tiktoken
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from .commands import setup
from .config import ConfigValue, RootConfig, config_manager
from .logger import request_logger
from .utils import MsgNode, clean_response, is_message_allowed

REGEX_USER_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500
TOKENIZER = tiktoken.get_encoding("cl100k_base")


logger = logging.getLogger(__name__)


async def main() -> None:
    """Entry point for the bot application.

    Initializes Discord intents, creates the bot instance, starts the console listener,
    and runs the bot with the configured token.

    :return: None
    :raises KeyboardInterrupt: Handled gracefully to allow clean shutdown.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(name=(config_manager.config.discord.status_message)[:128])

    bot = LLMCordBot(intents=intents, activity=activity)
    threading.Thread(target=console_listener, daemon=True).start()

    try:
        async with bot:
            await bot.start(config_manager.config.discord.bot_token)
    except KeyboardInterrupt:
        pass


def console_listener() -> None:
    """Listen for console commands in a blocking loop.

    Accepts 'reload', 'exit', 'stop', or 'quit' commands to control the bot process.

    :return: None
    """
    try:
        while True:
            command = input().strip().lower()
            if command == "reload":
                os._exit(2)
            elif command in ("exit", "stop", "quit"):
                os._exit(0)
            else:
                logger.warning("Unknown command: %s", command)
    except EOFError:
        pass


class LLMCordBot(commands.Bot):
    """The main bot class for llmcord, encapsulating state and logic."""

    @property
    def config(self) -> RootConfig:
        """Get the current configuration.

        :return: The root configuration object.
        """
        return config_manager.config

    # I'm too stupid to find a different solution to strict type checking in local scopes
    @property
    def safe_user(self) -> discord.ClientUser:
        """Get the bot user, raising an error if not initialized.

        :return: The bot's ClientUser.
        :raises RuntimeError: If the bot user is not yet initialized.
        """
        if self.user is None:
            error = "Bot user not initialized!"
            raise RuntimeError(error)

        return self.user

    def __init__(self, intents: discord.Intents, activity: discord.CustomActivity) -> None:
        """Initialize the LLMCordBot.

        :param intents: Discord intents configuration.
        :param activity: Custom activity status for the bot.
        """
        super().__init__(command_prefix="?", intents=intents, activity=activity)

        self.openai_clients: dict[str, AsyncOpenAI] = {}
        self.msg_nodes: dict[int, MsgNode] = {}
        self.last_task_time: float = 0.0
        self.httpx_client = httpx.AsyncClient()

    async def close(self) -> None:
        """Cleanup resources before shutting down."""
        await self.httpx_client.aclose()
        await super().close()

    async def setup_hook(self) -> None:
        """Set up internal discord.py hook for asynchronous setup."""
        await setup(self)
        await self.tree.sync()

        if client_id := self.config.discord.client_id:
            logger.info("Bot invite URL: https://discord.com/oauth2/authorize?client_id=*%d&permissions=412317191168&scope=bot", client_id)

        logger.info("Bot ready. Logged in as %s", self.safe_user)

    async def on_message(self, message: discord.Message) -> None:
        """Event handler for new messages. Orchestrates the response flow.

        :param message: The incoming Discord message to process.
        """
        # =============================
        # ======== 0. Gatekeep ========
        # =============================
        if not self.safe_user or message.author.bot:
            return

        is_dm = message.channel.type == discord.ChannelType.private
        is_mentioned = self.safe_user in message.mentions

        if not (is_dm or is_mentioned):
            return

        if not is_message_allowed(message, self.config.discord.permissions, allow_dms=self.config.discord.allow_dms):
            logger.info("Message blocked. User: %s ID: %d", message.author.name, message.author.id)
            return

        # =============================
        # === 1. Build the Request ====
        # =============================
        # - Build Context
        start_time = time.perf_counter()
        provider, model = self._get_llm(message.channel.id)
        # - Generate system prompts early to calculate their token cost
        pre_history = self._replace_placeholders(self.config.prompts.pre_history, message, model, provider)
        post_history = self._replace_placeholders(self.config.prompts.post_history, message, model, provider)

        system_token_count = 0
        if pre_history:
            system_token_count += len(TOKENIZER.encode(pre_history))
        if post_history:
            system_token_count += len(TOKENIZER.encode(post_history))

        # - Build the History
        message_history = await self._prepare_message_history(message)

        # - Build the Payload
        accept_images, accept_usernames = self._get_llm_specials(provider, model)
        messages_payload = await self._build_messages_payload(
            message_history=message_history,
            system_token_count=system_token_count,
            accept_images=accept_images,
            accept_usernames=accept_usernames,
        )

        # - Insert Prompts
        if pre_history:
            messages_payload.insert(0, {"role": "system", "content": pre_history})

        if post_history:
            messages_payload.append({"role": "system", "content": post_history})

        # - Finish the Payload
        provider_config, openai_client = self._get_provider_config(provider)
        chat_params = self._build_chat_params(model, messages_payload, provider, provider_config)

        elapsed_time = time.perf_counter() - start_time
        logger.info("Request prepared in %s seconds!", f"{elapsed_time:.4f}")

        # 2. Send the Request + 3. Get the Response (Fuck, why are they in one function)
        await self._generate_response(message, openai_client, chat_params)

        # 4. Cleanup
        self._prune_msg_nodes()

    def _get_llm(self, channel_id: int) -> tuple[str, str]:
        """Resolve the LLM provider, model.

        :param channel_id: The Discord channel that triggered the request.
        :return: A tuple containing provider name, model name
        """
        provider_slash_model = self.config.chat.channel_models.get(channel_id, self.config.chat.default_model)
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
        return provider, model

    def _get_llm_specials(self, provider: str, model: str) -> tuple[bool, bool]:
        """Resolve the LLM flags based on configuration.

        :param message: The Discord message that triggered the request.
        :return: A tuple containing accept_images flag, and accept_usernames flag.
        """
        accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
        accept_usernames = any(provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

        return accept_images, accept_usernames

    def _get_provider_config(self, provider: str) -> tuple[ConfigValue, AsyncOpenAI]:
        """Resolve the LLM client.

        :param channel_id: The Discord channel that triggered the request.
        :return: A tuple containing provider config, OpenAI client.
        """
        provider_config = self.config.llm.providers[provider]
        openai_client = self.get_openai_client(provider_config)

        return provider_config, openai_client

    def _replace_placeholders(self, text: str, msg: discord.Message, model: str, provider: str) -> str:
        """Replace dynamic placeholders in prompt strings."""
        now = datetime.now(timezone.utc)
        user_roles = getattr(msg.author, "roles", [])
        user_roles_str = ", ".join([role.name for role in user_roles if role.name != "@everyone"]) or "None"
        guild_emojis = getattr(msg.guild, "emojis", [])
        guild_emojis_str = ", ".join([str(emoji) for emoji in guild_emojis]) or "None"

        placeholders = {
            "{date}": now.strftime("%B %d %Y"),
            "{time}": now.strftime("%H:%M:%S %Z%z"),
            "{bot_name}": self.safe_user.display_name,
            "{bot_id}": str(self.safe_user.id),
            "{model}": model,
            "{provider}": provider,
            "{user_display_name}": msg.author.display_name,
            "{user_id}": str(msg.author.id),
            "{user_roles}": user_roles_str,
            "{guild_name}": msg.guild.name if msg.guild else "Direct Messages",
            "{guild_description}": msg.guild.description if msg.guild else "",
            "{guild_emojis}": guild_emojis_str,
            "{channel_name}": getattr(msg.channel, "name", ""),
            "{channel_topic}": getattr(msg.channel, "topic", ""),
            "{channel_nsfw}": str(getattr(msg.channel, "nsfw", False)),
        }
        for key, value in placeholders.items():
            text = text.replace(key, str(value))
        return text.strip()

    async def _prepare_message_history(self, message: discord.Message) -> list[discord.Message]:
        """Fetch and initialize the message history for context building.

        :param message: The triggering message.
        :return: A list of initialized message nodes.
        """
        config = self.config
        use_channel_context = config.chat.use_channel_context
        if use_channel_context and config.chat.force_reply_chains and message.reference:
            use_channel_context = False

        logger.debug("Initializing message nodes...")
        message_history = await self._fetch_history(message, config.chat.max_messages, use_channel_context=use_channel_context)
        await asyncio.gather(*(self._init_msg_node(m) for m in message_history))

        return message_history

    async def _fetch_history(self, message: discord.Message, max_messages: int, *, use_channel_context: bool) -> list[discord.Message]:
        """Retrieve message history either via channel history or reply chain.

        :param message: The trigger message.
        :param max_messages: Maximum number of messages to fetch.
        :param use_channel_context: Whether to use linear channel history.
        :return: A list of Discord messages.
        """
        logger.debug("Building message history... (Mode: %s)", "Channel History" if use_channel_context else "Reply Chain")

        if use_channel_context:
            return await self._fetch_channel_history(message, max_messages)

        return await self._fetch_reply_chain_history(message, max_messages)

    async def _fetch_channel_history(self, message: discord.Message, max_messages: int) -> list[discord.Message]:
        """Fetch linear message history from the channel."""
        message_history = [message]
        message_history.extend([msg async for msg in message.channel.history(limit=max_messages - 1, before=message)])
        return message_history[:max_messages]

    async def _fetch_reply_chain_history(self, message: discord.Message, max_messages: int) -> list[discord.Message]:
        """Fetch message history by traversing the reply chain."""
        message_history: list[discord.Message] = []
        history_ids: set[int] = set()
        current_msg: discord.Message | None = message

        while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
            history_ids.add(current_msg.id)
            message_history.append(current_msg)

            current_msg = await self._resolve_next_message(current_msg)

        return message_history

    async def _resolve_next_message(self, current_msg: discord.Message) -> discord.Message | None:
        """Determine the next message in the reply chain."""
        if current_msg.reference and current_msg.reference.message_id:
            return await self._fetch_referenced_message(current_msg)

        if self.safe_user.mention not in current_msg.content:
            return await self._fetch_previous_message(current_msg)

        return None

    async def _fetch_referenced_message(self, current_msg: discord.Message) -> discord.Message | None:
        """Fetch the message referenced by the current message."""
        if not current_msg.reference or not current_msg.reference.message_id:
            return None

        try:
            next_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            logger.exception("Failed to fetch parent for message %d", current_msg.reference.message_id)
            return None
        else:
            if isinstance(current_msg.channel, discord.Thread) and isinstance(current_msg.channel.parent, discord.abc.Messageable):
                next_msg = current_msg.channel.starter_message or await current_msg.channel.parent.fetch_message(current_msg.channel.id)

            return next_msg

    async def _fetch_previous_message(self, current_msg: discord.Message) -> discord.Message | None:
        """Fetch the immediately preceding message in the channel if it matches criteria."""
        async for prev in current_msg.channel.history(before=current_msg, limit=1):
            is_dm = current_msg.channel.type == discord.ChannelType.private
            allowed_types = (discord.MessageType.default, discord.MessageType.reply)

            if prev.type not in allowed_types:
                continue

            is_expected_author = prev.author in (self.safe_user, current_msg.author) if is_dm else prev.author == current_msg.author

            if is_expected_author:
                return prev

        return None

    async def _init_msg_node(self, msg: discord.Message) -> None:
        """Initialize a MsgNode for a message, processing attachments and text sources.

        :param msg: The Discord message to process.
        """
        node = self.msg_nodes.setdefault(msg.id, MsgNode())
        if node.text is not None:
            return

        async with node.lock:
            if node.text is not None:
                return

            text_parts = [msg.content.lstrip()] if msg.content.lstrip() else []
            text_parts.extend(self._get_embed_text(error) for error in msg.embeds)
            text_parts.extend(self._get_component_text(c) for c in msg.components)

            to_download = [a for a in msg.attachments if self._is_supported_attachment(a)]
            downloads = await asyncio.gather(*(self._download_attachment(a) for a in to_download))

            node.images = []
            for attachment, resp in downloads:
                if resp is None:
                    node.has_bad_attachments = True
                    continue

                content_type = attachment.content_type or ""
                if content_type.startswith("text"):
                    text_parts.append(resp.text)
                elif content_type.startswith("image"):
                    base64_data = b64encode(resp.content).decode()
                    node.images.append({"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_data}"}})

            node.text = "\n".join(filter(None, text_parts))
            node.role = "assistant" if msg.author == self.safe_user else "user"
            node.user_id = msg.author.id if node.role == "user" else None

            author = msg.author
            if msg.guild and not isinstance(author, discord.Member):
                author = msg.guild.get_member(author.id) or author

            node.user_display_name = author.display_name if node.role == "user" else None
            node.has_bad_attachments = len(msg.attachments) > len(to_download)

    def _get_embed_text(self, embed: discord.Embed) -> str:
        """Extract text from an embed."""
        fields = [embed.title, embed.description, getattr(embed.footer, "text", None)]
        return "\n".join(filter(None, fields))

    def _get_component_text(self, component: discord.Component) -> str:
        """Extract text from a component."""
        return getattr(component, "content", "") if component.type == discord.ComponentType.text_display else ""

    def _is_supported_attachment(self, attachment: discord.Attachment) -> bool:
        """Check if attachment type is supported."""
        return any(attachment.content_type.startswith(t) for t in ("text", "image")) if attachment.content_type else False

    async def _download_attachment(self, attachment: discord.Attachment) -> tuple[discord.Attachment, httpx.Response | None]:
        """Download an attachment."""
        try:
            resp = await self.httpx_client.get(attachment.url)
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as error:
            logger.warning("Failed to download attachment %s (%s): %s", attachment.filename, attachment.url, error)
            return attachment, None
        else:
            return attachment, resp

    async def _build_messages_payload(
        self,
        message_history: list[discord.Message],
        system_token_count: int,
        *,
        accept_images: bool,
        accept_usernames: bool,
    ) -> list[ChatCompletionMessageParam]:
        """Build the messages payload for the LLM, including context.

        :param message_history: The history of messages to process.
        :param system_token_count: Token count of system prompts.
        :param accept_images: Whether the model supports image inputs.
        :param accept_usernames: Whether the provider supports usernames in messages.
        :return: A list of messages for the payload.
        """
        messages_payload: list[ChatCompletionMessageParam] = []
        total_tokens = system_token_count

        for msg in message_history:
            if len(messages_payload) >= self.config.chat.max_messages:
                logger.debug("Message limit reached, breaking... (%d)", self.config.chat.max_messages)
                break

            node = self.msg_nodes[msg.id]
            if node.text is None:
                logger.debug("Empty message found, skipping...")
                continue

            message_payload, tokens = self._create_message_payload(
                node,
                accept_images=accept_images,
                accept_usernames=accept_usernames,
            )

            if message_payload is None:
                continue

            if total_tokens + tokens > self.config.chat.max_input_tokens and messages_payload:
                logger.debug("Context limit reached, breaking...")
                break

            total_tokens += tokens
            messages_payload.append(message_payload)

        logger.debug("Context ready. Messages: %d.", len(messages_payload))

        return messages_payload[::-1]

    def _create_message_payload(
        self,
        node: MsgNode,
        *,
        accept_images: bool,
        accept_usernames: bool,
    ) -> tuple[ChatCompletionMessageParam | None, int]:
        """Create a single message payload from a MsgNode.

        :param node: The message node to process.
        :param accept_images: Whether to include images.
        :param accept_usernames: Whether to include usernames.
        :return: A tuple of message_payload and token_count.
        """
        config = self.config
        formatted_text = node.text[: config.chat.max_text] if node.text else ""

        if config.chat.prefix_users and not accept_usernames and node.role == "user" and node.user_display_name:
            formatted_text = f"{node.user_display_name}({node.user_id}): {formatted_text}"

        text_tokens = len(TOKENIZER.encode(formatted_text))
        images_to_add = node.images[: config.chat.max_images] if accept_images else []
        image_tokens = len(images_to_add) * 1100
        msg_tokens = text_tokens + image_tokens

        content: str | list[ChatCompletionContentPartParam]
        if images_to_add:
            parts: list[ChatCompletionContentPartParam] = []
            if formatted_text:
                parts.append({"type": "text", "text": formatted_text})
            parts.extend(images_to_add)
            content = parts
        else:
            content = formatted_text

        if not content:
            return None, 0

        message_payload: ChatCompletionMessageParam
        if node.role == "user":
            user_payload: ChatCompletionUserMessageParam = {"role": "user", "content": content}
            if accept_usernames and node.user_id:
                sanitized_name = REGEX_USER_NAME_SANITIZER.sub("", node.user_display_name or "")[:64]
                user_payload["name"] = sanitized_name or str(node.user_id)
            message_payload = user_payload
        else:
            message_payload = {"role": "assistant", "content": formatted_text}

        return message_payload, msg_tokens

    def get_openai_client(self, provider_config: ConfigValue) -> AsyncOpenAI:
        """Retrieve or initialize an OpenAI-compatible client for a specific provider.

        :param provider_config: The configuration dictionary for the provider.
        :return: An AsyncOpenAI client instance.
        """
        if not isinstance(provider_config, dict):
            error = f"Provider config must be a dict, got {type(provider_config)}"
            raise TypeError(error)

        base_url = provider_config["base_url"]
        if base_url not in self.openai_clients:
            self.openai_clients[base_url] = AsyncOpenAI(
                base_url=base_url,
                api_key=provider_config.get("api_key", "sk-no-key-required"),
                http_client=self.httpx_client,
            )

        return self.openai_clients[base_url]

    def _build_chat_params(
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        provider: str,
        provider_config: ConfigValue,
    ) -> dict[str, Any]:
        """Construct the parameters for the OpenAI chat completion call.

        :param model: The model identifier.
        :param messages: The list of message payloads.
        :param provider: The provider identifier.
        :param provider_config: The configuration dictionary for the provider.
        :return: A dictionary of parameters for the API call.
        :raises TypeError: If provider_config is not a dictionary.
        """
        if not isinstance(provider_config, dict):
            error = f"Provider config must be a dict, got {type(provider_config)}"
            raise TypeError(error)

        raw_overrides = self.config.llm.models.get(f"{provider}/{model}")
        model_overrides: dict[str, object] = raw_overrides if isinstance(raw_overrides, dict) else {}

        extra_headers = provider_config.get("extra_headers")
        extra_query = provider_config.get("extra_query")
        raw_extra_body = provider_config.get("extra_body")
        extra_body_base = raw_extra_body if isinstance(raw_extra_body, dict) else {}
        extra_body: dict[str, object] = extra_body_base | model_overrides

        return {
            "model": model,
            "messages": messages,
            "stream": True,
            "extra_headers": extra_headers if isinstance(extra_headers, dict) else None,
            "extra_query": extra_query if isinstance(extra_query, dict) else None,
            "extra_body": extra_body,
        }

    async def _generate_response(
        self,
        trigger_msg: discord.Message,
        client: AsyncOpenAI,
        chat_params: dict[str, Any],
    ) -> None:
        """Handle the streaming of the LLM response back to Discord.

        :param trigger_msg: The message that triggered the response.
        :param client: The initialized OpenAI client.
        :param chat_params: The parameters for the API call.
        """
        start_time = time.perf_counter()
        max_len = 4000

        response_msgs: list[discord.Message] = []
        response_contents: list[str] = []

        logger.debug("Logging the request...")
        request_logger.log(chat_params)

        typing_manager = trigger_msg.channel.typing()
        typing_active = False

        try:
            stream = await client.chat.completions.create(**chat_params)
            is_first_chunk = True

            async for chunk in stream:
                if is_first_chunk:
                    logger.debug("Stream started...")
                    await typing_manager.__aenter__()
                    typing_active = True
                    is_first_chunk = False

                content = self._extract_chunk_content(chunk)
                if not content:
                    continue

                self._update_content_buffer(response_contents, content, max_len)
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                logger.debug("Stream finished. Reason: %s", finish_reason)

            await self._finalize_response(trigger_msg, response_msgs, response_contents)

        except Exception:
            logger.exception("Error while generating response!")
            await trigger_msg.channel.send("⚠️ An error occurred.")
        finally:
            if typing_active:
                await typing_manager.__aexit__(None, None, None)

            elapsed = time.perf_counter() - start_time
            logger.info("Response finished in %s seconds!", f"{elapsed:.4f}")
            self._release_node_locks(response_msgs, response_contents)

    def _extract_chunk_content(self, chunk: ChatCompletionChunk) -> str:
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

    def _update_content_buffer(self, buffer: list[str], new_content: str, max_len: int) -> None:
        """Append content to the buffer, creating new chunks if max length is exceeded.

        :param buffer: The list of message content chunks.
        :param new_content: The new text to append.
        :param max_len: The maximum length of a single message.
        """
        if not buffer or len(buffer[-1]) + len(new_content) > max_len:
            buffer.append("")
        buffer[-1] += new_content

    async def _finalize_response(
        self,
        trigger_msg: discord.Message,
        msgs: list[discord.Message],
        contents: list[str],
    ) -> None:
        """Process the final text and update the UI one last time.

        :param trigger_msg: The original trigger message.
        :param msgs: The list of sent response messages.
        :param contents: The raw content chunks.
        """
        full_text = "".join(contents)
        final_text = self._process_final_text(full_text)

        # Split final text into 2000-char chunks for plain text sending
        chunk_size = 2000
        text_chunks = [final_text[i : i + chunk_size] for i in range(0, len(final_text), chunk_size)]
        for chunk in text_chunks:
            view = LayoutView().add_item(TextDisplay(content=chunk))
            await self._send_response_node(trigger_msg, msgs, view=view)

    def _process_final_text(self, text: str) -> str:
        """Sanitize and process the final response text.

        :param text: The raw response text.
        :return: The processed text.
        """
        final_text = text
        if self.config.chat.sanitize_response:
            logger.debug("Sanitizing text...")
            final_text = clean_response(final_text)

        if "<think>" in final_text:
            logger.debug("Removing <think> block...")
            final_text = REGEX_THINK_BLOCK.sub("", final_text).strip()

        return final_text

    async def _send_response_node(
        self,
        trigger_msg: discord.Message,
        msgs: list[discord.Message],
        content: str = "",
        view: LayoutView | None = None,
    ) -> None:
        """Send a new message and initialize its MsgNode with a lock.

        :param trigger_msg: The original trigger message.
        :param msgs: The list of existing response messages (will be appended to).
        :param content: Message content.
        :param view: Message view.
        """
        target = trigger_msg if not msgs else msgs[-1]

        reply_kwargs: dict[str, Any] = {
            "content": content,
            "view": view,
            "silent": True,
        }

        msg = await target.reply(**reply_kwargs)
        msgs.append(msg)

        node = MsgNode(parent_msg=trigger_msg)
        self.msg_nodes[msg.id] = node
        await node.lock.acquire()

    def _release_node_locks(self, msgs: list[discord.Message], contents: list[str]) -> None:
        """Update MsgNodes with final text and release their locks.

        :param msgs: The list of sent messages.
        :param contents: The list of content chunks corresponding to messages.
        """
        full_content = "".join(contents)

        for msg in msgs:
            node = self.msg_nodes.get(msg.id)
            if node:
                node.text = full_content
                if node.lock.locked():
                    node.lock.release()

    def _prune_msg_nodes(self) -> None:
        """Prunes old MsgNodes to prevent memory leaks."""
        if len(self.msg_nodes) <= MAX_MESSAGE_NODES:
            return

        to_remove_count = len(self.msg_nodes) - MAX_MESSAGE_NODES
        logger.debug("Pruning %d old MsgNodes...", to_remove_count)

        sorted_ids = sorted(self.msg_nodes.keys())
        ids_to_delete = sorted_ids[:to_remove_count]

        for msg_id in ids_to_delete:
            self.msg_nodes.pop(msg_id, None)

        logger.debug("Successfully pruned %d nodes!", len(ids_to_delete))
