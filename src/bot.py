import asyncio
import logging
import os
import re
import threading
import time
from base64 import b64encode
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict

import discord
import httpx
import tiktoken
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from .config import EDITABLE_SETTINGS, RootConfig, config_manager
from .logger import request_logger
from .utils import MsgNode, clean_response

# Constants
REGEX_USER_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500
TOKENIZER = tiktoken.get_encoding("cl100k_base")


class LLMCordBot(commands.Bot):
    """The main bot class for llmcord, encapsulating state and logic.
    """

    @property
    def config(self) -> RootConfig:
        return config_manager.config

    # I'm too stupid to find a different solution to strict type checking in local scopes
    @property
    def safe_user(self) -> discord.ClientUser:
        if self.user is None:
            raise RuntimeError("Bot user not initialized!")
        return self.user

    def __init__(self, intents: discord.Intents, activity: discord.CustomActivity) -> None:
        super().__init__(command_prefix="?", intents=intents, activity=activity)
        self.openai_clients: dict[str, AsyncOpenAI] = {}
        self.msg_nodes: dict[int, MsgNode] = {}
        self.last_task_time: float = 0.0
        self.httpx_client = httpx.AsyncClient()

        # Register slash commands
        self._setup_commands()

    async def close(self) -> None:
        """Cleanup resources before shutting down."""
        await self.httpx_client.aclose()
        await super().close()

    async def setup_hook(self) -> None:
        """Set up internal discord.py hook for asynchronous setup."""
        await self.tree.sync()
        if client_id := self.config.discord.client_id:
            logging.info(f"Bot invite URL: https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot")
        logging.info(f"Bot ready. Logged in as {self.safe_user}")

    def get_openai_client(self, provider_config: dict[str, Any]) -> AsyncOpenAI:
        """Retrieve or initialize an OpenAI-compatible client for a specific provider.

        :param provider_config: The configuration dictionary for the provider.
        :return: An AsyncOpenAI client instance.
        """
        base_url = provider_config["base_url"]
        if base_url not in self.openai_clients:
            self.openai_clients[base_url] = AsyncOpenAI(
                base_url=base_url,
                api_key=provider_config.get("api_key", "sk-no-key-required"),
                http_client=self.httpx_client,
            )
        return self.openai_clients[base_url]

    def _is_message_allowed(self, msg: discord.Message) -> bool:
        """Check if the message author and channel are allowed based on configuration.

        :param msg: The Discord message to check.
        :return: True if allowed, False otherwise.
        """
        is_dm = msg.channel.type == discord.ChannelType.private
        permissions = self.config.discord.permissions

        if msg.author.id in permissions.users.admin_ids:
            return True

        role_ids = {role.id for role in getattr(msg.author, "roles", ())}
        allowed_users = permissions.users.allowed_ids
        blocked_users = permissions.users.blocked_ids
        allowed_roles = permissions.roles.allowed_ids
        blocked_roles = permissions.roles.blocked_ids

        allow_all_users = not allowed_users if is_dm else (not allowed_users and not allowed_roles)
        is_good_user = allow_all_users or msg.author.id in allowed_users or any(rid in allowed_roles for rid in role_ids)
        is_bad_user = not is_good_user or msg.author.id in blocked_users or any(rid in blocked_roles for rid in role_ids)

        if is_bad_user:
            return False

        channel_ids = set(
            filter(
                None,
                (
                    msg.channel.id,
                    getattr(msg.channel, "parent_id", None),
                    getattr(msg.channel, "category_id", None),
                ),
            ),
        )
        allowed_channels = permissions.channels.allowed_ids
        blocked_channels = permissions.channels.blocked_ids
        allow_dms = self.config.discord.allow_dms

        is_good_channel = allow_dms if is_dm else (not allowed_channels or any(cid in allowed_channels for cid in channel_ids))
        is_bad_channel = not is_good_channel or any(cid in blocked_channels for cid in channel_ids)

        return not is_bad_channel

    async def _fetch_history(self, message: discord.Message, max_messages: int, use_channel_context: bool) -> list[discord.Message]:
        """Retrieve message message_history either via channel message_history or reply chain.

        :param message: The trigger message.
        :param max_messages: Maximum number of messages to fetch.
        :param use_channel_context: Whether to use linear channel message_history.
        :return: A list of Discord messages.
        """
        message_history: list[discord.Message] = []
        history_ids: set[int] = set()
        current_msg: discord.Message | None = message

        logging.debug(f"Building message history... (Mode: {'Channel History' if use_channel_context else 'Reply Chain'})")

        if use_channel_context:
            message_history = [message]
            async for msg in message.channel.history(limit=max_messages - 1, before=message):
                message_history.append(msg)
            return message_history[:max_messages]
        while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
            history_ids.add(current_msg.id)
            message_history.append(current_msg)
            next_msg = None

            if current_msg.reference and current_msg.reference.message_id:
                try:
                    next_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)

                    if isinstance(current_msg.channel, discord.Thread) and isinstance(current_msg.channel.parent, discord.abc.Messageable):
                        next_msg = current_msg.channel.starter_message or await current_msg.channel.parent.fetch_message(current_msg.channel.id)

                    current_msg = next_msg
                except (discord.NotFound, discord.HTTPException):
                    logging.exception(f"Failed to fetch parent for message {current_msg.reference.message_id}")
                    break

            elif self.safe_user.mention not in current_msg.content:
                async for prev in current_msg.channel.history(before=current_msg, limit=1):
                    is_dm = current_msg.channel.type == discord.ChannelType.private
                    allowed_types = (discord.MessageType.default, discord.MessageType.reply)
                    is_valid_type = prev.type in allowed_types

                    is_expected_author = prev.author in (self.safe_user, current_msg.author) if is_dm else prev.author == current_msg.author

                    if is_valid_type and is_expected_author:
                        next_msg = prev
                    break
        return message_history

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
            text_parts.extend(self._get_embed_text(e) for e in msg.embeds)
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
            return attachment, resp
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logging.warning(f"Failed to download attachment {attachment.filename} ({attachment.url}): {e}")
            return attachment, None

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

    async def on_message(self, message: discord.Message) -> None:
        """Event handler for new messages. Orchestrates the response flow."""
        if not self.safe_user or message.author.bot:
            return

        is_dm = message.channel.type == discord.ChannelType.private
        if not is_dm and self.safe_user not in message.mentions:
            return

        if not self._is_message_allowed(message):
            logging.info(f"Message blocked. User: {message.author.name} ID: {message.author.id}")
            return

        start_time = time.perf_counter()
        config = self.config

        # 1. Configuration Resolution
        provider_slash_model = config.chat.channel_models.get(message.channel.id, config.chat.default_model)
        try:
            provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
            provider_config = config.llm.providers[provider]
            openai_client = self.get_openai_client(provider_config)
        except (ValueError, KeyError):
            logging.exception(f"Failed to load provider configuration for {provider_slash_model}!")
            return

        accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
        accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

        # 2. Context Building
        use_channel_context = config.chat.use_channel_context
        if use_channel_context and config.chat.force_reply_chains and message.reference:
            use_channel_context = False

        logging.debug("Initializing message nodes...")
        message_history = await self._fetch_history(message, config.chat.max_messages, use_channel_context)
        await asyncio.gather(*(self._init_msg_node(m) for m in message_history))

        logging.debug("Building context...")
        messages_payload: list[ChatCompletionMessageParam] = []
        user_warnings: set[str] = set()
        total_tokens = 0
        total_images = 0

        logging.debug("Replacing placeholders...")
        pre_history = self._replace_placeholders(config.prompts.pre_history, message, model, provider)
        post_history = self._replace_placeholders(config.prompts.post_history, message, model, provider)
        if pre_history:
            total_tokens += len(TOKENIZER.encode(pre_history))
        if post_history:
            total_tokens += len(TOKENIZER.encode(post_history))

        for msg in message_history:
            if len(messages_payload) >= config.chat.max_messages:
                logging.debug(f"Message limit reached, breaking... ({config.chat.max_messages})")
                break

            node = self.msg_nodes[msg.id]

            if node.text is None:
                logging.debug("Empty message found, skipping...")
                continue

            formatted_text = node.text[: config.chat.max_text]

            if config.chat.prefix_users and not accept_usernames and node.role == "user" and node.user_display_name:
                formatted_text = f"{node.user_display_name}({node.user_id}): {formatted_text}"

            text_tokens = len(TOKENIZER.encode(formatted_text))
            image_tokens = len(node.images[: config.chat.max_images]) * 1100 if accept_images else 0
            msg_tokens = text_tokens + image_tokens

            if total_tokens + msg_tokens > config.chat.max_input_tokens and messages_payload:
                user_warnings.add("⚠️ Context limit reached (older messages trimmed)")
                break

            total_tokens += msg_tokens

            content: str | list[ChatCompletionContentPartParam]
            images_to_add = node.images[: config.chat.max_images] if accept_images else []

            if images_to_add:
                parts: list[ChatCompletionContentPartParam] = []

                if formatted_text:
                    text_part: ChatCompletionContentPartTextParam = {"type": "text", "text": formatted_text}
                    parts.append(text_part)

                total_images = len(images_to_add)
                parts.extend(images_to_add)
                content = parts
            else:
                content = formatted_text

            if content:
                if node.role == "user":
                    user_payload: ChatCompletionUserMessageParam = {"role": "user", "content": content}
                    if accept_usernames and node.user_id:
                        sanitized_name = REGEX_USER_NAME_SANITIZER.sub("", node.user_display_name or "")[:64]
                        user_payload["name"] = sanitized_name or str(node.user_id)
                    messages_payload.append(user_payload)
                elif node.role == "assistant":
                    messages_payload.append({"role": "assistant", "content": formatted_text})

            if len(node.text or "") > config.chat.max_text:
                user_warnings.add(f"⚠️ Max {config.chat.max_text:,} characters per message")
        logging.debug(f"Context ready. Messages: {len(messages_payload)}. Images: {total_images}")

        logging.debug("Inserting prompts...")
        if pre_history:
            messages_payload.append({"role": "system", "content": pre_history})
        if post_history:
            messages_payload.insert(0, {"role": "system", "content": post_history})

        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Request prepared in {elapsed_time:.4f} seconds!")

        # 4. Generation
        await self._generate_response(message, openai_client, provider, model, messages_payload[::-1], provider_config, user_warnings)

        # 5. Cleanup
        self._prune_msg_nodes()

    async def _generate_response(
        self,
        trigger_msg: discord.Message,
        client: AsyncOpenAI,
        provider: str,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        provider_config: dict[str, Any],
        warnings: set[str],
    ) -> None:
        """Handle the streaming of the LLM response back to Discord."""
        start_time = time.perf_counter()

        use_plain = self.config.chat.use_plain_responses
        max_message_length = 4000 if use_plain else (4096 - len(STREAMING_INDICATOR))
        response_msgs: list[discord.Message] = []
        response_contents: list[str] = []

        class ChatParams(TypedDict):
            model: str
            messages: Sequence[ChatCompletionMessageParam]
            stream: Literal[True]
            extra_headers: NotRequired[dict[str, str] | None]
            extra_query: NotRequired[dict[str, object] | None]
            extra_body: NotRequired[dict[str, object] | None]

        model_overrides: dict[str, object] = self.config.llm.models.get(f"{provider}/{model}") or {}
        extra_headers = provider_config.get("extra_headers")
        extra_query = provider_config.get("extra_query")
        extra_body: dict[str, object] = (provider_config.get("extra_body") or {}) | model_overrides

        openai_params: ChatParams = {
            "model": model,
            "messages": messages,
            "stream": True,
            "extra_headers": extra_headers if isinstance(extra_headers, dict) else None,
            "extra_query": extra_query if isinstance(extra_query, dict) else None,
            "extra_body": extra_body,
        }

        logging.debug("Logging the request...")
        request_logger.log(openai_params)

        embed = None if use_plain else discord.Embed()
        if embed:
            for warning in sorted(warnings):
                embed.add_field(name=warning, value="", inline=False)

        async def reply_helper(content: str = "", embed: discord.Embed | None = None, view: LayoutView | None = None) -> None:
            target = trigger_msg if not response_msgs else response_msgs[-1]

            reply_kwargs: dict[str, Any] = {
                "content": content,
                "embed": embed,
                "view": view,
                "silent": True,
            }

            msg = await target.reply(**reply_kwargs)
            response_msgs.append(msg)
            self.msg_nodes[msg.id] = MsgNode(parent_msg=trigger_msg)
            await self.msg_nodes[msg.id].lock.acquire()

        try:
            async with trigger_msg.channel.typing():
                first_chunk_received = False
                full_stream_content = ""

                async for chunk in await client.chat.completions.create(**openai_params):
                    if not first_chunk_received:
                        logging.debug("Stream started...")
                        first_chunk_received = True

                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta_content = choice.delta.content or ""

                    # Handle potential list content (Mistral/Multimodal quirks)
                    if isinstance(delta_content, list):
                        logging.debug("Multimodal content detected...")
                        content_parts = delta_content
                        delta_content = ""
                        for part in content_parts:
                            if isinstance(part, str):
                                delta_content += part
                            elif isinstance(part, dict):
                                delta_content += part.get("text", "")
                            elif hasattr(part, "text"):
                                delta_content += part.text

                    full_stream_content += delta_content
                    if not full_stream_content:
                        continue

                    if not response_contents or len(response_contents[-1] + delta_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += delta_content

                    finish_reason = choice.finish_reason
                    if finish_reason is not None:
                        logging.debug(f"Stream finished. Reason: {finish_reason}")

                    if embed:
                        now_ts = datetime.now(timezone.utc).timestamp()
                        time_delta = now_ts - self.last_task_time
                        is_final = finish_reason is not None
                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS

                        if len(response_contents) > len(response_msgs) or ready_to_edit or is_final:
                            # The type checker now knows 'embed' is discord.Embed
                            embed.description = response_contents[-1] + (STREAMING_INDICATOR if not is_final else "")
                            embed.colour = EMBED_COLOR_COMPLETE if is_final else EMBED_COLOR_INCOMPLETE

                            if len(response_contents) > len(response_msgs):
                                await reply_helper(embed=embed)
                            else:
                                await response_msgs[-1].edit(embed=embed)
                            self.last_task_time = now_ts

                # Post-processing
                final_text = "".join(response_contents)

                if self.config.chat.sanitize_response:
                    logging.debug("Sanitizing text...")
                    final_text = clean_response(final_text)

                if "<think>" in final_text:
                    logging.debug("Removing <think> block...")
                    final_text = REGEX_THINK_BLOCK.sub("", final_text).strip()

                if use_plain:
                    for chunk in [final_text[i : i + 2000] for i in range(0, len(final_text), 2000)]:
                        await reply_helper(view=LayoutView().add_item(TextDisplay(content=chunk)))
                elif embed and response_msgs:
                    embed.description = final_text
                    embed.colour = EMBED_COLOR_COMPLETE
                    await response_msgs[-1].edit(embed=embed)

        except Exception:
            logging.exception("Error while generating response!")
            await trigger_msg.channel.send("⚠️ An error occurred.")
        finally:
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"Response finished in {elapsed_time:.4f} seconds!")
            for response_msg in response_msgs:
                node = self.msg_nodes[response_msg.id]
                node.text = "".join(response_contents)
                if node.lock.locked():
                    node.lock.release()

    def _prune_msg_nodes(self) -> None:
        """Prunes old MsgNodes to prevent memory leaks."""
        if len(self.msg_nodes) > MAX_MESSAGE_NODES:
            logging.debug("Pruning old MsgNodes...")

            sorted_ids = sorted(self.msg_nodes.keys())
            for msg_id in sorted_ids[: len(self.msg_nodes) - MAX_MESSAGE_NODES]:
                self.msg_nodes.pop(msg_id, None)

    def _setup_commands(self) -> None:
        """Set up the slash command tree."""
        config_group = app_commands.Group(name="config", description="Bot configuration commands")

        @config_group.command(name="model", description="Switch the default model")
        async def config_model(interaction: discord.Interaction, model: str) -> None:
            if interaction.user.id not in self.config.discord.permissions.users.admin_ids:
                await interaction.response.send_message("Permission denied.", ephemeral=True)
                return

            config_manager.set_default_model(model)

            channel_name = getattr(interaction.channel, "name", "DM")

            logging.info(f"Admin {interaction.user.name} switched default model to {model} (command sent from #{channel_name})")

            await interaction.response.send_message(f"[Default model set to `{model}`.]")

        @config_model.autocomplete("model")
        async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[Choice[str]]:
            default_model = self.config.chat.default_model
            models = self.config.llm.models

            choices = [Choice(name=f"◉ {default_model} (current default)", value=default_model)] if current.lower() in default_model.lower() else []
            choices += [Choice(name=f"○ {model}", value=model) for model in models if model != default_model and current.lower() in model.lower()]

            return choices[:25]

        @config_group.command(name="reload", description="Reload config from disk")
        async def config_reload(interaction: discord.Interaction) -> None:
            if interaction.user.id not in self.config.discord.permissions.users.admin_ids:
                await interaction.response.send_message("Permission denied.", ephemeral=True)
                return

            config_manager.load_config()

            logging.info(f"Admin {interaction.user.name} reloaded the configuration")

            await interaction.response.send_message("Configuration reloaded from disk.", ephemeral=True)

        @config_group.command(name="channelmodel", description="Switch the model for a specific channel")
        async def config_channel_model(interaction: discord.Interaction, model: str, channel: discord.abc.GuildChannel | None = None) -> None:
            if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
                await interaction.response.send_message("You don't have permission to change the channel model.", ephemeral=True)
                return

            target_channel = channel or interaction.channel
            if target_channel is None:
                return

            config_manager.set_channel_model(target_channel.id, model)

            if isinstance(target_channel, (discord.TextChannel, discord.VoiceChannel, discord.Thread, discord.StageChannel)):
                channel_mention = target_channel.mention
                channel_name = target_channel.name
            else:
                channel_mention = "Direct Messages"
                channel_name = channel_mention

            logging.info(f"Admin {interaction.user.name} switched channel model to {model} in #{channel_name} ({target_channel.id})")

            await interaction.response.send_message(f"[`channel_model` for {channel_mention} set to: `{model}`.]")

        @config_channel_model.autocomplete("model")
        async def config_channel_model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
            default_model = config_manager.config.chat.default_model
            channel_models = config_manager.config.chat.channel_models

            current_active = channel_models.get(interaction.channel_id or 0, default_model)
            is_overridden = interaction.channel_id in channel_models

            status_text = "(current channel)" if is_overridden else "(current default)"

            choices = [Choice(name=f"◉ {current_active} {status_text}", value=current_active)] if curr_str.lower() in current_active.lower() else []
            choices += [Choice(name=f"○ {model}", value=model) for model in config_manager.config.llm.models if model != current_active and curr_str.lower() in model.lower()]
            return choices[:25]

        @config_group.command(name="set", description="Edit a specific configuration setting")
        async def config_set(interaction: discord.Interaction, key: str, value: str) -> None:
            if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
                await interaction.response.send_message("You don't have permission to edit configuration.", ephemeral=True)
                return

            if key not in EDITABLE_SETTINGS:
                await interaction.response.send_message(
                    f"Invalid setting: `{key}`. Please select one from the list.",
                    ephemeral=True,
                )
                return

            try:
                current_value = config_manager.get_setting_value(key)
            except AttributeError:
                await interaction.response.send_message(
                    f"[Setting `{key}` not found in configuration structure.]",
                    ephemeral=True,
                )
                return

            target_type = type(current_value)

            if current_value is None:
                await interaction.response.send_message(
                    f"Setting `{key}` is not present in the current configuration, so its type cannot be inferred.",
                    ephemeral=True,
                )
                return

            parsed_value: int | bool | float | str

            try:
                if target_type is bool:
                    if value.lower() in ("true", "1", "yes", "on"):
                        parsed_value = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        parsed_value = False
                    else:
                        raise ValueError("Invalid boolean")
                elif target_type is int:
                    parsed_value = int(value)
                elif target_type is float:
                    parsed_value = float(value)
                else:
                    parsed_value = value

            except ValueError:
                await interaction.response.send_message(
                    f"Invalid value for `{key}`. Expected type: `{target_type.__name__}`.",
                    ephemeral=True,
                )
                return

            # Apply update
            config_manager.update_setting(key, parsed_value)

            logging.info(f"Admin {interaction.user.name} changed config {key} to {parsed_value}")
            await interaction.response.send_message(f"[Configuration updated: `{key}` set to `{parsed_value}`.]")

        @config_set.autocomplete("key")
        async def config_set_key_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
            return [Choice(name=key, value=key) for key in EDITABLE_SETTINGS if curr_str.lower() in key.lower()][:25]

        self.tree.add_command(config_group)


def console_listener() -> None:
    """Listen for console commands."""
    while True:
        try:
            command = input().strip().lower()
            if command == "reload":
                os._exit(2)
            elif command in ("exit", "stop", "quit"):
                os._exit(0)
            else:
                print(f"Unknown command: {command}")
        except EOFError:
            break


async def main() -> None:
    """Main entry point."""
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
