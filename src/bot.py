import asyncio
import logging
import os
import re
import threading
import time
from base64 import b64encode
from datetime import datetime

import discord
import httpx
import tiktoken
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from openai import AsyncOpenAI

from .config import EDITABLE_SETTINGS, RootConfig, config_manager
from .logger import request_logger
from .utils import MsgNode, clean_response

REGEX_USER_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

VISION_MODEL_TAGS = (
    "claude",
    "gemini",
    "gemma",
    "gpt-4",
    "gpt-5",
    "grok-4",
    "llama",
    "llava",
    "mistral",
    "o3",
    "o4",
    "vision",
    "vl",
)
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

TOKENIZER = tiktoken.get_encoding("cl100k_base")


openai_clients: dict[str, AsyncOpenAI] = {}
msg_nodes: dict[int, MsgNode] = {}
last_task_time: float = 0.0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config_manager.config.discord.status_message)[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix="?")

httpx_client = httpx.AsyncClient()


def get_openai_client(provider_config: dict) -> AsyncOpenAI:
    base_url = provider_config["base_url"]

    if base_url not in openai_clients:
        openai_clients[base_url] = AsyncOpenAI(
            base_url=base_url,
            api_key=provider_config.get("api_key", "sk-no-key-required"),
            http_client=httpx_client,
        )
    return openai_clients[base_url]


def is_message_allowed(msg: discord.Message, config: RootConfig) -> bool:
    is_dm = msg.channel.type == discord.ChannelType.private
    permissions = config.discord.permissions

    # Admin check
    if msg.author.id in permissions.users.admin_ids:
        return True

    # User/Role Checks
    role_ids = set(role.id for role in getattr(msg.author, "roles", ()))
    allowed_users = permissions.users.allowed_ids
    blocked_users = permissions.users.blocked_ids
    allowed_roles = permissions.roles.allowed_ids
    blocked_roles = permissions.roles.blocked_ids

    allow_all_users = not allowed_users if is_dm else (not allowed_users and not allowed_roles)

    is_good_user = allow_all_users or msg.author.id in allowed_users or any(id in allowed_roles for id in role_ids)

    is_bad_user = not is_good_user or msg.author.id in blocked_users or any(id in blocked_roles for id in role_ids)

    if is_bad_user:
        return False

    # Channel Checks
    channel_ids = set(
        filter(
            None,
            (
                msg.channel.id,
                getattr(msg.channel, "parent_id", None),
                getattr(msg.channel, "category_id", None),
            ),
        )
    )

    allowed_channels = permissions.channels.allowed_ids
    blocked_channels = permissions.channels.blocked_ids
    allow_dms = config.discord.allow_dms

    allow_all_channels = not allowed_channels
    is_good_channel = allow_dms if is_dm else (allow_all_channels or any(id in allowed_channels for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channels for id in channel_ids)

    if is_bad_channel:
        return False

    return True


def get_embed_text(embed: discord.Embed) -> str:
    """
    Extracts and joins all text fields from a Discord embed.
    """
    fields = [embed.title, embed.description, getattr(embed.footer, "text", None)]
    return "\n".join(filter(None, fields))


def get_component_text(component: discord.Component) -> str:
    """
    Extracts text from a component if it's a text display.
    """
    if component.type == discord.ComponentType.text_display:
        return getattr(component, "content", "")
    return ""


def is_supported_attachment(attachment: discord.Attachment) -> bool:
    """
    Checks if an attachment is a supported text or image type.
    """
    if not attachment.content_type:
        return False
    return any(attachment.content_type.startswith(t) for t in ("text", "image"))


async def download_attachment(attachment: discord.Attachment) -> tuple[discord.Attachment, httpx.Response]:
    """
    Downloads a single attachment and returns it with its response.
    """
    resp = await httpx_client.get(attachment.url)
    return attachment, resp


# --- Config Commands ---
config_group = discord.app_commands.Group(name="config", description="Bot configuration commands")


@config_group.command(name="model", description="Switch the default model (affects all channels)")
async def config_model(interaction: discord.Interaction, model: str) -> None:
    # Permission Check
    if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
        await interaction.response.send_message("[You don't have permission to change the default model.]", ephemeral=True)
        logging.info(f"User {interaction.user.name} tried to switch default model but was denied")
        return

    current_default = config_manager.config.discord.default_model
    if model == current_default:
        await interaction.response.send_message(f"[`default_model` is already: `{current_default}`.]", ephemeral=True)
        return

    config_manager.set_default_model(model)

    # Logging with Channel Name
    channel_name = getattr(interaction.channel, "name", "DM")
    logging.info(f"Admin {interaction.user.name} switched default model to {model} (command sent from #{channel_name})")

    await interaction.response.send_message(f"[`default_model` switched to: `{model}`.]")


@config_model.autocomplete("model")
async def config_model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    default_model = config_manager.config.discord.default_model

    # Highlights the current GLOBAL model
    choices = [Choice(name=f"◉ {default_model} (current default)", value=default_model)] if curr_str.lower() in default_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config_manager.config.llm.models if model != default_model and curr_str.lower() in model.lower()]
    return choices[:25]


@config_group.command(name="channelmodel", description="Switch the model for a specific channel")
async def config_channel_model(interaction: discord.Interaction, model: str, channel: discord.abc.GuildChannel) -> None:
    # Permission Check
    if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
        await interaction.response.send_message("[You don't have permission to change the channel model.]", ephemeral=True)
        logging.info(f"User {interaction.user.name} tried to switch CHANNEL model but was denied.")
        return

    target_channel = channel or interaction.channel

    if target_channel is None:
        return

    # Update the override
    config_manager.set_channel_model(target_channel.id, model)

    if isinstance(target_channel, (discord.TextChannel, discord.VoiceChannel, discord.Thread, discord.StageChannel)):
        channel_mention = target_channel.mention
        channel_name = target_channel.name
    else:
        channel_mention = "Direct Messages"
        channel_name = "Direct Messages"

    logging.info(f"Admin {interaction.user.name} switched channel model to {model} in #{channel_name} ({target_channel.id})")

    await interaction.response.send_message(f"[`channel_model` for {channel_mention} set to: `{model}`.]")


@config_channel_model.autocomplete("model")
async def config_channel_model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    # Determine what is currently active in this channel (Override OR Default)
    default_model = config_manager.config.discord.default_model
    channel_models = config_manager.config.discord.channel_models

    current_active = channel_models.get(interaction.channel_id or 0, default_model)
    is_overridden = interaction.channel_id in channel_models

    status_text = "(current channel)" if is_overridden else "(current default)"

    choices = [Choice(name=f"◉ {current_active} {status_text}", value=current_active)] if curr_str.lower() in current_active.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config_manager.config.llm.models if model != current_active and curr_str.lower() in model.lower()]
    return choices[:25]


@config_group.command(name="set", description="Edit a specific configuration setting")
async def config_set(interaction: discord.Interaction, key: str, value: str) -> None:
    if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
        await interaction.response.send_message("[You don't have permission to edit configuration.]", ephemeral=True)
        return

    if key not in EDITABLE_SETTINGS:
        await interaction.response.send_message(
            f"[Invalid setting: `{key}`. Please select one from the list.]",
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
            f"[Setting `{key}` is not present in the current configuration, so its type cannot be inferred.]",
            ephemeral=True,
        )
        return

    target_type = type(current_value)
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
            f"[Invalid value for `{key}`. Expected type: `{target_type.__name__}`.]",
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


@config_group.command(name="reload", description="Reload config-example.yaml and config.yaml")
async def config_reload(interaction: discord.Interaction) -> None:
    # Permission Check
    if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
        await interaction.response.send_message("[You don't have permission to reload the config.]", ephemeral=True)
        return

    config_manager.load_config()

    logging.info(f"Admin {interaction.user.name} reloaded the configuration")
    await interaction.response.send_message("[Configuration reloaded from disk.]", ephemeral=True)


discord_bot.tree.add_command(config_group)


@discord_bot.event
async def on_ready() -> None:
    if client_id := config_manager.config.discord.client_id:
        logging.info(f"Bot invite URL: https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot")
    logging.info(f"Bot ready. Logged in as {discord_bot.user}")
    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    if not discord_bot.user:
        return

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    logging.info(f"Message received. User: {new_msg.author.name} ID: {new_msg.author.id}")
    start_time = time.perf_counter()

    config = config_manager.config

    if not is_message_allowed(new_msg, config):
        logging.info(f"Message blocked. User: {new_msg.author.name} ID: {new_msg.author.id} Channel: {new_msg.channel.id}")
        return

    # --- Provider Setup ---
    default_model = config.discord.default_model
    channel_models = config.discord.channel_models

    provider_slash_model = channel_models.get(new_msg.channel.id, default_model)
    try:
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
        provider_config = config.llm.providers[provider]
        openai_client = get_openai_client(provider_config)
    except Exception as e:
        logging.error(f"Failed to load provider configuration for {provider_slash_model}: {e}")
        return

    model_parameters = config.llm.models.get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    # Limits
    max_text = config.chat.max_text
    max_images = config.chat.max_images if accept_images else 0
    max_messages = config.chat.max_messages
    max_input_tokens = config.chat.max_input_tokens

    use_plain_responses = config.chat.use_plain_responses
    use_channel_context = config.chat.use_channel_context
    prefix_users = config.chat.prefix_users
    force_reply_chains = config.chat.force_reply_chains
    sanitize_response = config.chat.sanitize_response

    # --- Context Building ---
    messages: list[dict] = []
    user_warnings: set[str] = set()
    message_history: list[discord.Message] = []
    total_tokens = 0

    pre_history_prompt_text = config.prompts.pre_history
    post_history_prompt_text = config.prompts.post_history

    # Reserve tokens for the pre_history prompt
    if pre_history_prompt_text:
        total_tokens += len(TOKENIZER.encode(pre_history_prompt_text))

    # Reserve tokens for the post_history prompt
    if post_history_prompt_text:
        total_tokens += len(TOKENIZER.encode(post_history_prompt_text))

    # Force reply-chain mode if replying to a specific message
    if (use_channel_context and force_reply_chains) and new_msg.reference is not None:
        use_channel_context = False

    logging.debug(f"Building message history... (Mode: {'Channel History' if use_channel_context else 'Reply Chain'})")

    if use_channel_context:
        message_history.append(new_msg)
        async for msg in new_msg.channel.history(limit=max_messages - 1, before=new_msg):
            message_history.append(msg)
        message_history = message_history[:max_messages]
    else:
        current_msg: discord.Message | None = new_msg
        history_ids = set()

        while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
            current_id = current_msg.id
            history_ids.add(current_id)
            message_history.append(current_msg)
            next_msg: discord.Message | None = None

            try:
                # 1. Check for Explicit Reply
                if current_msg.reference and current_msg.reference.message_id:
                    next_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)

                # 2. Check for Thread Start
                if not next_msg:
                    is_thread = current_msg.channel.type == discord.ChannelType.public_thread
                    if is_thread and current_msg.reference is None:
                        # If we are at the top of a thread, the "parent" is the starter message
                        if isinstance(current_msg.channel, discord.Thread) and isinstance(current_msg.channel.parent, discord.abc.Messageable):
                            next_msg = current_msg.channel.starter_message or await current_msg.channel.parent.fetch_message(current_msg.channel.id)

                # 3. Check for "Pseudo-Continuation" (Implicit History)
                if not next_msg and current_msg.reference is None and discord_bot.user.mention not in current_msg.content:
                    async for prev in current_msg.channel.history(before=current_msg, limit=1):
                        allowed_types = (discord.MessageType.default, discord.MessageType.reply)
                        is_valid_type = prev.type in allowed_types

                        if is_dm:
                            is_expected_author = prev.author in (discord_bot.user, current_msg.author)
                        else:
                            is_expected_author = prev.author == current_msg.author

                        if is_valid_type and is_expected_author:
                            next_msg = prev
                        break

                current_msg = next_msg

            except (discord.NotFound, discord.HTTPException):
                logging.exception(f"Failed to fetch parent for message {current_id}")
                break

    # --- Parallel Message Processing ---
    async def init_msg_node(msg: discord.Message):
        node = msg_nodes.setdefault(msg.id, MsgNode())

        if node.text is not None:
            return

        async with node.lock:
            if node.text is not None:
                return

            # 1. Gather all text sources
            text_parts = [msg.content.lstrip()] if msg.content.lstrip() else []
            text_parts.extend(get_embed_text(e) for e in msg.embeds)
            text_parts.extend(get_component_text(c) for c in msg.components)

            # 2. Filter and Download Attachments
            to_download = [a for a in msg.attachments if is_supported_attachment(a)]
            downloads = await asyncio.gather(*(download_attachment(a) for a in to_download))

            # 3. Process Downloads
            node.images = []
            for attachment, resp in downloads:
                content_type = attachment.content_type or ""
                if content_type.startswith("text"):
                    text_parts.append(resp.text)
                elif content_type.startswith("image"):
                    base64_data = b64encode(resp.content).decode()
                    node.images.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{content_type};base64,{base64_data}"},
                        }
                    )

            # 4. Finalize Node
            node.text = "\n".join(filter(None, text_parts))
            node.role = "assistant" if msg.author == discord_bot.user else "user"
            node.user_id = msg.author.id if node.role == "user" else None

            # Resolve display name: Prioritize server nickname over global name
            author = msg.author
            if msg.guild and not isinstance(author, discord.Member):
                author = msg.guild.get_member(author.id) or author

            node.user_display_name = author.display_name if node.role == "user" else None
            node.has_bad_attachments = len(msg.attachments) > len(to_download)

    logging.debug("Initializing message nodes...")

    # Run initialization for all history messages in parallel
    await asyncio.gather(*(init_msg_node(msg) for msg in message_history))

    logging.debug("Building context...")

    # --- Message Processing & Payload Construction ---
    total_images_in_context = 0
    for msg in message_history:
        if len(messages) >= max_messages:
            logging.debug(f"Message limit reached, breaking... ({messages})")
            break

        node = msg_nodes[msg.id]
        if node.text is None:
            logging.debug("Empty message found, skipping...")
            continue

        formatted_text = node.text[:max_text]

        # Apply user ID prefix if enabled and native usernames aren't supported
        if prefix_users and not accept_usernames and node.role == "user" and node.user_display_name is not None:
            formatted_text = f"{node.user_display_name}({node.user_id}): {formatted_text}"

        # --- Token Counting ---
        # 1. Calculate Text Tokens
        text_tokens = len(TOKENIZER.encode(formatted_text))

        # 2. Calculate Image Tokens
        # A safe "buffer" average is 1100 tokens per image.
        image_tokens = 0
        if node.images:
            image_tokens = len(node.images) * 1100

        msg_tokens = text_tokens + image_tokens

        # 3. Check if adding this message exceeds the limit
        if total_tokens + msg_tokens > max_input_tokens:
            if len(messages) == 0:
                pass
            else:
                break

        total_tokens += msg_tokens

        # Construct content payload
        content: str | list[dict[str, str]]
        images_to_add = node.images[:max_images]
        if images_to_add:
            content = ([dict(type="text", text=formatted_text)] if formatted_text else []) + images_to_add
            total_images_in_context += len(images_to_add)
        else:
            content = formatted_text

        if content:
            payload = dict(content=content, role=node.role)
            if accept_usernames and node.user_id and node.user_display_name:
                sanitized_name = REGEX_USER_NAME_SANITIZER.sub("", node.user_display_name)[:64]
                payload["name"] = sanitized_name if sanitized_name else str(node.user_id)
            elif accept_usernames and node.user_id:
                payload["name"] = str(node.user_id)
            messages.append(payload)

        # Warnings
        if len(node.text) > max_text:
            user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
        if total_tokens > max_input_tokens:
            user_warnings.add("⚠️ Context limit reached (older messages trimmed)")
        if len(node.images) > max_images:
            user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
        if node.has_bad_attachments:
            user_warnings.add("⚠️ Unsupported attachments")
        if not use_channel_context and (node.fetch_parent_failed or (node.parent_msg and len(messages) == max_messages)):
            user_warnings.add(f"⚠️ Only using last {len(messages)} message{'s' if len(messages) != 1 else ''}")

    logging.debug(f"Context ready. Messages: {len(messages)}. Images: {total_images_in_context}")

    logging.debug("Replacing placeholders...")

    now = datetime.now().astimezone()
    user_roles = getattr(new_msg.author, "roles", [])
    user_roles_str = ", ".join([role.name for role in user_roles if role.name != "@everyone"]) or "None"
    guild_emojis = getattr(new_msg.guild, "emojis", [])
    guild_emojis_str = ", ".join([str(emoji) for emoji in guild_emojis]) or "None"
    placeholders = {
        "{date}": now.strftime("%B %d %Y"),
        "{time}": now.strftime("%H:%M:%S %Z%z"),
        "{bot_name}": discord_bot.user.display_name,
        "{bot_id}": str(discord_bot.user.id),
        "{model}": model,
        "{provider}": provider,
        "{user_display_name}": new_msg.author.display_name,
        "{user_id}": str(new_msg.author.id),
        "{user_roles}": user_roles_str,
        "{guild_name}": new_msg.guild.name if new_msg.guild else "Direct Messages",
        "{guild_emojis}": guild_emojis_str,
        "{channel_name}": getattr(new_msg.channel, "name", "DM"),
        "{channel_topic}": getattr(new_msg.channel, "topic", "") or "",
        "{channel_nsfw}": str(getattr(new_msg.channel, "nsfw", False)),
    }

    def replace_placeholders(text: str) -> str:
        for key, value in placeholders.items():
            text = text.replace(key, str(value))
        return text.strip()

    logging.debug("Inserting prompts...")

    if pre_history_prompt_text:
        messages.append(dict(role="system", content=replace_placeholders(pre_history_prompt_text)))

    if post_history_prompt_text:
        messages.insert(
            0,
            dict(role="system", content=replace_placeholders(post_history_prompt_text)),
        )

    # --- Response Generation ---
    curr_content = finish_reason = None
    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
    )

    # Log the request
    logging.debug("Logging the request...")
    request_logger.log(openai_kwargs)

    elapsed_time = time.perf_counter() - start_time
    logging.info(f"Request prepared in {elapsed_time:.4f} seconds!")
    start_time = time.perf_counter()

    if use_plain_responses:
        max_message_length = 4000 # Discord message length limit
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed()
        for warning in sorted(user_warnings):
            embed.add_field(name=warning, value="", inline=False)

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        logging.debug(f"Sending request ({total_tokens} tokens) to {provider} ({model})...")
        async with new_msg.channel.typing():
            first_chunk_received = False

            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if not first_chunk_received:
                    logging.debug("Stream started...")
                    first_chunk_received = True

                if finish_reason is not None:
                    logging.debug(f"Stream finished. Reason: {finish_reason}")
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""

                # Handle potential list content (Mistral/Multimodal quirks)
                delta = choice.delta
                if isinstance(delta.content, list):
                    logging.debug("Multimodal content detected...")
                    curr_content = ""
                    for part in delta.content:
                        if isinstance(part, str):
                            curr_content += part
                        elif isinstance(part, dict):
                            curr_content += part.get("text", "")
                        elif hasattr(part, "text"):
                            curr_content += part.text
                else:
                    curr_content = delta.content or ""

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.colour = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            full_response = "".join(response_contents)
            original_response = full_response

            # --- Post-Processing ---
            if sanitize_response:
                logging.debug("Sanitizing...")
                full_response = clean_response(full_response)

            if "<think>" in full_response:
                logging.debug("Removing <think> block...")
                full_response = REGEX_THINK_BLOCK.sub("", full_response).strip()

            if full_response != original_response:
                response_contents = []
                if full_response:
                    for i in range(0, len(full_response), max_message_length):
                        chunk = full_response[i : i + max_message_length]
                        response_contents.append(chunk)

                if not use_plain_responses and response_msgs and response_contents:
                    embed.description = response_contents[-1]
                    embed.colour = EMBED_COLOR_COMPLETE
                    await response_msgs[-1].edit(embed=embed)

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")
        await new_msg.channel.send("⚠️ An error occurred.")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    elapsed_time = time.perf_counter() - start_time
    logging.info(f"Response finished in {elapsed_time:.4f} seconds")

    # Prune old MsgNodes
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        logging.debug("Pruning old MsgNodes...")
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


def console_listener() -> None:
    """Listens for console commands in a background thread."""
    while True:
        try:
            command = input()
            if command.strip().lower() == "reload":
                logging.info("Reloading...")
                # Exit with code 2 to signal the batch script to restart
                os._exit(2)
            elif command.strip().lower() in ("exit", "stop", "quit"):
                logging.info("Stopping...")
                os._exit(0)
            else:
                print(f"Unknown command: {command}")
        except EOFError:
            break


threading.Thread(target=console_listener, daemon=True).start()


async def main() -> None:
    await discord_bot.start(config_manager.config.discord.bot_token)
