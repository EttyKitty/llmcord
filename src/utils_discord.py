"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import asyncio
import logging
import time
from base64 import b64encode

import discord
import httpx

from .config_manager import PermissionsConfig, config_manager
from .custom_types import MessageCache, MessageList
from .models import MsgNode

logger = logging.getLogger(__name__)


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
    channel_ids: set[int] = set()
    channel_ids_list: list[int | None] = [msg.channel.id, getattr(msg.channel, "parent_id", None), getattr(msg.channel, "category_id", None)]
    for cid in channel_ids_list:
        if cid is not None:
            channel_ids.add(cid)

    allowed_channels = permissions.channels.allowed_ids
    blocked_channels = permissions.channels.blocked_ids

    # Determine if the channel is "good"
    is_good_channel = allow_dms if is_dm else (not allowed_channels or not channel_ids.isdisjoint(allowed_channels))

    # Determine if the channel is "bad"
    is_bad_channel = not is_good_channel or not channel_ids.isdisjoint(blocked_channels)

    return not is_bad_channel


def is_admin(user_id: int) -> bool:
    """Check if a user has admin permissions.

    :param user_id: The Discord user ID to check.
    :return: True if the user is an admin, False otherwise.
    """
    return user_id in config_manager.config.discord.permissions.users.admin_ids


async def fetch_history(
    message: discord.Message,
    max_messages: int,
    *,
    use_channel_context: bool,
    bot_user: discord.ClientUser,
) -> MessageList:
    """Retrieve message history efficiently using a local cache to avoid rate limits.

    :param message: The trigger message.
    :param max_messages: Maximum number of messages to fetch.
    :param use_channel_context: Whether to use linear channel history.
    :param bot_user: The bot's user object.
    :return: A list of Discord messages.
    """
    logger.debug("Building message history... (Mode: %s)", "Channel" if use_channel_context else "Reply Chain")
    start_time = time.perf_counter()

    if use_channel_context:
        message_history = await _fetch_channel_history(message, max_messages)
    else:
        message_history = await _fetch_reply_chain_history(message, max_messages, bot_user)

    elapsed_time = time.perf_counter() - start_time
    logger.debug("Fetched history in %s seconds!", f"{elapsed_time:.4f}")

    return message_history


async def _fetch_channel_history(
    message: discord.Message,
    max_messages: int,
) -> MessageList:
    """Fetch message history using channel context mode.

    :param message: The trigger message.
    :param max_messages: Maximum number of messages to fetch.
    :return: A list of Discord messages.
    """
    message_history = [message]
    message_history.extend([msg async for msg in message.channel.history(limit=max_messages - 1, before=message)])
    return message_history[:max_messages]


async def _fetch_reply_chain_history(
    message: discord.Message,
    max_messages: int,
    bot_user: discord.ClientUser,
) -> MessageList:
    """Fetch message history using reply chain mode.

    :param message: The trigger message.
    :param max_messages: Maximum number of messages to fetch.
    :param bot_user: The bot's user object.
    :return: A list of Discord messages.
    """
    # Fetch a batch of recent messages once to avoid individual API calls for every reply hop.
    local_cache: MessageCache = {msg.id: msg async for msg in message.channel.history(limit=100, before=message)}

    message_history: MessageList = []
    history_ids: set[int] = set()
    current_msg: discord.Message | None = message

    while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
        history_ids.add(current_msg.id)
        message_history.append(current_msg)

        next_msg = await _get_next_message_in_chain(current_msg, local_cache, bot_user)
        current_msg = next_msg

    return message_history


async def _get_next_message_in_chain(
    current_msg: discord.Message,
    local_cache: MessageCache,
    bot_user: discord.ClientUser,
) -> discord.Message | None:
    """Get the next message in the reply chain.

    :param current_msg: The current message in the chain.
    :param local_cache: Cache of recent messages.
    :param bot_user: The bot's user object.
    :return: The next message in the chain or None.
    """
    # Handle thread starter messages
    if _is_thread_starter_message(current_msg):
        thread = current_msg.channel
        if isinstance(thread, discord.Thread):
            starter_msg = thread.starter_message
            if starter_msg:
                return starter_msg
            return await fetch_referenced_message(current_msg, force_id=thread.id)

    # Try to resolve via Reply Reference
    if current_msg.reference and current_msg.reference.message_id:
        ref_id = current_msg.reference.message_id
        if ref_id in local_cache:
            return local_cache[ref_id]
        return await fetch_referenced_message(current_msg)

    # Fallback to "Previous Message" logic (if no reply and not mentioned)
    if bot_user.mention not in current_msg.content:
        return _find_previous_message_by_author(current_msg, local_cache, bot_user)

    return None


def _is_thread_starter_message(message: discord.Message) -> bool:
    """Check if a message is a thread starter.

    :param message: The message to check.
    :return: True if the message is a thread starter, False otherwise.
    """
    is_public_thread = message.channel.type == discord.ChannelType.public_thread
    return is_public_thread and message.reference is None and isinstance(message.channel, discord.Thread) and message.channel.parent is not None


def _find_previous_message_by_author(
    current_msg: discord.Message,
    local_cache: MessageCache,
    bot_user: discord.ClientUser,
) -> discord.Message | None:
    """Find the most recent previous message by the same author.

    :param current_msg: The current message.
    :param local_cache: Cache of recent messages.
    :param bot_user: The bot's user object.
    :return: The previous message by the same author or None.
    """
    # Search local cache for the most recent message by the same author (or bot in DMs)
    potential_prev = [m for m in local_cache.values() if m.id < current_msg.id and m.type in (discord.MessageType.default, discord.MessageType.reply)]
    potential_prev.sort(key=lambda x: x.id, reverse=True)

    is_dm = current_msg.channel.type == discord.ChannelType.private

    for p in potential_prev:
        is_expected = p.author in (bot_user, current_msg.author) if is_dm else p.author == current_msg.author
        if is_expected:
            return p

    return None


async def fetch_referenced_message(current_msg: discord.Message, force_id: int | None = None) -> discord.Message | None:
    """Fetch a specific referenced message, prioritizing cache.

    :param current_msg: The message containing the reference.
    :param force_id: Optional ID to fetch if the reference is not standard (e.g. thread starter).
    :return: The Discord message or None.
    """
    target_id = force_id or (current_msg.reference.message_id if current_msg.reference else None)
    if not target_id:
        return None

    # Check discord.py's internal cache first
    if not force_id and current_msg.reference and (cached := (current_msg.reference.cached_message or current_msg.reference.resolved)) and isinstance(cached, discord.Message):
        return cached

    try:
        # Use the parent channel if it's a thread starter we're looking for
        channel = current_msg.channel
        if force_id and isinstance(channel, discord.Thread) and isinstance(channel.parent, discord.TextChannel):
            return await channel.parent.fetch_message(target_id)

        return await channel.fetch_message(target_id)
    except (discord.NotFound, discord.HTTPException):
        return None


async def fetch_previous_message(current_msg: discord.Message, bot_user: discord.ClientUser) -> discord.Message | None:
    """Fetch the immediately preceding message in the channel if it matches criteria."""
    async for prev in current_msg.channel.history(before=current_msg, limit=1):
        is_dm = current_msg.channel.type == discord.ChannelType.private
        allowed_types = (discord.MessageType.default, discord.MessageType.reply)

        if prev.type not in allowed_types:
            continue

        is_expected_author = prev.author in (bot_user, current_msg.author) if is_dm else prev.author == current_msg.author

        if is_expected_author:
            return prev
    return None


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


async def init_msg_node(
    msg: discord.Message,
    msg_nodes: dict[int, MsgNode],
    bot_user: discord.ClientUser,
    httpx_client: httpx.AsyncClient,
) -> None:
    """Initialize a MsgNode for a message, processing attachments and text sources.

    :param msg: The Discord message to process.
    :param msg_nodes: The dictionary of processed message nodes.
    :param bot_user: The bot's client user to determine roles.
    :param httpx_client: The shared HTTPX client for downloads.
    """
    node = msg_nodes.setdefault(msg.id, MsgNode())
    if node.text is not None:
        return

    async with node.lock:
        if node.text is not None:
            return

        text_parts = [msg.content.lstrip()] if msg.content.lstrip() else []
        text_parts.extend(get_embed_text(embed) for embed in msg.embeds)
        text_parts.extend(get_component_text(c) for c in msg.components)

        to_download = [a for a in msg.attachments if is_supported_attachment(a)]
        downloads = await asyncio.gather(*(download_attachment(httpx_client, a) for a in to_download))

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
        node.role = "assistant" if msg.author == bot_user else "user"
        node.user_id = msg.author.id if node.role == "user" else None

        author = msg.author
        if msg.guild and not isinstance(author, discord.Member):
            author = msg.guild.get_member(author.id) or author

        node.user_display_name = author.display_name if node.role == "user" else None
        node.has_bad_attachments |= len(msg.attachments) > len(to_download)
