"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import logging
import time

import discord
import httpx

from .config import PermissionsConfig

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


async def fetch_history(
    message: discord.Message,
    max_messages: int,
    *,
    use_channel_context: bool,
    bot_user: discord.ClientUser,
) -> list[discord.Message]:
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
        message_history = [message]
        message_history.extend([msg async for msg in message.channel.history(limit=max_messages - 1, before=message)])
        return message_history[:max_messages]

    # Fetch a batch of recent messages once to avoid individual API calls for every reply hop.
    local_cache: dict[int, discord.Message] = {msg.id: msg async for msg in message.channel.history(limit=100, before=message)}

    message_history: list[discord.Message] = []
    history_ids: set[int] = set()
    current_msg: discord.Message | None = message

    while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
        history_ids.add(current_msg.id)
        message_history.append(current_msg)

        next_msg = None

        is_public_thread = current_msg.channel.type == discord.ChannelType.public_thread
        parent_is_thread_start = is_public_thread and current_msg.reference is None and isinstance(current_msg.channel, discord.Thread) and current_msg.channel.parent is not None

        if parent_is_thread_start and isinstance(current_msg.channel, discord.Thread):
            thread = current_msg.channel
            next_msg = thread.starter_message or await fetch_referenced_message(current_msg, force_id=thread.id)

        # Try to resolve via Reply Reference
        elif current_msg.reference and current_msg.reference.message_id:
            ref_id = current_msg.reference.message_id
            if ref_id in local_cache:
                next_msg = local_cache[ref_id]
            else:
                next_msg = await fetch_referenced_message(current_msg)

        # Fallback to "Previous Message" logic (if no reply and not mentioned)
        elif bot_user.mention not in current_msg.content:
            # Search local cache for the most recent message by the same author (or bot in DMs)
            potential_prev = [m for m in local_cache.values() if m.id < current_msg.id and m.type in (discord.MessageType.default, discord.MessageType.reply)]
            potential_prev.sort(key=lambda x: x.id, reverse=True)

            for p in potential_prev:
                is_dm = current_msg.channel.type == discord.ChannelType.private
                is_expected = p.author in (bot_user, current_msg.author) if is_dm else p.author == current_msg.author
                if is_expected:
                    next_msg = p
                    break

        current_msg = next_msg

    elapsed_time = time.perf_counter() - start_time
    logger.debug("Fetched history in %s seconds!", f"{elapsed_time:.4f}")

    return message_history


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
    fields = [embed.title, embed.description, getattr(embed.footer, "text", None)]
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
