"""Discord utility functions for message history and permissions.

This module provides utility functions for fetching message history,
checking admin permissions, and managing message caches.
"""

import asyncio
import logging
import time

import discord

from .config_manager import config_manager
from .custom_types import MessageCache, MessageList

logger = logging.getLogger(__name__)

DISCORD_API_TIMEOUT = 30.0


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
    logger.debug("Fetching message history... (Mode: %s)", "Channel" if use_channel_context else "Reply Chain")
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

    async def collect_history() -> MessageList:
        return [msg async for msg in message.channel.history(limit=max_messages - 1, before=message)]

    try:
        history = await asyncio.wait_for(collect_history(), timeout=DISCORD_API_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Timeout fetching channel history for message %s", message.id)
        history = []

    message_history = [message]
    message_history.extend(history)
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
    try:
        local_cache: MessageCache = await asyncio.wait_for(collect_cache(message), timeout=DISCORD_API_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Timeout fetching reply chain history for message %s", message.id)
        local_cache = {}

    message_history: MessageList = []
    history_ids: set[int] = set()
    current_msg: discord.Message | None = message

    while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
        history_ids.add(current_msg.id)
        message_history.append(current_msg)

        next_msg = await _get_next_message_in_chain(current_msg, local_cache, bot_user)
        current_msg = next_msg

    return message_history


async def collect_cache(message: discord.Message) -> MessageCache:
    """Collect a cache of recent messages before the given message.

    :param message: The Discord message to collect history before.
    :return: A dictionary mapping message IDs to Message objects.
    """
    return {msg.id: msg async for msg in message.channel.history(limit=100, before=message)}


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
            try:
                return await asyncio.wait_for(channel.parent.fetch_message(target_id), timeout=DISCORD_API_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Timeout fetching referenced message %s from parent channel", target_id)
                return None

        try:
            return await asyncio.wait_for(channel.fetch_message(target_id), timeout=DISCORD_API_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching referenced message %s", target_id)
            return None
    except (discord.NotFound, discord.HTTPException):
        return None
