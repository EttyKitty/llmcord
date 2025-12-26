"""Data models and text processing utilities.

This module defines the structure for message nodes used in conversation
chains and provides utility functions for sanitizing text input/output.
"""

import logging

import discord

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


async def fetch_history(message: discord.Message, max_messages: int, *, use_channel_context: bool, bot_user: discord.ClientUser) -> list[discord.Message]:
    """Retrieve message history either via channel history or reply chain.

    :param message: The trigger message.
    :param max_messages: Maximum number of messages to fetch.
    :param use_channel_context: Whether to use linear channel history.
    :param bot_user: The bot's user object to check for mentions.
    :return: A list of Discord messages.
    """
    logger.debug("Building message history... (Mode: %s)", "Channel" if use_channel_context else "Reply Chain")

    if use_channel_context:
        message_history = [message]
        message_history.extend([msg async for msg in message.channel.history(limit=max_messages - 1, before=message)])
        return message_history[:max_messages]

    # Reply Chain Logic
    message_history: list[discord.Message] = []
    history_ids: set[int] = set()
    current_msg: discord.Message | None = message

    while current_msg and len(message_history) < max_messages and current_msg.id not in history_ids:
        history_ids.add(current_msg.id)
        message_history.append(current_msg)

        # Resolve next message in chain
        next_msg = None
        if current_msg.reference and current_msg.reference.message_id:
            next_msg = await fetch_referenced_message(current_msg)
        elif bot_user.mention not in current_msg.content:
            next_msg = await fetch_previous_message(current_msg, bot_user)

        current_msg = next_msg

    return message_history


async def fetch_referenced_message(current_msg: discord.Message) -> discord.Message | None:
    """Fetch the message referenced by the current message."""
    if not current_msg.reference or not current_msg.reference.message_id:
        return None

    try:
        next_msg = await current_msg.channel.fetch_message(current_msg.reference.message_id)

        # Handle Thread starter messages
        if isinstance(current_msg.channel, discord.Thread) and isinstance(current_msg.channel.parent, discord.abc.Messageable) and next_msg.id == current_msg.channel.id:
            return current_msg.channel.starter_message or await current_msg.channel.parent.fetch_message(current_msg.channel.id)
    except (discord.NotFound, discord.HTTPException):
        logger.exception("Failed to fetch parent for message %d", current_msg.reference.message_id)
        return None
    else:
        return next_msg


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
