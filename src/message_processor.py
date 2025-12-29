"""Message Processor for handling Discord message operations.

This module provides a service for processing Discord messages,
including validation, response sending, and node management.
"""

import logging
from typing import Any

import discord

from .config_manager import RootConfig
from .custom_types import MsgNode
from .discord_utils import is_message_allowed

logger = logging.getLogger(__name__)
DISCORD_CHAR_LIMIT: int = 2000
MAX_MESSAGE_NODES: int = 500


class MessageProcessor:
    """Service for processing Discord messages and managing responses."""

    def __init__(self, config: RootConfig, bot_user: discord.ClientUser) -> None:
        """Initialize the message processor.

        :param config: The application configuration.
        :param bot_user: The bot's user object.
        """
        self.config = config
        self.bot_user = bot_user
        self.msg_nodes: dict[int, Any] = {}

    def valid_trigger_message(self, message: discord.Message) -> bool:
        """Determine if a message should be processed.

        :param message: The Discord message to check.
        :return: True if the message should be processed.
        """
        if message.author.bot:
            logger.debug("Skipping message because author is a bot. author=%s", getattr(message.author, "name", None))
            return False

        is_dm = message.channel.type == discord.ChannelType.private
        is_mentioned = self.bot_user in message.mentions

        if not (is_dm or is_mentioned):
            logger.debug("Skipping message: not DM and not mentioned. is_dm=%s is_mentioned=%s", is_dm, is_mentioned)
            return False

        if not is_message_allowed(message, self.config.discord.permissions, allow_dms=self.config.discord.allow_dms):
            logger.info("Message blocked. User: %s ID: %d", message.author.name, message.author.id)
            return False

        return True

    async def send_response_chunks(self, trigger_msg: discord.Message, content: str) -> list[discord.Message]:
        """Send response content in chunks to Discord.

        :param trigger_msg: The triggering message.
        :param content: The full response content.
        :return: List of sent messages.
        """
        msgs: list[discord.Message] = []
        remaining_text = content

        while remaining_text:
            if len(remaining_text) <= DISCORD_CHAR_LIMIT:
                chunk = remaining_text
                remaining_text = ""
            else:
                # Try to split at the last newline within the limit
                split_index = remaining_text.rfind("\n", 0, DISCORD_CHAR_LIMIT)
                # If no newline, try to split at the last space
                if split_index == -1:
                    split_index = remaining_text.rfind(" ", 0, DISCORD_CHAR_LIMIT)
                # If no space, hard split at the limit
                if split_index == -1:
                    split_index = DISCORD_CHAR_LIMIT

                chunk = remaining_text[:split_index].strip()
                remaining_text = remaining_text[split_index:].strip()

            if not chunk:
                continue

            target: discord.Message = msgs[-1] if msgs else trigger_msg
            new_msg: discord.Message = await target.reply(content=chunk, silent=True)
            msgs.append(new_msg)

            # Create and lock node for the new message
            node = MsgNode()
            await node.lock.acquire()
            self.msg_nodes[new_msg.id] = node

        return msgs

    def release_node_locks(self, msgs: list[discord.Message], full_content: str) -> None:
        """Update and release locks on message nodes.

        :param msgs: The messages that were sent.
        :param full_content: The complete response content.
        """
        for msg in msgs:
            node = self.msg_nodes.get(msg.id)
            if node:
                node.text = full_content
                if hasattr(node.lock, "locked") and node.lock.locked():
                    node.lock.release()

    def prune_msg_nodes(self) -> None:
        """Prune old message nodes to prevent memory leaks."""
        if len(self.msg_nodes) <= MAX_MESSAGE_NODES:
            return

        to_remove_count = len(self.msg_nodes) - MAX_MESSAGE_NODES
        logger.debug("Pruning %d old MsgNodes...", to_remove_count)

        sorted_ids = sorted(self.msg_nodes.keys())
        ids_to_delete = sorted_ids[:to_remove_count]

        for msg_id in ids_to_delete:
            self.msg_nodes.pop(msg_id, None)

        logger.debug("Successfully pruned %d nodes!", len(ids_to_delete))
