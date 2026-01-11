import discord
from loguru import logger

from .config_manager import RootConfig, config_manager
from .custom_types import MessageNodeCache

DISCORD_CHAR_LIMIT: int = 2000


class DiscordService:
    """Service for processing Discord messages and managing responses."""

    @property
    def config(self) -> RootConfig:
        """Get the current configuration.

        :return: The root configuration object.
        """
        return config_manager.config

    def __init__(self, message_nodes: MessageNodeCache) -> None:
        """Initialize the message processor.

        :param config: The application configuration.
        :param message_nodes: The shared message nodes mapping.
        """
        self.message_nodes = message_nodes

    async def send_response_chunks(self, trigger_msg: discord.Message, content: str) -> list[discord.Message]:
        """Send response content in chunks to Discord.

        :param trigger_msg: The triggering message.
        :param content: The full response content.
        :return: List of sent messages.
        """
        msgs: list[discord.Message] = []

        if not content:
            logger.warning("Discord service received no text!")
            return msgs

        remaining_text = content

        while remaining_text:
            if len(remaining_text) <= DISCORD_CHAR_LIMIT:
                chunk = remaining_text.strip()
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

        logger.info("Response sent!")

        return msgs
