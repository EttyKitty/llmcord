"""LLMCord Bot Main Module.

This module contains the core implementation of the LLMCord Discord bot.
It handles the bot's lifecycle, event processing, message validation,
context construction, and interaction with various LLM providers using
OpenAI-compatible APIs.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Final

import aiohttp
import discord
import httpx
import litellm
from discord.ext import commands

from .commands_manager import setup_commands
from .config_manager import RootConfig, config_manager
from .llm_service import LLMService
from .message_processor import MessageProcessor

MAX_MESSAGE_NODES: Final[int] = 500
DISCORD_CHAR_LIMIT: Final[int] = 2000
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")


logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True


class LLMCordBot(commands.Bot):
    """The main bot class for llmcord, encapsulating state and logic.

    This class manages the bot's lifecycle, message processing, conversation
    context building, and interaction with LLM providers via OpenAI-compatible APIs.

    :ivar llm_service: Service for LLM operations.
    :ivar message_processor: Service for message processing.
    :ivar httpx_client: Shared HTTP client for external requests.
    :ivar exit_code: Exit code.
    """

    @property
    def config(self) -> RootConfig:
        """Get the current configuration.

        :return: The root configuration object.
        """
        return config_manager.config

    @property
    def safe_user(self) -> discord.ClientUser:
        """Get the bot's user object safely.

        :return: The bot's client user.
        :raises RuntimeError: If the bot is not logged in.
        """
        if self.user is None:
            msg = "Bot user is not available - bot may not be logged in"
            raise RuntimeError(msg)
        return self.user

    def __init__(self, intents: discord.Intents, activity: discord.CustomActivity) -> None:
        """Initialize the LLMCordBot.

        :param intents: Discord intents configuration.
        :param activity: Custom activity status for the bot.
        """
        start_time = time.perf_counter()
        super().__init__(command_prefix="?", intents=intents, activity=activity)
        logger.debug("Bot init finished in %.2f seconds", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        self.httpx_client = httpx.AsyncClient()
        logger.debug("HTTPX client set in %.2f seconds", time.perf_counter() - start_time)
        self.exit_code: int = 0

        # Initialize services after bot is ready
        self.llm_service: LLMService | None = None
        self.message_processor: MessageProcessor | None = None

    async def close(self) -> None:
        """Cleanup resources before shutting down."""
        await self.httpx_client.aclose()
        await super().close()

    async def setup_hook(self) -> None:
        """Set up internal discord.py hook for asynchronous setup."""
        start_time = time.perf_counter()
        await setup_commands(self)
        logger.debug("Commands setup finished in %.2f seconds", time.perf_counter() - start_time)

        sync_start = time.perf_counter()
        await self.tree.sync()
        logger.debug("Command tree synced in %.2f seconds", time.perf_counter() - sync_start)

        # Initialize services now that bot user is available
        user = self.safe_user  # This will raise if user is None
        self.llm_service = LLMService(self.config, self.httpx_client)
        self.message_processor = MessageProcessor(self.config, user)

        if client_id := self.config.discord.client_id:
            logger.info("Bot invite URL: https://discord.com/oauth2/authorize?client_id=%s&permissions=412317191168&scope=bot", client_id)

        logger.info("Bot ready. Logged in as %s. Total hook setup time: %.2f seconds", self.safe_user, time.perf_counter() - start_time)

    async def on_message(self, message: discord.Message) -> None:
        """Event handler for new messages. Orchestrates the response flow.

        :param message: The incoming Discord message to process.
        """
        if not self.message_processor or not self.llm_service:
            logger.warning("Services not initialized yet")
            return

        # At this point, services are guaranteed to be initialized, so safe_user should be available
        user = self.safe_user  # This will raise if user is None (shouldn't happen)

        if not self.message_processor.valid_trigger_message(message):
            return

        logger.info("Message recieved. User: %s ID: %d", message.author.name, message.author.id)

        try:
            # 1. Prepare the request payload
            start_time = time.perf_counter()
            chat_params = await self.llm_service.prepare_request(message, self.message_processor.msg_nodes, user)
            logger.info("Request prepared in %.4f seconds", time.perf_counter() - start_time)

            # 2. Generate and send response
            await self._generate_response(message, chat_params)
        except Exception:
            logger.exception("Failed to process message from %s", message.author.name)
        finally:
            self.message_processor.prune_msg_nodes()

    async def _generate_response(self, message: discord.Message, chat_params: dict[str, Any]) -> None:
        """Generate and send the LLM response.

        :param message: The triggering Discord message.
        :param chat_params: The parameters for the LLM call.
        """
        # Services should be initialized at this point (checked in on_message)
        if not self.llm_service or not self.message_processor:
            logger.error("Services not available in _generate_response - this should not happen")
            return

        start_time = time.perf_counter()
        response_text = await self.llm_service.generate_response(chat_params)
        logger.info("Response generated in %.4f seconds", time.perf_counter() - start_time)

        if response_text:
            await self.message_processor.send_response_chunks(message, response_text)


async def main() -> int:
    """Entry point for the bot application.

    :return: The exit code (0 for stop, 1 for error, 2 for reload).
    """
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(name=(config_manager.config.discord.status_message)[:128])

    bot = LLMCordBot(intents=intents, activity=activity)
    start_time = time.perf_counter()
    threading.Thread(target=console_listener, args=(bot,), daemon=True).start()
    logger.debug("Thread started in %.2f seconds", time.perf_counter() - start_time)

    retry_delay = 5
    max_delay = 60

    while True:
        try:
            async with bot:
                start_time = time.perf_counter()
                await bot.start(config_manager.config.discord.bot_token)
                logger.debug("Bot started in %.2f seconds", time.perf_counter() - start_time)
            break
        except discord.LoginFailure:
            logger.exception("Invalid Discord token.")
            bot.exit_code = 1
            break
        except (discord.GatewayNotFound, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Network error: %s. Retrying in %d seconds...", e, retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)
        except Exception:
            logger.exception("Fatal error in bot loop")
            bot.exit_code = 1
            break

        # If the bot was closed via console (reload/exit), break the loop
        if bot.exit_code != 0:
            break

    return bot.exit_code


def console_listener(bot: LLMCordBot) -> None:
    """Listen for console commands in a blocking loop.

    Accepts 'reload', 'exit', 'stop', or 'quit' commands to control the bot process.

    :param bot: The running bot instance used to trigger a graceful shutdown.
    """
    try:
        while True:
            command = input().strip().lower()
            if command == "reload":
                bot.exit_code = 2
                asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
                break
            if command in ("exit", "stop", "quit"):
                bot.exit_code = 0
                asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
                break
            logger.warning("Unknown command: %s", command)
    except EOFError:
        pass
    except Exception:
        logger.exception("Error in console listener")
