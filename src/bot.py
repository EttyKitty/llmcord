"""LLMCord Bot Main Module.

This module contains the core implementation of the LLMCord Discord bot.
It handles the bot's lifecycle, event processing, message validation,
context construction, and interaction with various LLM providers using
OpenAI-compatible APIs.
"""

import asyncio
import threading
import time
from typing import Final

import aiohttp
import discord
import httpx
from discord.ext import commands
from loguru import logger

from .commands_manager import setup_commands
from .config_manager import RootConfig, config_manager
from .discord_service import DiscordService
from .llm_service import LLMService
from .logging_utils_ import timer
from .message_service import MessageService
from .regex_utils import process_response_text

DISCORD_REST_SUCCESS: Final[int] = 200
DISCORD_REST_INTERVAL: Final[int] = 60


class LLMCordBot(commands.Bot):
    """The main bot class for llmcord, encapsulating state and logic.

    This class manages the bot's lifecycle, message processing, conversation
    context building, and interaction with LLM providers via OpenAI-compatible APIs.

    :ivar llm_service: Service for LLM operations.
    :ivar message_service: Service for message processing.
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
        logger.debug("Bot init finished in {:.2f} seconds", time.perf_counter() - start_time)

        start_time = time.perf_counter()
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        logger.debug("HTTPX client set in {:.2f} seconds", time.perf_counter() - start_time)
        self.exit_code: int = 0

        # Initialize services after bot is ready
        self.llm_service: LLMService | None = None
        self.message_service: MessageService | None = None
        self.discord_service: DiscordService | None = None

    async def close(self) -> None:
        """Cleanup resources before shutting down."""
        await self.httpx_client.aclose()
        await super().close()

    async def setup_hook(self) -> None:
        """Set up internal discord.py hook for asynchronous setup."""
        start_time = time.perf_counter()
        logger.debug("Setting up commands...")
        await setup_commands(self)
        logger.debug("Commands setup finished in {:.2f} seconds", time.perf_counter() - start_time)

        sync_start = time.perf_counter()
        logger.debug("Syncing command tree...")
        await self.tree.sync()
        logger.debug("Command tree synced in {:.2f} seconds!", time.perf_counter() - sync_start)

        # Initialize services now that bot user is available
        logger.debug("Initializing services...")
        self.llm_service = LLMService(self.httpx_client)
        self.message_service = MessageService(config=self.config, user=self.safe_user, httpx_client=self.httpx_client)
        self.discord_service = DiscordService(config=self.config, message_nodes=self.message_service.message_nodes)
        logger.debug("Services initialized!")

        # Verify we can reach Discord's REST API and that the token/user are valid.
        verified = await self._verify_discord_rest()
        if not verified:
            logger.warning("Discord REST verification failed")
            self.exit_code = 0
        else:
            logger.info("Bot ready. Logged in as {}. Total hook setup time: {:.2f} seconds", self.safe_user, time.perf_counter() - start_time)

            if client_id := self.config.discord.client_id:
                logger.info("Bot invite URL: https://discord.com/oauth2/authorize?client_id={}&permissions=412317191168&scope=bot", client_id)

        # Start periodic verification task
        self.loop.create_task(self._periodic_verification(DISCORD_REST_INTERVAL))

    async def on_message(self, message: discord.Message) -> None:
        """Event handler for new messages. Orchestrates the response flow.

        :param message: The incoming Discord message to process.
        """
        if not self.message_service or not self.llm_service or not self.discord_service:
            logger.warning("Services not initialized yet")
            return

        if not self._valid_trigger_message(message):
            return

        logger.info("Message received. User: {} ID: {}", message.author.name, message.author.id)

        try:
            with timer("LLM payload preparation"):
                llm_payload = await self.message_service.construct_llm_payload(message)

            with timer("LLM request"):
                async with message.channel.typing():
                    response_text = await self.llm_service.perform_completion(llm_payload)

            if response_text.startswith("__STOP_RESPONSE__"):
                _, reason = response_text.split("|")

                logger.info("Response aborted by LLM. Reason: {}", reason)
                return

            with timer("Discord response"):
                response_text = process_response_text(response_text, sanitize=self.config.chat.sanitize_response, bot_name=self.safe_user.display_name)
                await self.discord_service.send_response_chunks(message, response_text)
        except (httpx.RequestError, discord.DiscordException):
            logger.exception("Failed to process message from {}", message.author.name)
        finally:
            self.message_service.prune_msg_nodes()

    async def _verify_discord_rest(self, retries: int = 3, backoff: float = 4, timeout: float = 10.0) -> bool:
        """REST check to ensure Discord API and bot token are reachable.

        Performs a GET on `/users/@me` using the configured bot token and
        validates the returned user id matches the logged-in user. Retries on
        transient failures.

        :return: True if verification succeeded, False otherwise.
        """
        token = self.config.discord.bot_token
        if not token:
            logger.warning("No bot token configured for readiness verification")
            return False

        url = "https://discord.com/api/v10/users/@me"
        headers = {"Authorization": f"Bot {token}"}

        for attempt in range(1, retries + 1):
            try:
                resp = await self.httpx_client.get(url, headers=headers, timeout=timeout)
                if resp.status_code == DISCORD_REST_SUCCESS:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    if data and str(data.get("id")) == str(self.safe_user.id):
                        return True
                    logger.warning("Discord REST returned unexpected data or id mismatch: {}", data)
                    return False
                logger.warning("Discord REST returned status {} on attempt {}", resp.status_code, attempt)
            except httpx.RequestError:
                logger.warning("Discord readiness check attempt {} failed", attempt)

            if attempt < retries:
                await asyncio.sleep(backoff * attempt)

        logger.warning("Discord REST verification failed after {} attempts", retries)
        return False

    async def _periodic_verification(self, interval: int) -> None:
        """Periodically verify Discord REST API connection."""
        try:
            while True:
                await asyncio.sleep(interval)
                verified = await self._verify_discord_rest()
                if not verified:
                    logger.warning("Periodic Discord REST verification failed")
        except asyncio.CancelledError:
            logger.debug("Periodic verification task cancelled")

    def _is_message_allowed(self, message: discord.Message) -> bool:
        """Check if the message author and channel are allowed based on configuration.

        :param message: The Discord message to check.
        :return: True if allowed, False otherwise.
        """
        is_dm = message.channel.type == discord.ChannelType.private

        config = self.config.discord

        # 1. Admin Bypass
        if message.author.id in config.permissions.users.admin_ids:
            return True

        # 2. User/Role Validation
        role_ids = {role.id for role in getattr(message.author, "roles", ())}
        allowed_users = config.permissions.users.allowed_ids
        blocked_users = config.permissions.users.blocked_ids
        allowed_roles = config.permissions.roles.allowed_ids
        blocked_roles = config.permissions.roles.blocked_ids

        # Determine if the user is "good" (allowed by default or explicitly)
        allow_all_users = not allowed_users if is_dm else (not allowed_users and not allowed_roles)
        is_good_user = allow_all_users or message.author.id in allowed_users or not role_ids.isdisjoint(allowed_roles)

        # Determine if the user is "bad" (not good or explicitly blocked)
        is_bad_user = not is_good_user or message.author.id in blocked_users or not role_ids.isdisjoint(blocked_roles)

        if is_bad_user:
            return False

        # 3. Channel Validation
        # Collect current channel ID, parent ID (for threads), and category ID
        channel_ids: set[int] = set()
        channel_ids_list: list[int | None] = [message.channel.id, getattr(message.channel, "parent_id", None), getattr(message.channel, "category_id", None)]
        for cid in channel_ids_list:
            if cid is not None:
                channel_ids.add(cid)

        allowed_channels = config.permissions.channels.allowed_ids
        blocked_channels = config.permissions.channels.blocked_ids

        # Determine if the channel is "good"
        is_good_channel = config.allow_dms if is_dm else (not allowed_channels or not channel_ids.isdisjoint(allowed_channels))

        # Determine if the channel is "bad"
        is_bad_channel = not is_good_channel or not channel_ids.isdisjoint(blocked_channels)

        return not is_bad_channel

    def _valid_trigger_message(self, message: discord.Message) -> bool:
        """Determine if a message should be processed.

        :param message: The Discord message to check.
        :return: True if the message should be processed.
        """
        if message.author.bot:
            return False

        is_dm = message.channel.type == discord.ChannelType.private
        is_mentioned = self.safe_user in message.mentions

        if not (is_dm or is_mentioned):
            return False

        if not self._is_message_allowed(message):
            logger.info("Message blocked. User: {} ID: {}", message.author.name, message.author.id)
            return False

        return True


async def main() -> int:
    """Entry point for the bot application.

    :return: The exit code (0 for stop, 1 for error, 2 for reload).
    """
    logger.debug("Initializing bot.py...")
    start_time = time.perf_counter()
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(name=(config_manager.config.discord.status_message)[:128])

    bot = LLMCordBot(intents=intents, activity=activity)

    threading.Thread(target=console_listener, args=(bot,), daemon=True).start()
    logger.debug("Bot.py initialization finished in {:.2f} seconds!", time.perf_counter() - start_time)

    retry_delay = 5
    max_delay = 60

    while True:
        try:
            async with bot:
                start_time = time.perf_counter()
                logger.debug("About to start bot...")
                await bot.start(config_manager.config.discord.bot_token)
            break
        except discord.LoginFailure:
            logger.exception("Invalid Discord token.")
            bot.exit_code = 1
            break
        except (discord.GatewayNotFound, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Network error: {}. Retrying in {} seconds...", e, retry_delay)
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
            logger.warning("Unknown command: {}", command)
    except EOFError:
        pass
    except Exception:
        logger.exception("Error in console listener")
