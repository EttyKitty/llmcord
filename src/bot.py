"""LLMCord Bot Main Module.

This module contains the core implementation of the LLMCord Discord bot.
It handles the bot's lifecycle, event processing, message validation,
context construction, and interaction with various LLM providers using
OpenAI-compatible APIs.
"""

import asyncio
import logging
import sys
import threading
import time
from typing import Any, Final

import discord
import httpx
import tiktoken
from discord.ext import commands
from openai import AsyncOpenAI

from .commands_manager import setup
from .config_manager import RootConfig, config_manager
from .utils import MsgNode, build_chat_params, build_messages_payload, get_llm_provider_model, get_llm_specials, get_llm_stream, get_provider_config, init_msg_node
from .utils_discord import fetch_history, is_message_allowed
from .utils_logging import request_logger
from .utils_regex import process_response_text, replace_placeholders

MAX_MESSAGE_NODES: Final[int] = 500
TOKENIZER = tiktoken.get_encoding("cl100k_base")
DISCORD_CHAR_LIMIT: Final[int] = 2000

logger = logging.getLogger(__name__)


class LLMCordBot(commands.Bot):
    """The main bot class for llmcord, encapsulating state and logic.

    This class manages the bot's lifecycle, message processing, conversation
    context building, and interaction with LLM providers via OpenAI-compatible APIs.

    :ivar openai_clients: Cache of AsyncOpenAI clients keyed by base URL.
    :ivar msg_nodes: Cache of MsgNode objects keyed by message ID.
    :ivar last_task_time: Timestamp of the last processed task.
    :ivar httpx_client: Shared HTTP client for external requests.
    :ivar exit_code: Exit code.
    """

    @property
    def config(self) -> RootConfig:
        """Get the current configuration.

        :return: The root configuration object.
        """
        return config_manager.config

    # I'm too stupid to find a different solution to strict type checking in local scopes
    @property
    def safe_user(self) -> discord.ClientUser:
        """Get the bot user, raising an error if not initialized.

        :return: The bot's ClientUser.
        :raises RuntimeError: If the bot user is not yet initialized.
        """
        if self.user is None:
            error = "Bot user not initialized!"
            raise RuntimeError(error)

        return self.user

    def __init__(self, intents: discord.Intents, activity: discord.CustomActivity) -> None:
        """Initialize the LLMCordBot.

        :param intents: Discord intents configuration.
        :param activity: Custom activity status for the bot.
        """
        super().__init__(command_prefix="?", intents=intents, activity=activity)

        self.openai_clients: dict[str, AsyncOpenAI] = {}
        self.msg_nodes: dict[int, MsgNode] = {}
        self.last_task_time: float = 0.0
        self.httpx_client = httpx.AsyncClient()
        self.exit_code: int = 0

    async def close(self) -> None:
        """Cleanup resources before shutting down."""
        await self.httpx_client.aclose()
        await super().close()

    async def setup_hook(self) -> None:
        """Set up internal discord.py hook for asynchronous setup."""
        await setup(self)
        await self.tree.sync()

        if client_id := self.config.discord.client_id:
            logger.info("Bot invite URL: https://discord.com/oauth2/authorize?client_id=%s&permissions=412317191168&scope=bot", client_id)

        logger.info("Bot ready. Logged in as %s", self.safe_user)

    async def on_message(self, message: discord.Message) -> None:
        """Event handler for new messages. Orchestrates the response flow.

        :param message: The incoming Discord message to process.
        """
        # =============================
        # ======== 0. Gatekeep ========
        # =============================
        if message.author.bot:
            return

        is_dm = message.channel.type == discord.ChannelType.private
        is_mentioned = self.safe_user in message.mentions

        if not (is_dm or is_mentioned):
            return

        if not is_message_allowed(message, self.config.discord.permissions, allow_dms=self.config.discord.allow_dms):
            logger.info("Message blocked. User: %s ID: %d", message.author.name, message.author.id)
            return

        try:
            # 1. Prepare the request payload
            start_time = time.perf_counter()
            client, chat_params = await self._prepare_llm_request(message)

            logger.info("Request prepared in %.4f seconds", time.perf_counter() - start_time)

            # 2. Generate and send response
            await self._generate_response(message, client, chat_params)
        except Exception:
            logger.exception("Failed to process message from %s", message.author.name)
        finally:
            self._prune_msg_nodes()

    async def _prepare_llm_request(self, message: discord.Message) -> tuple[AsyncOpenAI, dict[str, Any]]:
        """Build the history, calculate tokens, and construct the LLM payload.

        :param message: The triggering Discord message.
        :return: A tuple of the OpenAI client and the chat parameters.
        """
        provider, model = get_llm_provider_model(
            channel_id=message.channel.id,
            channel_models=self.config.chat.channel_models,
            default_model=self.config.chat.default_model,
        )

        # Build History
        message_history = await self._prepare_message_history(message, self.config.chat.max_messages)
        await asyncio.gather(*(init_msg_node(m, self.msg_nodes, self.safe_user, self.httpx_client) for m in message_history))

        # Handle System Prompts & Tokens
        pre_history = replace_placeholders(self.config.prompts.pre_history, message, self.safe_user, model, provider)
        post_history = replace_placeholders(self.config.prompts.post_history, message, self.safe_user, model, provider)

        system_token_count = sum(len(TOKENIZER.encode(p)) for p in (pre_history, post_history) if p)

        accept_images, accept_usernames = get_llm_specials(provider, model)
        messages_payload = build_messages_payload(
            message_history=message_history,
            msg_nodes=self.msg_nodes,
            max_input_tokens=self.config.chat.max_input_tokens - system_token_count,
            max_text=self.config.chat.max_text,
            max_images=self.config.chat.max_images,
            prefix_users=self.config.chat.prefix_users,
            accept_images=accept_images,
            accept_usernames=accept_usernames,
        )

        if pre_history:
            messages_payload.insert(0, {"role": "system", "content": pre_history})
        if post_history:
            messages_payload.append({"role": "system", "content": post_history})

        provider_config, openai_client = get_provider_config(provider=provider, providers_config=self.config.llm.providers, client_cache=self.openai_clients, httpx_client=self.httpx_client)

        chat_params = build_chat_params(model=model, messages=messages_payload, provider=provider, provider_config=provider_config, llm_models_config=self.config.llm.models)

        return openai_client, chat_params

    async def _prepare_message_history(self, message: discord.Message, max_messages: int) -> list[discord.Message]:
        """Fetch and initialize the message history for context building.

        :param message: The triggering message.
        :return: A list of initialized message nodes.
        """
        config = self.config
        use_channel_context = config.chat.use_channel_context
        if use_channel_context and config.chat.force_reply_chains and message.reference:
            use_channel_context = False

        logger.debug("Initializing message nodes...")

        return await fetch_history(
            message=message,
            max_messages=max_messages,
            use_channel_context=use_channel_context,
            bot_user=self.safe_user,
        )

    async def _generate_response(
        self,
        trigger_msg: discord.Message,
        client: AsyncOpenAI,
        chat_params: dict[str, Any],
    ) -> None:
        """Handle the lifecycle of an LLM request and its delivery to Discord.

        This method orchestrates the stream consumption and ensures the resulting
        text is partitioned and sent to the correct Discord channel.

        :param trigger_msg: The message that triggered the response.
        :param client: The initialized OpenAI client.
        :param chat_params: The parameters for the API call.
        """
        overall_start: float = time.perf_counter()
        request_logger.log(chat_params)

        full_content: str = ""
        response_msgs: list[discord.Message] = []

        try:
            # 1. Consume the stream and accumulate content
            full_content = await self._consume_stream(trigger_msg, client, chat_params)

            if not full_content:
                return

            # 2. Process the text (Sanitize, regex, etc.)
            final_text: str = process_response_text(
                text=full_content,
                sanitize=self.config.chat.sanitize_response,
            )

            # 3. Send the response in chunks
            response_msgs = await self._send_response_chunks(trigger_msg, final_text)

        except Exception:
            logger.exception("Error during response generation")
            await trigger_msg.channel.send("⚠️ An error occurred while processing your request.")
        finally:
            self._release_node_locks(response_msgs, full_content)
            logger.info("Response finished in %.4f seconds", time.perf_counter() - overall_start)

    async def _consume_stream(self, trigger_msg: discord.Message, client: AsyncOpenAI, chat_params: dict[str, Any]) -> str:
        """Consume the LLM stream and maintain the Discord typing indicator.

        :param trigger_msg: The message that triggered the response.
        :param client: The initialized OpenAI client.
        :param chat_params: The parameters for the API call.
        :return: The full accumulated string content from the LLM.
        """
        full_content: list[str] = []

        async with trigger_msg.channel.typing():
            logger.debug("Streaming started...")
            async for content in get_llm_stream(client, chat_params):
                full_content.extend(content)

        return "".join(full_content)

    async def _send_response_chunks(self, trigger_msg: discord.Message, content: str) -> list[discord.Message]:
        """Split the final content and send it as one or more Discord messages.

        :param trigger_msg: The original message from the user.
        :param content: The full sanitized text to send.
        :return: A list of the messages sent.
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

            node: MsgNode = MsgNode()
            self.msg_nodes[new_msg.id] = node
            await node.lock.acquire()

        return msgs

    def _release_node_locks(self, msgs: list[discord.Message], full_content: str) -> None:
        """Update MsgNodes with the final generated text and release their locks.

        :param msgs: The list of messages sent during this response cycle.
        :param full_content: The complete text generated by the LLM.
        """
        for msg in msgs:
            node = self.msg_nodes.get(msg.id)
            if node:
                node.text = full_content
                if node.lock.locked():
                    node.lock.release()

    def _prune_msg_nodes(self) -> None:
        """Prunes old MsgNodes to prevent memory leaks."""
        if len(self.msg_nodes) <= MAX_MESSAGE_NODES:
            return

        to_remove_count = len(self.msg_nodes) - MAX_MESSAGE_NODES
        logger.debug("Pruning %d old MsgNodes...", to_remove_count)

        sorted_ids = sorted(self.msg_nodes.keys())
        ids_to_delete = sorted_ids[:to_remove_count]

        for msg_id in ids_to_delete:
            self.msg_nodes.pop(msg_id, None)

        logger.debug("Successfully pruned %d nodes!", len(ids_to_delete))


async def main() -> None:
    """Entry point for the bot application.

    Initializes Discord intents, creates the bot instance, starts the console listener,
    and runs the bot with the configured token.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(name=(config_manager.config.discord.status_message)[:128])

    bot = LLMCordBot(intents=intents, activity=activity)
    threading.Thread(target=console_listener, args=(bot,), daemon=True).start()

    try:
        async with bot:
            await bot.start(config_manager.config.discord.bot_token)
    except KeyboardInterrupt:
        pass
    finally:
        # Access the exit code from the bot instance
        exit_status = bot.exit_code
        logger.info("Bot process exiting with code %d", exit_status)
        sys.exit(exit_status)


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
