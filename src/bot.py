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
from typing import Any

import discord
import httpx
import tiktoken
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from openai import AsyncOpenAI

from .commands import setup
from .config import RootConfig, config_manager
from .discord_utils import fetch_history, is_message_allowed
from .logger import request_logger
from .utils import MsgNode, build_chat_params, build_messages_payload, extract_chunk_content, get_llm_provider_model, get_llm_specials, get_provider_config, init_msg_node, process_response_text, replace_placeholders, update_content_buffer

MAX_MESSAGE_NODES = 500
TOKENIZER = tiktoken.get_encoding("cl100k_base")


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

        # =============================
        # === 1. Build the Request ====
        # =============================

        # - Build Context
        start_time = time.perf_counter()
        provider, model = get_llm_provider_model(
            channel_id=message.channel.id,
            channel_models=self.config.chat.channel_models,
            default_model=self.config.chat.default_model,
        )

        # - Generate system prompts early to calculate their token cost
        pre_history = replace_placeholders(self.config.prompts.pre_history, message, self.safe_user, model, provider)
        post_history = replace_placeholders(self.config.prompts.post_history, message, self.safe_user, model, provider)

        # - Build the History
        message_history = await self._prepare_message_history(message, self.config.chat.max_messages)

        # - Initialize message nodes
        await asyncio.gather(*(init_msg_node(m, self.msg_nodes, self.safe_user, self.httpx_client) for m in message_history))

        # - Build the Payload
        system_token_count = 0
        if pre_history:
            system_token_count += len(TOKENIZER.encode(pre_history))
        if post_history:
            system_token_count += len(TOKENIZER.encode(post_history))

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

        # - Insert Prompts
        if pre_history:
            messages_payload.insert(0, {"role": "system", "content": pre_history})
        if post_history:
            messages_payload.append({"role": "system", "content": post_history})

        # - Finish the Payload
        provider_config, openai_client = get_provider_config(provider=provider, providers_config=self.config.llm.providers, client_cache=self.openai_clients, httpx_client=self.httpx_client)
        chat_params = build_chat_params(model=model, messages=messages_payload, provider=provider, provider_config=provider_config, llm_models_config=self.config.llm.models)

        elapsed_time = time.perf_counter() - start_time
        logger.info("Request prepared in %s seconds!", f"{elapsed_time:.4f}")

        # 2. Send the Request + 3. Get the Response (Fuck, why are they in one function)
        await self._generate_response(message, openai_client, chat_params)

        # 4. Cleanup
        self._prune_msg_nodes()

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
        """Handle the streaming of the LLM response back to Discord.

        :param trigger_msg: The message that triggered the response.
        :param client: The initialized OpenAI client.
        :param chat_params: The parameters for the API call.
        """
        overall_start = time.perf_counter()
        max_len = 4000

        response_msgs: list[discord.Message] = []
        response_contents: list[str] = []

        request_logger.log(chat_params)

        typing_manager = trigger_msg.channel.typing()
        typing_active = False

        try:
            stream = await client.chat.completions.create(**chat_params)

            is_first_chunk = True
            async for chunk in stream:
                if is_first_chunk:
                    logger.debug("Streaming started...")
                    await typing_manager.__aenter__()
                    typing_active = True
                    is_first_chunk = False

                content = extract_chunk_content(chunk)
                if not content:
                    continue

                update_content_buffer(response_contents, content, max_len)

                if chunk.choices and chunk.choices[0].finish_reason:
                    logger.debug("Streaming finished. Reason: %s", chunk.choices[0].finish_reason)

            await self._finalize_response(trigger_msg, response_msgs, response_contents)

        except Exception:
            logger.exception("Error while generating response!")
            await trigger_msg.channel.send("⚠️ An error occurred.")
        finally:
            if typing_active:
                await typing_manager.__aexit__(None, None, None)

            overall_elapsed = time.perf_counter() - overall_start
            logger.info("Response generation finished in %s seconds!", f"{overall_elapsed:.4f}")

            self._release_node_locks(response_msgs, response_contents)

    async def _finalize_response(
        self,
        trigger_msg: discord.Message,
        msgs: list[discord.Message],
        contents: list[str],
    ) -> None:
        """Process the final text and update the UI one last time.

        :param trigger_msg: The original trigger message.
        :param msgs: The list of sent response messages.
        :param contents: The raw content chunks.
        """
        full_text = "".join(contents)
        final_text = process_response_text(text=full_text, sanitize=self.config.chat.sanitize_response)

        # Split final text into 2000-char chunks for plain text sending
        chunk_size = 2000
        text_chunks = [final_text[i : i + chunk_size] for i in range(0, len(final_text), chunk_size)]
        for chunk in text_chunks:
            view = LayoutView().add_item(TextDisplay(content=chunk))
            await self._send_response_node(trigger_msg, msgs, view=view)

    async def _send_response_node(
        self,
        trigger_msg: discord.Message,
        msgs: list[discord.Message],
        content: str = "",
        view: LayoutView | None = None,
    ) -> None:
        """Send a new message and initialize its MsgNode with a lock.

        :param trigger_msg: The original trigger message.
        :param msgs: The list of existing response messages (will be appended to).
        :param content: Message content.
        :param view: Message view.
        """
        target = trigger_msg if not msgs else msgs[-1]

        reply_kwargs: dict[str, Any] = {
            "content": content,
            "view": view,
            "silent": True,
        }

        msg = await target.reply(**reply_kwargs)
        msgs.append(msg)

        node = MsgNode(parent_msg=trigger_msg)
        self.msg_nodes[msg.id] = node
        await node.lock.acquire()

    def _release_node_locks(self, msgs: list[discord.Message], contents: list[str]) -> None:
        """Update MsgNodes with final text and release their locks.

        :param msgs: The list of sent messages.
        :param contents: The list of content chunks corresponding to messages.
        """
        full_content = "".join(contents)

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
