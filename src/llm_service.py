"""LLM Service for handling LLM interactions.

This module provides a service layer for LLM operations, including
provider resolution, message preparation, and API calls.
"""

import asyncio
import json
import logging
from typing import Any, NamedTuple, cast

import discord
import httpx
import litellm
from litellm import utils as litellm_utils

from .config_manager import ConfigValue, RootConfig
from .llm_utils import build_chat_params, get_llm_provider_model
from .message_utils import MessagePayloadParams, build_messages_payload
from .models import MsgNode
from .tools import tool_manager
from .utils_discord import fetch_history, init_msg_node
from .utils_logging import request_logger
from .utils_regex import replace_placeholders

logger = logging.getLogger(__name__)

PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")


class BuildMessagesParams(NamedTuple):
    """Parameters for building messages payload."""

    message_history: list[discord.Message]
    msg_nodes: dict[int, MsgNode]
    pre_history: str | None
    post_history: str | None
    model: str
    provider: str


class LLMService:
    """Service for handling LLM provider interactions and message processing."""

    def __init__(self, config: RootConfig, httpx_client: httpx.AsyncClient) -> None:
        """Initialize the LLM service.

        :param config: The application configuration.
        :param httpx_client: Shared HTTP client for requests.
        """
        self.config = config
        self.httpx_client = httpx_client

    async def prepare_request(
        self,
        message: discord.Message,
        msg_nodes: dict[int, Any],
        bot_user: discord.ClientUser,
    ) -> dict[str, Any]:
        """Prepare the LLM request parameters for a message.

        :param message: The Discord message to process.
        :param msg_nodes: Dictionary of message nodes.
        :param bot_user: The bot's user object.
        :return: Chat parameters for the LLM call.
        """
        provider, model = self._get_provider_and_model(message)
        message_history = await self._prepare_message_history(message, bot_user)
        await self._initialize_message_nodes(message_history, msg_nodes, bot_user)

        build_params = await self._create_build_parameters(message, message_history, msg_nodes, model, provider, bot_user)
        messages_payload = self._build_and_trim_messages_payload(build_params)

        return self._create_chat_parameters(model, messages_payload, provider)

    def _get_provider_and_model(self, message: discord.Message) -> tuple[str, str]:
        """Get the LLM provider and model for a message.

        :param message: The Discord message.
        :return: Tuple of (provider, model).
        """
        return get_llm_provider_model(
            channel_id=message.channel.id,
            channel_models=self.config.chat.channel_models,
            default_model=self.config.chat.default_model,
        )

    async def _initialize_message_nodes(
        self,
        message_history: list[discord.Message],
        msg_nodes: dict[int, Any],
        bot_user: discord.ClientUser,
    ) -> None:
        """Initialize message nodes for the message history.

        :param message_history: List of messages to process.
        :param msg_nodes: Dictionary to store message nodes.
        :param bot_user: The bot's user object.
        """
        await asyncio.gather(*(init_msg_node(m, msg_nodes, bot_user, self.httpx_client) for m in message_history))

    async def _create_build_parameters(
        self,
        message: discord.Message,
        message_history: list[discord.Message],
        msg_nodes: dict[int, Any],
        model: str,
        provider: str,
        bot_user: discord.ClientUser,
    ) -> BuildMessagesParams:
        """Create build parameters for message processing.

        :param message: The Discord message.
        :param message_history: List of messages in the history.
        :param msg_nodes: Dictionary of message nodes.
        :param model: The LLM model to use.
        :param provider: The LLM provider.
        :param bot_user: The bot's user object.
        :return: BuildMessagesParams object.
        """
        pre_history, post_history = self._generate_system_prompts(message, model, provider, bot_user)
        return BuildMessagesParams(
            message_history=message_history,
            msg_nodes=msg_nodes,
            pre_history=pre_history,
            post_history=post_history,
            model=model,
            provider=provider,
        )

    def _create_chat_parameters(
        self,
        model: str,
        messages_payload: list[dict[str, Any]],
        provider: str,
    ) -> dict[str, Any]:
        """Create the final chat parameters for the LLM call.

        :param model: The LLM model to use.
        :param messages_payload: The processed messages payload.
        :param provider: The LLM provider.
        :return: Dictionary of chat parameters.
        """
        provider_config = self._get_provider_config(provider)
        return build_chat_params(
            model=model,
            messages=messages_payload,
            provider=provider,
            provider_config=provider_config,
            llm_models_config=self.config.llm.models,
        )

    async def generate_response(self, chat_params: dict[str, Any]) -> str | None:
        """Generate a response from the LLM with tool calling support.

        :param chat_params: The parameters for the LLM call.
        :return: The generated response text, or None if failed.
        """
        request_logger.log(chat_params)

        chat_params["tools"] = tool_manager.get_tool_definitions()
        chat_params["tool_choice"] = "auto"

        return await self._perform_completion_with_tools(chat_params)

    async def _prepare_message_history(self, message: discord.Message, bot_user: discord.ClientUser) -> list[discord.Message]:
        """Prepare the message history for context."""
        config = self.config
        use_channel_context = config.chat.use_channel_context
        if use_channel_context and config.chat.force_reply_chains and message.reference:
            use_channel_context = False

        return await fetch_history(
            message=message,
            max_messages=config.chat.max_messages,
            use_channel_context=use_channel_context,
            bot_user=bot_user,
        )

    def _generate_system_prompts(self, message: discord.Message, model: str, provider: str, bot_user: discord.ClientUser) -> tuple[str | None, str | None]:
        """Generate system prompts."""
        pre_history = replace_placeholders(self.config.prompts.pre_history, message, bot_user, model, provider)
        post_history = replace_placeholders(self.config.prompts.post_history, message, bot_user, model, provider)
        return pre_history, post_history

    def _build_and_trim_messages_payload(
        self,
        build_params: BuildMessagesParams,
    ) -> list[dict[str, Any]]:
        """Build and trim the messages payload."""
        payload_params = self._create_payload_params(build_params)
        content_messages = self._build_content_messages(build_params, payload_params)
        messages_payload = self._assemble_messages_payload(build_params, content_messages)

        return self._trim_messages_if_needed(messages_payload, build_params)

    def _create_payload_params(self, build_params: BuildMessagesParams) -> MessagePayloadParams:
        """Create payload parameters based on model capabilities."""
        accept_images = litellm.supports_vision(build_params.model)
        accept_usernames = self._provider_supports_usernames(build_params.provider)

        return MessagePayloadParams(
            max_text=self.config.chat.max_text,
            max_images=self.config.chat.max_images,
            prefix_users=self.config.chat.prefix_users,
            accept_images=accept_images,
            accept_usernames=accept_usernames,
        )

    def _provider_supports_usernames(self, provider: str) -> bool:
        """Check if provider supports usernames in messages."""
        return any(provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    def _build_content_messages(
        self,
        build_params: BuildMessagesParams,
        payload_params: MessagePayloadParams,
    ) -> list[dict[str, Any]]:
        """Build content messages from message history."""
        return build_messages_payload(
            message_history=build_params.message_history,
            msg_nodes=build_params.msg_nodes,
            params=payload_params,
        )

    def _assemble_messages_payload(
        self,
        build_params: BuildMessagesParams,
        content_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble the complete messages payload with system prompts."""
        messages_payload: list[dict[str, Any]] = []

        self._add_system_prompt_if_exists(messages_payload, build_params.pre_history)
        messages_payload.extend(content_messages)
        self._add_system_prompt_if_exists(messages_payload, build_params.post_history)

        return messages_payload

    def _add_system_prompt_if_exists(
        self,
        messages_payload: list[dict[str, Any]],
        prompt_content: str | None,
    ) -> None:
        """Add system prompt to payload if content exists."""
        if prompt_content:
            messages_payload.append({"role": "system", "content": prompt_content})

    def _trim_messages_if_needed(
        self,
        messages_payload: list[dict[str, Any]],
        build_params: BuildMessagesParams,
    ) -> list[dict[str, Any]]:
        """Trim messages payload if it exceeds token limit."""
        full_model_name = f"{build_params.provider}/{build_params.model}"
        trimmed_result = litellm_utils.trim_messages(  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream
            messages_payload,
            model=full_model_name,
            max_tokens=self.config.chat.max_input_tokens,
        )
        # Cast to expected type since litellm's annotations are incomplete
        return cast("list[dict[str, Any]]", trimmed_result) if isinstance(trimmed_result, list) else messages_payload

    def _get_provider_config(self, provider: str) -> ConfigValue:
        """Get provider configuration."""
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            error = f"Provider '{provider}' not found in configuration."
            raise KeyError(error)
        return provider_config

    async def _perform_completion_with_tools(self, chat_params: dict[str, Any]) -> str | None:
        """Perform LLM completion with tool calling."""
        try:
            for i in range(5):
                async with asyncio.timeout(60):
                    response = await litellm.acompletion(**chat_params, client=self.httpx_client)  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream

                # Cast to ModelResponse since we're not using streaming
                model_response = cast("Any", response)  # litellm types are incomplete

                if not hasattr(model_response, "choices") or not model_response.choices:
                    continue

                message = model_response.choices[0].message  # litellm types are incomplete

                # Check for tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    logger.debug("Iteration %d: LLM requested %d tool calls", i, len(message.tool_calls))
                    chat_params["messages"].append(message.model_dump())

                    # Execute tools in parallel
                    results = await asyncio.gather(*(self._execute_tool(tc) for tc in message.tool_calls))
                    chat_params["messages"].extend(results)
                    continue

                # Return the content
                if hasattr(message, "content") and message.content is not None:
                    return str(message.content)
                return ""

            # Removed return None here
        except Exception:
            logger.exception("Error during LLM completion")
            return None
        else:
            return None

    async def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call."""
        # Type assertions for tool_call structure
        tool_call_dict = tool_call
        function_dict = tool_call_dict.get("function", {})

        func_name = str(function_dict.get("name", ""))
        args_str = str(function_dict.get("arguments", ""))

        try:
            args = json.loads(args_str)
            content = await tool_manager.execute_tool(func_name, args)
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Tool execution failed: %s. Error: %s", func_name, e)
            content = f"Error executing tool {func_name}: {e}"

        return {
            "role": "tool",
            "tool_call_id": str(tool_call_dict.get("id", "")),
            "name": func_name,
            "content": str(content),
        }
