"""LLM Service Module.

This module provides the LLMService class for performing LLM completions
with tool-calling support.
"""

import asyncio
import os
from typing import Any, cast

import discord
import litellm
from loguru import logger

from .llm_tools import run_tool_call
from .logging_utils_ import request_logger

os.environ["LITELLM_LOG"] = "ERROR"
litellm.telemetry = False
litellm.modify_params = True

MAX_TOOL_ITERATIONS = 5


async def perform_completion(chat_params: dict[str, Any], client: discord.Client) -> str:
    """Perform LLM completion with iterative tool calling support.

    :param chat_params: The LLM completion parameters including messages and model.
    :param client: Discord client for tool calls that require Discord API access.
    :return: The final response content, or empty string on failure/exhaustion.
    """
    try:
        params = chat_params.copy()
        params["messages"] = list(params["messages"])

        for i in range(MAX_TOOL_ITERATIONS):
            request_logger.log(params)
            response = await litellm.acompletion(**params, timeout=180)  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream

            model_response = cast("Any", response)  # litellm types are incomplete

            if not hasattr(model_response, "choices") or not model_response.choices:
                logger.debug("Iteration {}: No choices in response", i)
                continue

            message = model_response.choices[0].message  # litellm types are incomplete

            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.debug("Iteration {}: LLM requested {} tool calls", i, len(message.tool_calls))
                params["messages"].append(message.model_dump())

                results = await asyncio.gather(*(run_tool_call(tc, client) for tc in message.tool_calls))

                for res in results:
                    if res["content"].startswith("__STOP_RESPONSE__"):
                        logger.debug("Iteration {}: Abort sentinel detected in tool results", i)
                        return res["content"]

                params["messages"].extend(results)
                continue

            if hasattr(message, "content") and message.content is not None:
                return str(message.content)
            return ""

    except Exception:
        logger.exception("Error during LLM completion")
        return ""

    logger.warning("Tool call loop exhausted after 5 iterations without final response")
    return ""
