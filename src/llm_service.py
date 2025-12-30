import asyncio
import json
import logging
import os
from typing import Any, cast

import httpx
import litellm

from .llm_tools import tool_manager
from .logging_utils_ import request_logger

logger = logging.getLogger(__name__)
os.environ["LITELLM_LOG"] = "ERROR"
litellm.telemetry = False
litellm.modify_params = True


class LLMService:
    def __init__(self, httpx_client: httpx.AsyncClient) -> None:
        """Initialize the LLM service.

        :param httpx_client: Shared HTTP client for requests.
        """
        self.httpx_client = httpx_client

    async def perform_completion(self, chat_params: dict[str, Any]) -> str:
        """Perform LLM completion with tool calling."""
        try:
            request_logger.log(chat_params)
            for i in range(5):
                async with asyncio.timeout(60):
                    response = await litellm.acompletion(**chat_params, client=self.httpx_client)  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream

                model_response = cast("Any", response)  # litellm types are incomplete

                if not hasattr(model_response, "choices") or not model_response.choices:
                    continue

                message = model_response.choices[0].message  # litellm types are incomplete

                if hasattr(message, "tool_calls") and message.tool_calls:
                    logger.debug("Iteration %d: LLM requested %d tool calls", i, len(message.tool_calls))
                    chat_params["messages"].append(message.model_dump())

                    results = await asyncio.gather(*(self._execute_tool(tc) for tc in message.tool_calls))
                    chat_params["messages"].extend(results)
                    continue

                if hasattr(message, "content") and message.content is not None:
                    return str(message.content)
                return ""

        except Exception:
            logger.exception("Error during LLM completion")
            return ""
        else:
            return ""

    async def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call."""
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
