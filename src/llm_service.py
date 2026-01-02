import asyncio
import json
import os
from typing import Any, cast

import httpx
import litellm
from loguru import logger

from .llm_tools import tool_manager
from .logging_utils_ import request_logger

os.environ["LITELLM_LOG"] = "ERROR"
litellm.telemetry = False
litellm.modify_params = True


class LLMService:
    """Service for performing LLM completions with tool-calling support."""

    def __init__(self, httpx_client: httpx.AsyncClient) -> None:
        """Initialize the LLM service.

        :param httpx_client: Shared HTTP client for requests.
        """
        self.httpx_client = httpx_client

    async def perform_completion(self, chat_params: dict[str, Any]) -> str:
        """Perform LLM completion with tool calling."""
        try:
            params = chat_params.copy()
            params["messages"] = list(params["messages"])

            request_logger.log(params)
            for i in range(5):
                response = await litellm.acompletion(**params, client=self.httpx_client, timeout=180)  # type: ignore[no-untyped-call] # litellm has incomplete type stubs, remove when fixed upstream

                model_response = cast("Any", response)  # litellm types are incomplete

                if not hasattr(model_response, "choices") or not model_response.choices:
                    logger.debug("Iteration {}: No choices in response", i)
                    continue

                message = model_response.choices[0].message  # litellm types are incomplete

                if hasattr(message, "tool_calls") and message.tool_calls:
                    logger.debug("Iteration {}: LLM requested {} tool calls", i, len(message.tool_calls))
                    params["messages"].append(message.model_dump())

                    results = await asyncio.gather(*(self._execute_tool(tc) for tc in message.tool_calls))

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
            logger.warning("Tool execution failed: {}. Error: {}", func_name, e)
            content = f"Error executing tool {func_name}: {e}"

        return {
            "role": "tool",
            "tool_call_id": str(tool_call_dict.get("id", "")),
            "name": func_name,
            "content": str(content),
        }
