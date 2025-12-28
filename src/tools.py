"""LLMCord Tools Module.

This module defines the tools available to the LLM and the logic to execute them.
"""

import logging
import warnings
from typing import Any, cast

from ddgs import (
    DDGS,  # type: ignore[import-untyped] # DDGS library has incomplete type stubs, remove when fixed upstream
)

# Suppress the duckduckgo_search rename warning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")


logger = logging.getLogger(__name__)


class ToolManager:
    """Manages LLM tools and their execution."""

    @staticmethod
    def get_tool_definitions() -> list[dict[str, Any]]:
        """Return the JSON schemas for available tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current events, facts, or general information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Map a tool name to its Python implementation and execute it."""
        try:
            if name == "web_search":
                return await self.web_search(arguments.get("query", ""))
        except Exception as e:
            logger.exception("Failed to execute tool %s", name)
            return f"Error executing tool: {e}"
        else:
            return f"Error: Tool '{name}' not found."

    async def web_search(self, query: str) -> str:
        """Perform a web search using DuckDuckGo."""
        if not query:
            return "Error: No query provided."

        logger.info("Performing web search for: %s", query)
        results: list[str] = []
        try:
            # The latest DDGS version works best as a context manager
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=5)
                for r in ddgs_gen:
                    r_dict = cast("dict[str, str]", r)
                    results.append(f"Title: {r_dict['title']}\nSnippet: {r_dict['body']}\nURL: {r_dict['href']}\n")
        except Exception as e:
            logger.exception("DuckDuckGo search failed!")
            return f"Search failed: {e}"

        return "\n---\n".join(results) if results else "No results found."


tool_manager: ToolManager = ToolManager()
