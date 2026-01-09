"""LLMCord Tools Module.

This module defines the tools available to the LLM and the logic to execute them.
"""

import ipaddress
import json
import re
import socket
import urllib.parse
from pathlib import Path
from typing import Any, cast

import discord
import httpx
import trafilatura
from ddgs import DDGS  # type: ignore[import-untyped]
from loguru import logger

MAX_CONTENT_SIZE = 500000
TOOLS_PATH = Path(__file__).parent / "llm_tools.json"
MSG_LINK_PATTERN = re.compile(r"channels/(?P<guild_id>\d+)/(?P<channel_id>\d+)/(?P<message_id>\d+)")
GENERIC_ERROR = "Error: Unable to read the linked message. I may lack permissions, or the content is inaccessible."
DEFAULT_CONTEXT_PADDING = 10

class ToolManager:
    """Manages LLM tools and their execution."""

    def __init__(self) -> None:
        self.client: discord.Client | None = None

    def bind_client(self, client: discord.Client) -> None:
        """Bind the Discord client to the tool manager."""
        self.client = client

    @staticmethod
    def get_tool_definitions() -> list[dict[str, Any]]:
        """Return the JSON schemas for available tools."""
        with (TOOLS_PATH).open() as f:
            return json.load(f)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Map a tool name to its Python implementation and execute it."""
        try:
            if name == "web_search":
                return await self._web_search(arguments.get("query", ""))
            if name == "open_link":
                return await self._open_link(arguments.get("url", ""))
            if name == "ignore_message":
                return await self._ignore_message(arguments.get("reason", "No reason provided"))
            if name == "read_message_link":
                return await self._read_message_link(arguments.get("link", ""), arguments.get("context_padding", DEFAULT_CONTEXT_PADDING))
        except Exception as e:
            logger.exception("Failed to execute tool {}", name)
            return f"Error executing tool: {e}"
        else:
            return f"Error: Tool '{name}' not found."

    @staticmethod
    async def _web_search(query: str) -> str:
        """Perform a web search using DuckDuckGo."""
        if not query:
            return "Error: No query provided."

        logger.info("Performing web search for: {}", query)
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

    @staticmethod
    def _is_safe_host(netloc: str) -> bool:
        """Check if the host is safe to access (not localhost or private IP)."""
        # Remove port if present
        host = netloc.split(":")[0].lower()

        # Block localhost variations
        if host in ("localhost", "127.0.0.1", "::1"):
            return False

        try:
            # Collect all resolved addresses first
            all_addrs: list[Any] = []

            # Try IPv4
            try:
                addr_info = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
                all_addrs.extend(info[4][0] for info in addr_info)
            except socket.gaierror:
                pass

            # Try IPv6
            try:
                addr_info = socket.getaddrinfo(host, None, socket.AF_INET6, socket.SOCK_STREAM)
                all_addrs.extend(info[4][0] for info in addr_info)
            except socket.gaierror:
                pass

            for resolved_ip in all_addrs:
                ip = ipaddress.ip_address(resolved_ip)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False
        except ValueError:
            pass

        return True

    async def _open_link(self, url: str) -> str:
        """Fetch and extract the main content of a web page with security measures."""
        if not url:
            logger.debug("open_link: No URL provided")
            return "Error: No URL provided."

        logger.debug("open_link: Validating URL: {}", url)

        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug("open_link: Invalid scheme '{}' for URL: {}", parsed.scheme, url)
            return "Error: Invalid URL scheme. Only HTTP and HTTPS are allowed."
        if not parsed.netloc:
            logger.debug("open_link: Missing netloc for URL: {}", url)
            return "Error: Invalid URL format."

        logger.debug("open_link: Checking host safety for: {}", parsed.netloc)
        if not self._is_safe_host(parsed.netloc):
            logger.debug("open_link: Host rejected (localhost or private network): {}", parsed.netloc)
            return "Error: Access to localhost or private networks is not allowed for security reasons."

        logger.info("open_link: URL validation passed, fetching content from: {}", url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.google.com/",
                "Connection": "keep-alive",
            }
            async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=False, http2=True) as client:
                logger.debug("open_link: Sending HTTP request to: {}", url)
                response = await client.get(url)
                response.raise_for_status()
                logger.debug("open_link: HTTP {} response received", response.status_code)

                raw_html = response.text

                logger.debug("open_link: Extracting main content from HTML")
                extracted_content = trafilatura.extract(raw_html, include_comments=False)

                if not extracted_content:
                    logger.debug("open_link: No extractable content found, returning plain text")
                    extracted_content = raw_html[:3000]

                content_length = len(extracted_content)
                logger.debug("open_link: Response content length: {} bytes", content_length)

                if content_length > MAX_CONTENT_SIZE:
                    logger.debug("open_link: Content exceeds max size limit ({} > {})", content_length, MAX_CONTENT_SIZE)
                    return f"Error: Page content exceeds maximum size limit ({MAX_CONTENT_SIZE} bytes)!"

                logger.info("open_link: Successfully extracted content from: {}", url)
                return extracted_content
        except httpx.TimeoutException:
            logger.debug("open_link: Request timed out for: {}", url)
            return "Error: Request timed out."
        except httpx.HTTPStatusError as e:
            logger.debug("open_link: HTTP error {} for URL: {}", e.response.status_code, url)
            return f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        except Exception as e:
            logger.exception("open_link: Unexpected error fetching URL: {}", url)
            return f"Error: Failed to fetch content - {e}"

    @staticmethod
    async def _ignore_message(reason: str) -> str:
        """Logic for ignoring a message.

        Returns a sentinel string that the bot logic can intercept to
        cancel the response.
        """
        logger.info("LLM decided to ignore message. Reason: {}", reason)
        return f"__STOP_RESPONSE__|{reason}"

    async def _read_message_link(self, link: str, context_padding: int) -> str:
        """Fetch a message and its context from a Discord link."""
        # A single, generic error for the LLM so it doesn't hallucinate fixes for permissions it doesn't have.

        if not self.client:
            logger.error("ToolManager: Discord client is not bound.")
            return GENERIC_ERROR

        match = MSG_LINK_PATTERN.search(link)
        if not match:
            logger.debug("ToolManager: Invalid link format received: {}", link)
            return GENERIC_ERROR

        data = match.groupdict()
        channel_id = int(data["channel_id"])
        msg_id = int(data["message_id"])

        try:
            # 1. Resolve Channel
            channel = self.client.get_channel(channel_id)
            if not channel:
                channel = await self.client.fetch_channel(channel_id)

            # 2. Validate Channel Type
            if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
                logger.debug("ToolManager: Unsupported channel type: {}", type(channel))
                return GENERIC_ERROR

            # 3. Fetch Target & Context
            # We fetch the target first to ensure it exists
            target_msg = await channel.fetch_message(msg_id)

            # Fetch surrounding history (async generator -> list)
            context_msgs = [msg async for msg in channel.history(around=target_msg, limit=context_padding)]
            context_msgs.sort(key=lambda m: m.created_at)

            # 4. Format Output
            output = [f"**Context for linked message in #{channel.name}:**\n"]
            for msg in context_msgs:
                marker = " (TARGET)" if msg.id == msg_id else ""
                content = msg.content or "[Attachment/Embed]"
                output.append(f"{marker}[{msg.created_at:%Y-%m-%d %H:%M}] {msg.author.display_name}({msg.author.id}): {content}")

            return "\n".join(output)

        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
            # Catch-all for Discord API errors (Permissions, Deleted messages, Unknown channels)
            logger.warning("ToolManager: Failed to fetch message link '{}'. Reason: {}", link, e)
            return GENERIC_ERROR
        except Exception:
            # Catch-all for unexpected logic errors
            logger.exception("ToolManager: Unexpected error parsing message link '{}'", link)
            return GENERIC_ERROR


tool_manager: ToolManager = ToolManager()
