"""LLMCord Tools Module.

This module defines the tools available to the LLM and the logic to execute them.
"""

import ipaddress
import json
import logging
import socket
import urllib.parse
from pathlib import Path
from typing import Any, cast

import httpx
import trafilatura
from ddgs import (
    DDGS,  # type: ignore[import-untyped] # DDGS library has incomplete type stubs, remove when fixed upstream
)

MAX_CONTENT_SIZE = 500000  # 50KB limit for fetched content
TOOLS_PATH = Path(__file__).parent / "llm_tools.json"

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages LLM tools and their execution."""

    @staticmethod
    def get_tool_definitions() -> list[dict[str, Any]]:
        """Return the JSON schemas for available tools."""
        with (TOOLS_PATH).open() as f:
            return json.load(f)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Map a tool name to its Python implementation and execute it."""
        try:
            if name == "web_search":
                return await self.web_search(arguments.get("query", ""))
            if name == "open_link":
                return await self.open_link(arguments.get("url", ""))
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

    def _is_safe_host(self, netloc: str) -> bool:
        """Check if the host is safe to access (not localhost or private IP)."""
        # Remove port if present
        host = netloc.split(":")[0].lower()

        # Block localhost variations
        if host in ("localhost", "127.0.0.1", "::1"):
            return False

        # Try to parse as IP
        try:
            ip = ipaddress.ip_address(host)
            # Block private, loopback, link-local IPs
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            # Not an IP, it's a hostname
            # For hostnames, try to resolve and check IPs
            try:
                # Only check the first resolved IP for performance
                addr_info = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
                if addr_info:
                    resolved_ip = addr_info[0][4][0]
                    ip = ipaddress.ip_address(resolved_ip)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return False
            except (socket.gaierror, ValueError):
                # If can't resolve, allow (could be public domain)
                pass

        return True

    async def open_link(self, url: str) -> str:
        """Fetch and extract the main content of a web page with security measures."""
        if not url:
            logger.debug("open_link: No URL provided")
            return "Error: No URL provided."

        logger.debug("open_link: Validating URL: %s", url)

        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug("open_link: Invalid scheme '%s' for URL: %s", parsed.scheme, url)
            return "Error: Invalid URL scheme. Only HTTP and HTTPS are allowed."
        if not parsed.netloc:
            logger.debug("open_link: Missing netloc for URL: %s", url)
            return "Error: Invalid URL format."

        logger.debug("open_link: Checking host safety for: %s", parsed.netloc)
        if not self._is_safe_host(parsed.netloc):
            logger.debug("open_link: Host rejected (localhost or private network): %s", parsed.netloc)
            return "Error: Access to localhost or private networks is not allowed for security reasons."

        logger.info("open_link: URL validation passed, fetching content from: %s", url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.google.com/",
                "Connection": "keep-alive",
            }
            # Follow redirects and enable HTTP/2 for better compatibility with modern sites.
            async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=False, http2=True) as client:
                logger.debug("open_link: Sending HTTP request to: %s", url)
                response = await client.get(url)
                response.raise_for_status()
                logger.debug("open_link: HTTP %d response received", response.status_code)

                raw_html = response.text

                # Extract main content using trafilatura
                logger.debug("open_link: Extracting main content from HTML")
                extracted_content = trafilatura.extract(raw_html, include_comments=False)

                if not extracted_content:
                    logger.debug("open_link: No extractable content found, returning plain text")
                    extracted_content = raw_html[:3000]  # Fallback: first 3KB of raw content

                content_length = len(extracted_content)
                logger.debug("open_link: Response content length: %d bytes", content_length)

                # Limit content size to prevent abuse
                if content_length > MAX_CONTENT_SIZE:
                    logger.debug("open_link: Content exceeds max size limit (%d > %d)", content_length, MAX_CONTENT_SIZE)
                    return "Error: Page content is too large! Opening URL failed! REPORT!"

                logger.info("open_link: Successfully extracted content from: %s", url)
                return extracted_content
        except httpx.TimeoutException:
            logger.debug("open_link: Request timed out for: %s", url)
            return "Error: Request timed out."
        except httpx.HTTPStatusError as e:
            logger.debug("open_link: HTTP error %d for URL: %s", e.response.status_code, url)
            return f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        except Exception as e:
            logger.exception("open_link: Unexpected error fetching URL: %s", url)
            return f"Error: Failed to fetch content - {e}"


tool_manager: ToolManager = ToolManager()
