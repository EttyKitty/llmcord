"""Module for handling application logging configuration and utilities."""

import asyncio
import ctypes
import functools
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Coroutine, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, ParamSpec, TypeVar, cast

from loguru import logger

# Third-party loggers that produce excessive output and are silenced to WARNING level
NOISY_LOGGERS = [
    "primp",
    "rquest",
    "cookie_store",
    "discord",
    "httpx",
    "httpcore",
    "openai",
    "asyncio",
    "LiteLLM Router",
    "LiteLLM",
    "LiteLLM Proxy",
]

P = ParamSpec("P")
R = TypeVar("R")
# --- Fix for Windows Colors ---
if os.name == "nt":
    # Enables ANSI support in Windows CMD via kernel32 calls
    # Using ctypes avoids subprocess/shell injection risks associated with os.system or subprocess
    try:
        _kernel32 = ctypes.windll.kernel32
        _handle = _kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        _mode = ctypes.c_ulong()

        # ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
        if _kernel32.GetConsoleMode(_handle, ctypes.byref(_mode)):
            _mode.value |= 0x0004
            _kernel32.SetConsoleMode(_handle, _mode)
    except (OSError, AttributeError):
        # Gracefully degrade if ANSI support cannot be enabled
        pass


class Trace:
    """A simple tracing class to measure elapsed time between laps."""

    def __init__(self) -> None:
        """Initialize the Trace instance."""
        self.start = time.perf_counter()
        self.last = self.start

    def lap(self, label: str) -> None:
        """Log the duration since the last lap."""
        now = time.perf_counter()
        duration = now - self.last
        self.last = now
        logger.debug("{}: {:.4f} seconds", label, duration)


@contextmanager
def timer(label: str, level: int = logging.DEBUG) -> Generator[None, None, None]:
    """Log the duration of a block of code."""
    start = time.perf_counter()
    try:
        logger.log(level, "{}...", label)
        yield
    finally:
        duration = time.perf_counter() - start
        logger.log(level, "{} took {:.4f} seconds", label, duration)


def time_performance(label: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log the execution time of synchronous or asynchronous functions."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.perf_counter()
                logger.debug("{}...", label)
                result = await cast("Coroutine[Any, Any, R]", func(*args, **kwargs))
                logger.debug("{} took {:.4f} seconds", label, time.perf_counter() - start)
                return result

            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            logger.debug("{}...", label)
            result = func(*args, **kwargs)
            logger.debug("{} took {:.4f} seconds", label, time.perf_counter() - start)
            return result

        return sync_wrapper

    return decorator


class RequestLogger:
    """Handles logging of LLM request payloads to a file. Outputs pretty-printed JSON for readability."""

    def __init__(self, filename: str = "logs/llm_requests.json") -> None:
        """Initialize the RequestLogger with a file handler.

        :param filename: Path to the log file.
        """
        self.logger = logging.getLogger("request_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    @time_performance("Payload Logging")
    def log(self, payload: Mapping[str, object]) -> None:
        """Sanitize and log the request payload as a pretty-printed JSON object.

        :param payload: The request payload dictionary to log.
        :return: None
        """
        try:
            # Create a shallow copy
            log_entry: dict[str, object] = dict(payload)

            # Inject timestamp for context
            log_entry["_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

            # Redact sensitive headers
            extra_headers = log_entry.get("extra_headers")

            if isinstance(extra_headers, dict):
                headers_copy: dict[str, object] = dict(cast("dict[str, object]", extra_headers))

                for key in list(headers_copy.keys()):
                    if any(sensitive in key.lower() for sensitive in ("api", "auth", "key", "token")):
                        headers_copy[key] = "REDACTED"

                log_entry["extra_headers"] = headers_copy

            log_message = json.dumps(log_entry, default=str, ensure_ascii=False, indent=4)
            self.logger.info(log_message)
        except (OSError, TypeError, ValueError):
            logger.exception("Failed to log the payload!")


def setup_logging() -> None:
    """Configure Loguru to replace standard logging and handle colors/sinks."""
    logger.remove()

    log_format = "<magenta>{time:YYYY-MM-DD HH:mm:ss}</magenta> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <white>{message}</white>"

    logger.level("DEBUG", color="<white>")
    logger.level("INFO", color="<green>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<light-red>")
    logger.level("CRITICAL", color="<bold><red>")

    logger.add(sys.stderr, format=log_format, level="DEBUG", colorize=True)

    for noisy in NOISY_LOGGERS:
        logger.disable(noisy)


# Initialize immediately on import
setup_logging()
request_logger = RequestLogger()
