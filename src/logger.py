"""Module for handling application logging configuration and utilities."""

import ctypes
import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import ClassVar

from main import logger

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
                headers_copy: dict[str, object] = dict(extra_headers)

                for key in list(headers_copy.keys()):
                    if any(sensitive in key.lower() for sensitive in ("api", "auth", "key", "token")):
                        headers_copy[key] = "REDACTED"

                log_entry["extra_headers"] = headers_copy

            log_message = json.dumps(log_entry, default=str, ensure_ascii=False, indent=4)
            self.logger.info(log_message)
        except (OSError, TypeError, ValueError):
            logger.exception("Failed to log LLM request!")


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to logs based on severity."""

    # ANSI Escape Codes
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text.

        :param record: The log record to format.
        :return: The formatted log string.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_logging() -> None:
    """Configure the root logger and silences noisy libraries."""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())

    logging.basicConfig(level=logging.DEBUG, handlers=[console_handler])

    # Silence noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


# Initialize immediately on import
setup_logging()
request_logger = RequestLogger()
