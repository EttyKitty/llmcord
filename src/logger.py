import json
import logging
import os
from datetime import datetime
from typing import Any

# --- Fix for Windows Colors ---
if os.name == "nt":
    os.system("")  # This simple hack enables ANSI support in Windows CMD


class RequestLogger:
    """
    Handles logging of LLM request payloads to a file.
    Outputs pretty-printed JSON for readability.
    """

    def __init__(self, filename: str = "logs/llm_requests.json"):
        self.logger = logging.getLogger("request_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def log(self, payload: dict[str, Any]) -> None:
        """
        Sanitizes and logs the request payload as a pretty-printed JSON object.
        """
        try:
            # Create a shallow copy
            log_entry = payload.copy()

            # Inject timestamp for context
            log_entry["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Redact sensitive headers
            if "extra_headers" in log_entry and log_entry["extra_headers"]:
                headers = log_entry["extra_headers"].copy()
                for key in headers:
                    if any(sensitive in key.lower() for sensitive in ("api", "auth", "key", "token")):
                        headers[key] = "REDACTED"
                log_entry["extra_headers"] = headers

            log_message = json.dumps(log_entry, default=str, ensure_ascii=False, indent=4)

            self.logger.info(log_message)
        except Exception as e:
            logging.error(f"Failed to log LLM request: {e}")


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

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_logging():
    """Configures the root logger and silences noisy libraries."""
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
