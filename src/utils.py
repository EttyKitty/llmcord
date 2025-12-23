import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import discord
import tiktoken

VISION_MODEL_TAGS = (
    "claude",
    "gemini",
    "gemma",
    "gpt-4",
    "gpt-5",
    "grok-4",
    "llama",
    "llava",
    "mistral",
    "o3",
    "o4",
    "vision",
    "vl",
)
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

REGEX_USER_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
REGEX_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
REGEX_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
REGEX_MULTI_SPACE = re.compile(r" {2,}")
REGEX_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\r?\n|$)")

TYPOGRAPHY_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "…": "...",
    }
)

TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    user_display_name: Optional[str] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def clean_response(text: str) -> str:
    text = text.translate(TYPOGRAPHY_MAP)
    text = REGEX_MULTI_SPACE.sub(" ", text)
    text = REGEX_EXCESSIVE_NEWLINES.sub("\n\n", text)
    text = REGEX_TRAILING_WHITESPACE.sub("", text)
    return text.strip()
