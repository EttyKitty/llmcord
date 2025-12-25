import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal, Optional
from openai.types.chat import ChatCompletionContentPartImageParam


import discord

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


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[ChatCompletionContentPartImageParam] = field(default_factory=list)

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
