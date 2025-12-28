"""Message payload utilities for LLM requests.

This module contains functions for building and processing message payloads
for LLM API calls, including support for text, images, and user formatting.
"""

import logging
from typing import Any, NamedTuple

from discord import Message

from .models import MsgNode
from .utils_regex import sanitize_symbols

logger = logging.getLogger(__name__)


class MessagePayloadParams(NamedTuple):
    """Parameters for message payload creation."""

    max_text: int
    max_images: int
    prefix_users: bool
    accept_images: bool
    accept_usernames: bool


def create_message_payload(
    node: MsgNode,
    params: MessagePayloadParams,
) -> dict[str, Any] | None:
    """Create a single message payload from a MsgNode.

    :param node: The message node to process.
    :param params: Parameters for payload creation.
    :return: The message payload or None if empty.
    """
    formatted_text = node.text[: params.max_text] if node.text else ""

    if params.prefix_users and not params.accept_usernames and node.role == "user" and node.user_display_name:
        formatted_text = f"{node.user_display_name}({node.user_id}): {formatted_text}"

    images_to_add: list[dict[str, Any]] = node.images[: params.max_images] if params.accept_images else []

    content: str | list[dict[str, Any]]
    if images_to_add:
        parts: list[dict[str, Any]] = []
        if formatted_text:
            parts.append({"type": "text", "text": formatted_text})
        parts.extend(images_to_add)
        content = parts
    else:
        content = formatted_text

    if not content:
        return None

    message_payload: dict[str, Any]
    if node.role == "user":
        user_payload: dict[str, Any] = {"role": "user", "content": content}
        if params.accept_usernames and node.user_id:
            user_payload["name"] = sanitize_symbols(node.user_display_name or "")[:64] or str(node.user_id)
        message_payload = user_payload
    else:
        message_payload = {"role": "assistant", "content": formatted_text}

    return message_payload


def build_messages_payload(
    message_history: list[Message],
    msg_nodes: dict[int, MsgNode],
    params: MessagePayloadParams,
) -> list[dict[str, Any]]:
    """Build the messages payload for the LLM.

    :param message_history: The history of messages to process.
    :param msg_nodes: The dictionary of processed message nodes.
    :param params: Parameters for payload creation.
    :return: A list of messages for the payload.
    """
    messages_payload: list[dict[str, Any]] = []

    for msg in message_history:
        node = msg_nodes.get(msg.id)
        if not node or node.text is None:
            logger.debug("Empty or missing message node found, skipping...")
            continue

        message_payload = create_message_payload(node=node, params=params)

        if message_payload:
            messages_payload.append(message_payload)

    return messages_payload[::-1]
