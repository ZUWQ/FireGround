"""Normalize teammate messages without deciding whether to interrupt an action."""
from __future__ import annotations

import copy
from typing import Any

MESSAGE_PRIORITIES = frozenset({"urgent", "important", "useful"})


def normalize_message(
    info: str | dict[str, Any], *, priority: str | None = None,
    sender: str | None = None,
) -> dict[str, Any]:
    """Return an independent message; legacy text is non-interrupting by default.

    Priority controls scheduling independently of the message's semantic type.
    Unknown metadata is retained for future transports and routing.
    """
    if isinstance(info, str):
        message: dict[str, Any] = {"message": info}
    elif isinstance(info, dict):
        message = copy.deepcopy(info)
    else:
        raise ValueError("Teammate information must be a string or message object")

    text = message.get("message", message.get("content", ""))
    if not isinstance(text, str):
        raise ValueError("Teammate message must contain string message/content")
    selected = priority if priority is not None else message.get("priority", "useful")
    if not isinstance(selected, str) or selected.strip().lower() not in MESSAGE_PRIORITIES:
        raise ValueError("Message priority must be urgent, important or useful")

    message["message"] = text
    message["type"] = message.get("type") or "status_update"
    message["priority"] = selected.strip().lower()
    if sender is not None:
        message["sender"] = sender
    return message
