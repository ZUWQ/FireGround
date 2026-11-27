"""Small episode-local memory shared by individual and central agents."""
from __future__ import annotations

import copy
from typing import Any

DEFAULT_NOTE_MAX_CHARS = 2000


def validate_memory_limits(history_size: int, note_max_chars: int) -> None:
    if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 0:
        raise ValueError("max_memory_size must be a nonnegative integer")
    if isinstance(note_max_chars, bool) or not isinstance(note_max_chars, int) or note_max_chars <= 0:
        raise ValueError("memory_note_max_chars must be a positive integer")


class EpisodeMemory:
    """A bounded chronological history and one model-authored replacement note.

    The runtime supplies history records and decides when an answer is accepted.
    This store does not rank events, interpret plans, or share state across agents.
    """

    def __init__(self, history_size: int, note_max_chars: int = DEFAULT_NOTE_MAX_CHARS):
        validate_memory_limits(history_size, note_max_chars)
        self.history_size = history_size
        self.note_max_chars = note_max_chars
        self.note = ""
        self.history: list[dict[str, Any]] = []

    def apply_note(self, value: str | None) -> None:
        """None preserves the note; a string replaces it, including empty text."""
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError("memory note must be a string or null")
        if len(value) > self.note_max_chars:
            raise ValueError("memory note exceeds memory_note_max_chars")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("memory note must contain valid Unicode") from exc
        self.note = value

    def append(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise ValueError("memory history entry must be an object")
        self.history.append(copy.deepcopy(entry))
        if len(self.history) > self.history_size:
            del self.history[:len(self.history) - self.history_size]

    def update(self, step: int, **fields: Any) -> None:
        """Update the latest retained record for a step; never restore evicted rows."""
        for entry in reversed(self.history):
            if entry.get("step") == step:
                entry.update(copy.deepcopy(fields))
                break

    def snapshot(self) -> dict[str, Any]:
        return {"note": self.note, "note_max_chars": self.note_max_chars,
                "short_term": copy.deepcopy(self.history)}

    def clear(self) -> None:
        self.note = ""
        self.history.clear()


__all__ = ["EpisodeMemory", "DEFAULT_NOTE_MAX_CHARS", "validate_memory_limits"]
