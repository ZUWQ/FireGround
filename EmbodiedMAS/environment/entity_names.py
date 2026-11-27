"""Semantic NPC names shared by local and global oracle ingestion."""
from __future__ import annotations

import re
from typing import Any


_NPC_PREFIX = re.compile(r"^BP_NPC_", re.IGNORECASE)
_NPC_INSTANCE_SUFFIX = re.compile(r"_C(?:_\d+)?$", re.IGNORECASE)


def simplify_npc_name(value: Any) -> str:
    """Keep the lowercase resident name, excluding UE wrappers and punctuation."""
    if not isinstance(value, str):
        return "npc"
    name = _NPC_PREFIX.sub("", value.strip())
    name = _NPC_INSTANCE_SUFFIX.sub("", name)
    return "".join(character for character in name if character.isalnum()).lower() or "npc"
