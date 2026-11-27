"""Full internal oracle normalization; public fields belong to the adapter."""
from __future__ import annotations

import copy
from typing import Any

from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.environment.entity_names import simplify_npc_name
from EmbodiedMAS.environment.regions import locate_room
from EmbodiedMAS.serialization import location_from_perception


class OracleNames:
    """A context-scoped name registry shared by cooperating agents.

    Register only observed identities. Existing names never change or get reused;
    case-insensitive uniqueness matches the action target resolver.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, str] = {}
        self._used: set[str] = set()

    def assign(self, identity: str | None, original: str) -> str:
        if identity is not None and identity in self._by_id:
            return self._by_id[identity]
        base = str(original or "object").strip() or "object"
        alias, suffix = base, 0
        while alias.casefold() in self._used:
            suffix += 1
            alias = f"{base}_{suffix}"
        self._used.add(alias.casefold())
        if identity is not None:
            self._by_id[identity] = alias
        return alias

    def following_names(self, state: Any) -> list[str] | None:
        reported = state.get("following_NPC") if isinstance(state, dict) else None
        if not isinstance(reported, dict):
            return None
        identities = [stable_entity_id({"id": identity}) for identity in reported]
        if any(identity is None for identity in identities):
            return None
        # Self-state proves these NPCs exist, but supplies no trustworthy names.
        return [self.assign(identity, "NPC") for identity in identities]


def names_for_context(context: Any) -> OracleNames:
    names = getattr(context, "_emas_oracle_names", None)
    if names is None:
        names = OracleNames()
        context._emas_oracle_names = names
    return names


def room_for(value: dict[str, Any], semantic_maps: list[dict[str, Any]]) -> str:
    explicit = value.get("room")
    if isinstance(explicit, str) and explicit.strip() and explicit.strip() != "unknown":
        return explicit.strip()
    return locate_room(semantic_maps, value.get("location", value)) or "unknown"


def normalize_entity(source: dict[str, Any], *, kind: str,
                     names: OracleNames, semantic_maps: list[dict[str, Any]]) -> dict[str, Any]:
    entry = copy.deepcopy(source)
    identity = stable_entity_id(entry)
    original = entry.get("name") or entry.get("entity_name") or ("NPC" if kind == "npc_info" else "object")
    if kind == "npc_info":
        # Global providers and replayed frames share the local NPC name contract.
        entry.setdefault("raw_name", original)
        original = simplify_npc_name(original)
    alias = names.assign(identity, original)
    if identity is not None:
        entry["entity_id"] = identity
    location = entry.get("location")
    if location is not None:
        entry["location"] = ([float(value) for value in location]
                             if isinstance(location, (list, tuple))
                             else location_from_perception(location))
    entry.update(name=alias, entity_name=alias, target_ref=alias,
                 entity_type="actor" if kind == "actor_info" else "npc",
                 room=room_for(entry, semantic_maps))
    return entry
