"""Complete local-oracle snapshots and an identity-keyed target history."""
from __future__ import annotations

import copy
import time
from typing import Any, Awaitable, Callable

from EmbodiedMAS.environment.perception_api import PerceptionInfo, record_visible_entities
from EmbodiedMAS.agent.perception.records import ObservationFrame
from EmbodiedMAS.agent.perception.oracle import names_for_context, normalize_entity, room_for

ObjectList = dict[str, list[dict[str, Any]]]


class PerceptionModule:
    def __init__(self, context: Any, *, interval_sec: float = 2.0) -> None:
        self._context = context
        self._perception_info = PerceptionInfo(context)
        self.interval_sec = float(interval_sec)
        self._names = names_for_context(context)
        self._semantic_maps: list[dict[str, Any]] = []
        self._known: dict[str, dict[str, dict[str, Any]]] = {
            "actor_info": {}, "npc_info": {},
        }

    def set_semantic_maps(self, semantic_maps: list[dict[str, Any]]) -> None:
        self._semantic_maps = copy.deepcopy(semantic_maps)

    def merged_object_list(self) -> ObjectList:
        """Return last observed state by identity; this is not current visibility."""
        return {kind: copy.deepcopy(list(entries.values()))
                for kind, entries in self._known.items()}

    async def capture(
        self, agent_id: bytes | str, *, get_agent_position: Callable[[], Awaitable[Any]],
        tick: int, timeout: float = 5.0, view: str = "front",
        batch_id: str | None = None, action_id: int | None = None,
        capture_kind: str = "front",
    ) -> ObservationFrame:
        started = time.monotonic()
        try:
            result = await self._perception_info.get_perception(agent_id, timeout=timeout)
        except Exception as exc:
            record_visible_entities(self._context, agent_id, None,
                requested_at_monotonic=started, error=exc, modality="text", tick=tick,
                view=view, batch_id=batch_id, action_id=action_id, capture_kind=capture_kind)
            raise
        captured_at = time.monotonic()
        record_visible_entities(self._context, agent_id, result,
            requested_at_monotonic=started, captured_at_monotonic=captured_at,
            modality="text", tick=tick, view=view, batch_id=batch_id,
            action_id=action_id, capture_kind=capture_kind)
        agent_position = await get_agent_position()
        if not isinstance(agent_position, dict):
            raise RuntimeError("Text capture did not produce an agent pose")
        agent_position = copy.deepcopy(agent_position)
        agent_position["room"] = room_for(agent_position, self._semantic_maps)
        current: ObjectList = {"actor_info": [], "npc_info": []}
        for kind in current:
            for source in result[kind]:
                current[kind].append(normalize_entity(
                    source, kind=kind, names=self._names, semantic_maps=self._semantic_maps,
                ))
        entities = tuple(entry for entries in current.values() for entry in entries)
        targets = {entity["target_ref"]: tuple(entity["location"])
                   for entity in entities if entity.get("location") is not None}
        snapshot = ObservationFrame(
            tick=tick, agent_position=copy.deepcopy(agent_position), entities=entities,
            captured_at_monotonic=captured_at, view=view, batch_id=batch_id,
            action_id=action_id, capture_kind=capture_kind, targets=targets,
        )
        # Publish history only after the complete snapshot has been validated.
        for kind, entries in current.items():
            for entry in entries:
                # Execution history owns its state independently of returned frames.
                self._known[kind][entry["target_ref"]] = copy.deepcopy(entry)
        known = {kind: list(entries.values()) for kind, entries in self._known.items()}
        self._perception_info.filter_destroyed_cache(known)
        self._known = {kind: {entry["target_ref"]: entry for entry in entries}
                       for kind, entries in known.items()}
        return snapshot
