"""Strict global oracle contract; incomplete local caches are never substituted."""
from __future__ import annotations

import copy
import inspect
import math
import time
from numbers import Real
from typing import Any

from EmbodiedMAS.agent.perception.oracle import names_for_context, normalize_entity, room_for
from EmbodiedMAS.agent.perception.records import ObservationFrame
from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.environment.perception_api import PerceptionCaptureError, record_visible_entities


class GlobalObservationUnavailable(PerceptionCaptureError):
    """A complete global state source is unavailable or violates its contract."""


def _number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_entity(entity: Any, *, npc: bool) -> None:
    if not isinstance(entity, dict) or stable_entity_id(entity) is None:
        raise GlobalObservationUnavailable("Global entities require stable identities")
    if not isinstance(entity.get("name"), str) or not entity["name"].strip():
        raise GlobalObservationUnavailable("Global entities require names")
    location = entity.get("location")
    if isinstance(location, dict):
        location = [location.get(axis) for axis in ("x", "y", "z")]
    if not isinstance(location, (list, tuple)) or len(location) != 3 or not all(_number(v) for v in location):
        raise GlobalObservationUnavailable("Global entities require complete finite internal coordinates")
    if not isinstance(entity.get("burning_state"), bool):
        raise GlobalObservationUnavailable("Global entities require current burning_state")
    key = "health" if npc else "burned_percentage"
    if not _number(entity.get(key)) or (not npc and not 0 <= entity[key] <= 100):
        raise GlobalObservationUnavailable(f"Global entities require valid {key}")


def ensure_global_provider(context: Any) -> Any:
    """Install only for explicitly selected global text episodes; no RPC here."""
    provider = getattr(context, "global_oracle_provider", None)
    if provider is None:
        if context is None or not hasattr(context, "conn"):
            raise GlobalObservationUnavailable("Global query_info provider requires a connected context")
        from .provider import QueryInfoGlobalProvider
        provider = context.global_oracle_provider = QueryInfoGlobalProvider(context)
    if not callable(getattr(provider, "get_snapshot", None)):
        raise GlobalObservationUnavailable("Global provider requires a callable get_snapshot")
    return provider


class GlobalOraclePerceptionModule:
    """Capture an explicitly complete snapshot from an authorized global source.

    Provider: context.global_oracle_provider.get_snapshot(timeout=...) or a
    constructor-injected object with the same method. By default the updated UE
    query_info endpoint supplies complete state. Local caches are never substituted.
    """
    observation_scope = "global"

    def __init__(self, context: Any, *, interval_sec: float = 2.0,
                 snapshot_provider: Any = None) -> None:
        self._context = context
        self.interval_sec = float(interval_sec)
        self._provider = snapshot_provider if snapshot_provider is not None else ensure_global_provider(context)
        self._names = names_for_context(context)
        self._semantic_maps: list[dict[str, Any]] = []
        self._known: dict[str, list[dict[str, Any]]] = {"actor_info": [], "npc_info": []}

    def set_semantic_maps(self, semantic_maps: list[dict[str, Any]]) -> None:
        self._semantic_maps = copy.deepcopy(semantic_maps)

    def merged_object_list(self) -> dict[str, list[dict[str, Any]]]:
        return copy.deepcopy(self._known)

    async def capture(self, agent_id: Any, *, get_agent_position: Any, tick: int,
                      timeout: float = 5.0, view: str = "global",
                      batch_id: str | None = None, action_id: int | None = None,
                      capture_kind: str = "global", **kwargs: Any) -> ObservationFrame:
        if view != "global" or capture_kind != "global":
            raise ValueError("Global oracle capture requires the global view and capture kind")
        get_snapshot = getattr(self._provider, "get_snapshot", None)
        if not callable(get_snapshot):
            raise GlobalObservationUnavailable(
                "A complete global oracle provider with get_snapshot is required")
        result = get_snapshot(timeout=timeout)
        if inspect.isawaitable(result):
            result = await result
        if (not isinstance(result, dict) or result.get("scope") != "global"
                or result.get("complete") is not True
                or any(not isinstance(result.get(kind), list) for kind in ("actor_info", "npc_info"))):
            raise GlobalObservationUnavailable("Provider must return an explicitly complete global snapshot")
        # Validate the full response before publishing names, targets, or history.
        identities = set()
        for kind in ("actor_info", "npc_info"):
            for entity in result[kind]:
                _validate_entity(entity, npc=kind == "npc_info")
                identity = stable_entity_id(entity)
                if identity in identities:
                    raise GlobalObservationUnavailable("Global snapshot contains duplicate identities")
                identities.add(identity)
        destroyed = result.get("destroyed_entities", [])
        if not isinstance(destroyed, list):
            raise GlobalObservationUnavailable("Global destroyed_entities must be a list")
        for entity in destroyed:
            identity = stable_entity_id(entity) if isinstance(entity, dict) else None
            if (identity is None or identity in identities or entity.get("destroyed") is not True
                    or not isinstance(entity.get("name"), str) or not entity["name"].strip()
                    or entity.get("kind") not in {"actor_info", "npc_info"}):
                raise GlobalObservationUnavailable("Invalid global destroyed entity")
            identities.add(identity)
        captured_at = result.get("completed_at_monotonic", time.monotonic())
        if not _number(captured_at):
            raise GlobalObservationUnavailable("Global capture requires a finite capture time")
        requested_at = result.get("requested_at_monotonic")
        if requested_at is not None and (not _number(requested_at) or requested_at > captured_at):
            raise GlobalObservationUnavailable("Global capture requires valid sampling times")
        pose = await get_agent_position()
        if not isinstance(pose, dict):
            raise GlobalObservationUnavailable("Global capture did not produce an agent pose")
        pose = copy.deepcopy(pose)
        pose["room"] = room_for(pose, self._semantic_maps)
        current = {kind: [normalize_entity(entity, kind=kind, names=self._names,
                                          semantic_maps=self._semantic_maps)
                          for entity in result[kind]]
                   for kind in ("actor_info", "npc_info")}
        entities = tuple(entry for entries in current.values() for entry in entries)
        from EmbodiedMAS.environment.entity_names import simplify_npc_name
        public_destroyed = []
        for entity in destroyed:
            npc = entity["kind"] == "npc_info"
            name = simplify_npc_name(entity["name"]) if npc else entity["name"]
            public_destroyed.append({"name": self._names.assign(stable_entity_id(entity), name),
                                     "entity_type": "npc" if npc else "actor"})
        metadata = {"destroyed_entities": public_destroyed, "source": result.get("source", "global_oracle_provider")}
        for key in ("requested_at_monotonic", "completed_at_monotonic"):
            if key in result:
                metadata[key] = result[key]
        frame = ObservationFrame(
            tick=tick, agent_position=pose, entities=entities,
            captured_at_monotonic=captured_at, view="global", batch_id=batch_id,
            global_metadata=metadata,
            action_id=action_id, capture_kind="global",
            targets={entity["target_ref"]: tuple(entity["location"]) for entity in entities})
        self._known = copy.deepcopy(current)
        record_visible_entities(self._context, agent_id, result, scope="global",
            source=result.get("source", "global_oracle_provider"),
            modality="text", tick=tick, view=view, batch_id=batch_id,
            action_id=action_id, capture_kind=capture_kind,
            captured_at_monotonic=frame.captured_at_monotonic)
        return frame
