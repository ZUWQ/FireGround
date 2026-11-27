"""Complete task-state snapshots from the updated UE QueryState endpoint."""
from __future__ import annotations

import asyncio
import copy
import math
import re
import time
from typing import Any

import tongsim as ts

from EmbodiedMAS.agent.perception.global_oracle import GlobalObservationUnavailable, _validate_entity
from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.environment.perception_api import _load_object_list, _lookup_object_list_row
from EmbodiedMAS.serialization import json_safe

_NPC_CLASS = re.compile(r"^BP_NPC_", re.IGNORECASE)


class QueryInfoGlobalProvider:
    """Episode-owned catalog; every snapshot reads new state, never cached facts.

    The registry must cover all task objects/NPCs. Prime after registry refresh
    and before ignition so anonymous destroyed records can be identified by GUID.
    Missing live task IDs are incomplete data, not evidence of destruction.
    """

    def __init__(self, context: Any) -> None:
        self._conn = context.conn
        self._catalog: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _prepare(self, rows: Any, *, initial: bool = False) -> tuple[dict, dict]:
        if not isinstance(rows, list) or not rows:
            raise GlobalObservationUnavailable("query_info returned an empty or invalid global registry")
        objects = _load_object_list()
        if not objects:
            raise GlobalObservationUnavailable("Global observation requires a nonempty ObjectList catalog")
        catalog = {} if initial else copy.deepcopy(self._catalog)
        live = {"actor_info": [], "npc_info": []}
        seen: set[str] = set()
        for source in rows:
            if not isinstance(source, dict):
                raise GlobalObservationUnavailable("query_info returned an invalid actor record")
            identity = stable_entity_id(source)
            if identity is None or identity in seen:
                raise GlobalObservationUnavailable("query_info requires unique stable actor identities")
            seen.add(identity)
            destroyed = source.get("destroyed")
            if not isinstance(destroyed, bool):
                raise GlobalObservationUnavailable("query_info requires an explicit destroyed flag")
            previous = catalog.get(identity)
            if destroyed:
                if previous is None:
                    raise GlobalObservationUnavailable(
                        "Unknown destroyed actor; initialize the global catalog before ignition")
                previous["destroyed"] = True
                continue
            raw_name = source.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise GlobalObservationUnavailable("query_info requires actor names")
            raw_name = raw_name.strip()
            class_name = str(source.get("class_path", "")).rsplit(".", 1)[-1]
            row = _lookup_object_list_row(raw_name, objects)
            npc = bool(_NPC_CLASS.match(raw_name) or _NPC_CLASS.match(class_name))
            kind = "npc_info" if npc else ("actor_info" if row and row.get("read") == "1" else None)
            if previous and previous.get("destroyed"):
                raise GlobalObservationUnavailable("Destroyed GUID reappeared; use a fresh episode context after reset")
            if previous and previous["kind"] != kind:
                raise GlobalObservationUnavailable("Actor category changed within a global episode")
            name = raw_name if npc or kind is None else row.get("simplified_name", "").strip()
            metadata = {"id": identity, "name": name, "raw_name": raw_name,
                        "kind": kind, "destroyed": False}
            catalog[identity] = metadata
            if kind is None:
                continue
            location = json_safe(source.get("location"))
            if isinstance(location, dict):
                location = [location.get(axis) for axis in ("x", "y", "z")]
            entry = {**metadata, "location": location, "burning_state": source.get("burning_state")}
            key = "health" if npc else "burned_percentage"
            entry[key] = source.get(key)
            _validate_entity(entry, npc=npc)
            live[kind].append(entry)
        missing = [identity for identity, entry in catalog.items()
                   if entry["kind"] is not None and not entry["destroyed"] and identity not in seen]
        if missing:
            raise GlobalObservationUnavailable("query_info omitted live task entities from the complete snapshot")
        if not any(entry["kind"] is not None for entry in catalog.values()):
            raise GlobalObservationUnavailable("query_info contains no recognized task objects or NPCs")
        destroyed_entities = [copy.deepcopy(entry) for entry in catalog.values()
                              if entry["kind"] is not None and entry["destroyed"]]
        return {"scope": "global", "complete": True, **live,
                "destroyed_entities": destroyed_entities, "source": "query_info"}, catalog

    def prime_registry(self, rows: list[dict]) -> None:
        """Use the startup's successful post-refresh query; perform no extra RPC."""
        _, catalog = self._prepare(rows, initial=True)
        self._catalog = catalog

    async def get_snapshot(self, *, timeout: float = 5.0) -> dict:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Global snapshot timeout must be positive and finite")
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    started = time.monotonic()
                    rpc = ts.UnaryAPI.query_info
                    rows = await getattr(rpc, "__wrapped__", rpc)(self._conn)
                    completed = time.monotonic()
                    result, catalog = self._prepare(rows)
                    result.update(requested_at_monotonic=started, completed_at_monotonic=completed)
                    self._catalog = catalog
                    return result
        except GlobalObservationUnavailable:
            raise
        except Exception as exc:
            raise GlobalObservationUnavailable(f"Global query_info capture failed: {exc}") from exc
