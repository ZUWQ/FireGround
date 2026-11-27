"""Raw episode monitoring, reusing the legacy UE metric queries.

This collector owns no policy, score formula, baseline or termination rule. Its
oracle data is written only to evaluation files and never returned to an agent.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import tongsim as ts

from EmbodiedMAS.environment.safety import npc_is_safe, validate_safe_areas
from EmbodiedMAS.serialization import json_safe

_STREAMS = {"world_states", "models", "actions", "observations", "lifecycle"}
_QUERIES = {
    "property": "get_obj_residual", "fire": "get_burned_area",
    "npc_health": "get_npc_health", "npc_positions": "get_npc_postions",
    "registry": "query_info",
}
_QUERY_TIMEOUT_SEC = 3.0


def record_event(context: Any, stream: str, payload: dict[str, Any]) -> None:
    """No-op outside a monitored episode; do not alter model-visible data."""
    monitor = getattr(context, "_emas_monitor", None)
    if monitor is not None:
        monitor.record(stream, payload)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value: Any) -> Any:
    """Preserve invalid raw numbers as explicit strings rather than invalid JSON."""
    value = json_safe(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


class EpisodeMonitor:
    def __init__(self, context: Any, output_dir: Path | str, run_id: str,
                 structure: str, config: dict[str, Any], sample_interval_sec: float = 1.0):
        if not _finite(sample_interval_sec) or sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be positive and finite")
        if getattr(context, "_emas_monitor", None) is not None:
            raise RuntimeError("This context already has an episode monitor")
        self.context, self.output_dir, self.run_id = context, Path(output_dir), str(run_id)
        self.structure, self.config = str(structure), copy.deepcopy(config)
        self._pause_controller = (getattr(context, "_emas_vision_pause_controller", None)
                                  or getattr(context, "_emas_t1_pause_controller", None))
        self._pause_changed = asyncio.Event()
        areas = config.get("SafeArea")
        if areas is None:
            from EmbodiedMAS.agent.config.environment_config import load_environment_config
            areas = load_environment_config()["SafeArea"]
        self.safe_areas = validate_safe_areas(areas)
        self.sample_interval_sec = float(sample_interval_sec)
        self.started_at = self.deadline = None
        self.first_zero_fire_time = None
        self.first_zero_fire_elapsed_sec = None
        self.first_zero_fire_sample_id = None
        self.agents: dict[str, dict] = {}
        self.initial_snapshot: dict | None = None
        self.pre_ignition_snapshot: dict | None = None
        self.final_snapshot: dict | None = None
        self._sample_id = 0
        self._counts = {name: 0 for name in _STREAMS}
        self._model_events: dict[tuple, dict] = {}
        self._model_event_counts: dict[str, int] = {}
        self._action_events: dict[tuple, list[dict]] = {}
        self._visible_objects: set[str] = set()
        self._snapshot_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopped = self._closed = False
        self.termination_reason: str | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create all streams so an empty episode still has a complete manifest.
        for stream in _STREAMS:
            (self.output_dir / f"{stream}.jsonl").touch(exist_ok=True)
        self._write_metadata()
        self.record("lifecycle", {"event": "monitor_created", "structure": self.structure,
            "config": self.config, "safe_areas": self.safe_areas,
            "sample_interval_sec": self.sample_interval_sec, "position_unit": "UE_UU"})
        context._emas_monitor = self

    def _clock_payload(self) -> dict:
        if self._pause_controller is None:
            return {}
        state = self._pause_controller.snapshot()
        return {"clock": "active_wall_seconds", "pause_intervals": state["pause_intervals"],
                "total_paused_sec": state["total_paused_sec"], "pause_state": state["pause_state"],
                "pause_ledger_complete": (state.get("ledger_complete", True)
                    and not state["paused"] and state["pause_state"] in {"running", "closed"}),
                **({"pause_protocol": "vision_grace_pause_v1", "model_grace_active_sec": 20.0,
                     "model_attempt_wall_timeout_sec": 180.0, "model_max_attempts": 2}
                    if getattr(self.context, "_emas_vision_pause_controller", None) is self._pause_controller else {})}

    def on_pause_changed(self, controller: Any) -> None:
        self._pause_controller = controller
        self._pause_changed.set()
        self.record("lifecycle", {"event": "model_pause_state", **controller.snapshot()})
        self._write_metadata()

    def _write_metadata(self) -> None:
        self._write_json("metadata.json", {
            "schema_version": 1, "run_id": self.run_id, "structure": self.structure,
            "config": self.config, "sample_interval_sec": self.sample_interval_sec,
            "position_unit": "UE_UU", "clock": "monotonic_wall_seconds",
            "safe_areas": self.safe_areas, "agents": self.agents,
            "query_timeout_sec": _QUERY_TIMEOUT_SEC,
            "started_at_monotonic": self.started_at, "deadline_monotonic": self.deadline,
            "source": "legacy metric UE RPCs; raw data only",
            **self._clock_payload(),
        })

    def _write_json(self, name: str, value: dict) -> None:
        (self.output_dir / name).write_text(json.dumps(_clean(value), ensure_ascii=False,
            indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def attach_agent(self, engine_guid: Any, agent_id: str, role: str) -> None:
        from EmbodiedMAS.environment.perception_api import _evaluation_identity
        identity = _evaluation_identity(engine_guid)
        if identity is None:
            raise ValueError("Monitor agent requires an engine identity")
        self.agents[identity] = {"agent_id": str(agent_id), "role": str(role)}
        self._write_metadata()
        self.record("lifecycle", {"event": "agent_registered", "engine_guid": identity,
            "agent_id": str(agent_id), "role": str(role)})

    def record(self, stream: str, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        if stream not in _STREAMS:
            raise ValueError(f"Unknown evaluation stream: {stream}")
        record = {**_clean(copy.deepcopy(payload)), "run_id": self.run_id,
                  "recorded_at_monotonic": time.monotonic()}
        with (self.output_dir / f"{stream}.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._counts[stream] += 1
        if stream == "models":
            event = record.get("event")
            if isinstance(event, str):
                self._model_event_counts[event] = self._model_event_counts.get(event, 0) + 1
            if isinstance(event, str) and event.startswith("http_"):
                return  # Raw attempt telemetry must not become extra logical model calls.
            owner = record.get("agent_id", record.get("controller_id", "central"))
            key = (owner, record.get("call_id", record.get("step", self._counts[stream])))
            # All events remain on disk. Count each call once, using its latest fields.
            self._model_events.setdefault(key, {}).update(record)
        elif stream == "actions":
            owner = record.get("agent_id", record.get("robot_id"))
            action_id = record.get("action_id")
            if owner is not None and action_id is not None:
                self._action_events.setdefault((owner, action_id), []).append(record)
        elif stream == "observations" and record.get("valid", True):
            for value in record.get("visible_object_ids", record.get("visible_entity_ids", [])) or []:
                if value is not None:
                    self._visible_objects.add(str(value))

    @staticmethod
    def _validate(name: str, value: Any) -> None:
        if name == "property":
            if not _finite(value):
                raise ValueError("Property total must be a finite number")
        elif name == "fire":
            if not isinstance(value, dict) or not _finite(value.get("burning_num")) or value["burning_num"] < 0:
                raise ValueError("Fire response requires a nonnegative burning_num")
        elif name == "npc_health":
            if not isinstance(value, dict) or any(not _finite(item) for item in value.values()):
                raise ValueError("NPC health must map identities to finite values")
        elif name == "npc_positions":
            if not isinstance(value, dict):
                raise ValueError("NPC positions must be an identity mapping")
        elif name == "registry":
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError("Registry must be a list of objects")

    async def _query(self, name: str, method: str) -> dict:
        result = {"requested_at_monotonic": time.monotonic(), "valid": False,
                  "data": None, "error": None}
        try:
            rpc = getattr(ts.UnaryAPI, method)
            async with asyncio.timeout(_QUERY_TIMEOUT_SEC):
                raw = await getattr(rpc, "__wrapped__", rpc)(self.context.conn)
            result["data"] = _clean(raw)
            self._validate(name, raw)
            result["valid"] = True
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["completed_at_monotonic"] = time.monotonic()
        return result

    def _npcs(self, queries: dict) -> list[dict]:
        health, positions = queries["npc_health"], queries["npc_positions"]
        hp = health["data"] if health["valid"] else {}
        xyz = positions["data"] if positions["valid"] else {}
        initial = self.initial_snapshot or self.pre_ignition_snapshot or {}
        ids = set(hp) | set(xyz) | {item["id"] for item in initial.get("npcs", [])}
        result = []
        for identity in sorted(ids):
            entry = {"id": identity, "health": hp.get(identity), "health_valid": identity in hp,
                     "position": xyz.get(identity), "position_valid": False, "is_safe": None,
                     "safety_valid": False, "safety_error": None}
            if identity in xyz:
                try:
                    entry["is_safe"] = npc_is_safe(xyz[identity], self.safe_areas)
                    entry["position_valid"] = entry["safety_valid"] = True
                except ValueError as exc:
                    entry["safety_error"] = str(exc)
            result.append(entry)
        return result

    @staticmethod
    def _object_catalog(registry: dict) -> dict:
        from EmbodiedMAS.environment.perception_api import _evaluation_identity, _load_object_list, _lookup_object_list_row
        objects = []
        if registry["valid"]:
            catalog = _load_object_list()
            for entry in registry["data"]:
                row = _lookup_object_list_row(str(entry.get("name", "")), catalog)
                identity = _evaluation_identity(entry.get("id"))
                if row is not None and row.get("read") == "1" and identity is not None:
                    objects.append({"id": identity, "name": entry.get("name"),
                                    "destroyed": entry.get("destroyed")})
        return {"valid": registry["valid"], "objects": objects,
                "selection": "ObjectList.csv read=1; initial catalog is not an observation"}

    def _pause_limit(self, state: dict) -> float:
        limit = state.get("limit_sec")
        if limit is None:
            limit = getattr(getattr(self._pause_controller, "runtime", None), "_time_limit_sec", None)
        return float("inf") if limit is None else limit

    async def capture_snapshot(self, phase: str) -> dict:
        async with self._snapshot_lock:
            requested = time.monotonic()
            queries = dict(zip(_QUERIES, await asyncio.gather(
                *(self._query(name, rpc) for name, rpc in _QUERIES.items()))))
            completed = time.monotonic()
            self._sample_id += 1
            snapshot = {"sample_id": self._sample_id, "phase": str(phase),
                "requested_at_monotonic": requested, "completed_at_monotonic": completed,
                "elapsed_sec": None if self.started_at is None else completed - self.started_at,
                "after_deadline_sec": None if self.deadline is None else max(0.0, completed - self.deadline),
                "queries": queries, "npcs": self._npcs(queries),
                "object_catalog": self._object_catalog(queries["registry"])}
            if self._pause_controller is not None:
                state = self._pause_controller.snapshot(at_wall=completed)
                snapshot.update(clock="active_wall_seconds", active_elapsed_sec=(
                    state["elapsed_sec"] if self.started_at is not None else None),
                    active_after_deadline_sec=(max(0.0, state["elapsed_sec"] - self._pause_limit(state))
                                              if self.started_at is not None else None))
            if phase == "pre_ignition" and self.pre_ignition_snapshot is None:
                self.pre_ignition_snapshot = copy.deepcopy(snapshot)
            if phase == "initial" and self.initial_snapshot is None:
                self.initial_snapshot = copy.deepcopy(snapshot)
            if phase == "final":
                self.final_snapshot = copy.deepcopy(snapshot)
            fire = queries["fire"]
            in_episode = self.started_at is not None and requested >= self.started_at and phase != "pre_ignition"
            within_deadline = self.deadline is None or fire["completed_at_monotonic"] <= self.deadline
            if self._pause_controller is not None and self.started_at is not None:
                fire_state = self._pause_controller.snapshot(at_wall=fire["completed_at_monotonic"])
                within_deadline = fire_state["elapsed_sec"] <= self._pause_limit(fire_state)
            if (self.first_zero_fire_time is None and in_episode and within_deadline
                    and fire["valid"] and fire["data"]["burning_num"] == 0):
                self.first_zero_fire_time = fire["completed_at_monotonic"]
                self.first_zero_fire_elapsed_sec = self.first_zero_fire_time - self.started_at
                if self._pause_controller is not None:
                    self.first_zero_fire_elapsed_sec -= self._pause_controller.paused_duration_at(self.first_zero_fire_time)
                self.first_zero_fire_sample_id = self._sample_id
            self.record("world_states", snapshot)
            return snapshot

    async def start(self, *, started_at: float, deadline: float | None = None) -> None:
        if self.started_at is not None or self._stopped or self._closed:
            raise RuntimeError("Use a fresh monitor for each episode")
        if not _finite(started_at) or (deadline is not None and (not _finite(deadline) or deadline < started_at)):
            raise ValueError("Invalid episode clock")
        self.started_at, self.deadline = float(started_at), deadline
        self._write_metadata()
        self.record("lifecycle", {"event": "episode_started", "started_at_monotonic": self.started_at,
            "deadline_monotonic": self.deadline})
        await self.capture_snapshot("initial")
        self._task = asyncio.create_task(self._sample_loop(), name=f"evaluation-{self.run_id}")

    async def _sample_loop(self) -> None:
        if self._pause_controller is not None:
            await self._sample_loop_active()
            return
        next_sample = time.monotonic() + self.sample_interval_sec
        while not self._stop_event.is_set():
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            wake = next_sample if self.deadline is None else min(next_sample, self.deadline)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=max(0.0, wake - time.monotonic()))
                break
            except TimeoutError:
                pass
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            await self.capture_snapshot("periodic")
            # Skip missed intervals; do not build a backlog of metric RPCs.
            while next_sample <= time.monotonic():
                next_sample += self.sample_interval_sec

    async def _sample_loop_active(self) -> None:
        """Count only confirmed running time; RPC timeout clocks stay real."""
        next_sample = self._pause_controller.snapshot()["elapsed_sec"] + self.sample_interval_sec
        while not self._stop_event.is_set():
            self._pause_changed.clear()
            state = self._pause_controller.snapshot()
            remaining = state["remaining_sec"]
            if not state["paused"] and remaining is not None and remaining <= 0:
                break
            if not state["paused"] and state["elapsed_sec"] >= next_sample:
                await self.capture_snapshot("periodic")
                elapsed = self._pause_controller.snapshot()["elapsed_sec"]
                while next_sample <= elapsed:
                    next_sample += self.sample_interval_sec
                continue
            timeout = None if state["paused"] else max(0.0, min(
                next_sample - state["elapsed_sec"], remaining if remaining is not None else float("inf")))
            stop = asyncio.create_task(self._stop_event.wait())
            changed = asyncio.create_task(self._pause_changed.wait())
            try:
                await asyncio.wait({stop, changed}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (stop, changed):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(stop, changed, return_exceptions=True)

    async def stop(self, *, reason: str) -> None:
        if self._stopped:
            return
        self._stop_event.set()
        if self._task is not None:
            await self._task
        self.termination_reason = str(reason)
        await self.capture_snapshot("final")
        self._stopped = True
        self.record("lifecycle", {"event": "episode_stopped", "reason": self.termination_reason,
            "first_zero_fire_time": self.first_zero_fire_time,
            "first_zero_fire_elapsed_sec": self.first_zero_fire_elapsed_sec,
            "final_sample_id": (self.final_snapshot or {}).get("sample_id")})
        self._write_json("summary.json", self.summary())

    def summary(self) -> dict:
        models = list(self._model_events.values())
        token_totals: dict[str, float] = {}
        known_usage = 0
        for record in models:
            usage = record.get("usage")
            if isinstance(usage, dict):
                known_usage += 1
                for key, value in usage.items():
                    if _finite(value):
                        token_totals[key] = token_totals.get(key, 0) + value
        return {"schema_version": 1, "run_id": self.run_id, "structure": self.structure,
            "termination_reason": self.termination_reason, "started_at_monotonic": self.started_at,
            "deadline_monotonic": self.deadline, "stream_record_counts": self._counts,
            "first_zero_fire_time": self.first_zero_fire_time,
            "first_zero_fire_elapsed_sec": self.first_zero_fire_elapsed_sec,
            "first_zero_fire_sample_id": self.first_zero_fire_sample_id,
            "pre_ignition_sample_id": (self.pre_ignition_snapshot or {}).get("sample_id"),
            "initial_sample_id": (self.initial_snapshot or {}).get("sample_id"),
            "final_sample_id": (self.final_snapshot or {}).get("sample_id"),
            "model_call_count": len(models), "model_calls_with_usage": known_usage,
            "model_calls_without_usage": len(models) - known_usage, "known_usage_totals": token_totals,
            "model_event_counts": {event: self._model_event_counts.get(event, 0)
                for event in ("call_started", "call_completed", "call_error", "call_cancelled")},
            "physical_action_count": sum(any(record.get("event") == "action_started" for record in records)
                                         for records in self._action_events.values()),
            "finished_action_count": sum(any(record.get("event") == "action_finished" for record in records)
                                         for records in self._action_events.values()),
            "initial_object_ids": sorted({item["id"] for item in
                (self.initial_snapshot or self.pre_ignition_snapshot or {}).get("object_catalog", {}).get("objects", [])}),
            "observed_object_ids": sorted(self._visible_objects),
            "note": "Raw collection statistics only; no baseline or benchmark score calculated.",
            **self._clock_payload()}

    async def close(self) -> None:
        if self._closed:
            return
        try:
            if not self._stopped:
                await self.stop(reason="closed")
        finally:
            if getattr(self.context, "_emas_monitor", None) is self:
                delattr(self.context, "_emas_monitor")
            self._closed = True


__all__ = ["EpisodeMonitor", "record_event"]
