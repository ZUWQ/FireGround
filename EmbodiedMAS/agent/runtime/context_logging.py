"""Small decision-context audit records; deliberately insufficient for full replay."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "decision-context-v1"
OUTCOME_STATUSES = frozenset({
    "accepted", "discarded", "executing", "executed", "rejected", "cancelled", "error",
})
_CONFIG_FIELDS = frozenset({
    "agent_type", "modality", "observation_scope", "record_mode", "cooperation_type",
    "coordination_mode", "time_limit_sec", "max_memory_size", "memory_note_max_chars", "max_tokens",
    "pause_during_llm", "perception_interval_sec", "perception_capture_timeout_sec",
    "navigation_tolerance_m", "vision_backend", "burn_time", "SafeArea",
    "config_version", "code_version", "vision_feedback_protocol",
})
_MESSAGE_FIELDS = ("message_id", "sender", "type", "priority", "message", "content", "captured_at")
_FEEDBACK_DATA_FIELDS = (
    "feedback_protocol", "execution_feedback", "effect_feedback",
    "source", "room", "route_kind", "phase", "verification", "observed_effect",
    "effect_success", "burning_state", "closed_loop", "execution_started", "duration_sec", "observation_tick",
    "completed_segments", "next_segment", "target", "target_ref", "actual",
    "actual_end_position", "last_observed_position", "coordinate_unit", "stop_reason",
    "partial_exploration", "visited_waypoint_count", "skipped_waypoint_count", "alternative_waypoints_used", "projected_waypoints_used",
    "skipped_waypoints", "skipped_waypoint_reasons",
    "candidate_rejections", "completed_waypoints", "following_npcs", "reported_following_npcs",
    "previous_following_npcs", "newly_following_npcs", "no_longer_reported_npcs",
)


def _select(value: Any, fields: tuple[str, ...] | frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: copy.deepcopy(value[key]) for key in fields if key in value}


def _message(value: Any) -> dict[str, Any]:
    return _select(value, _MESSAGE_FIELDS)


def _context(value: Any) -> dict[str, Any]:
    """Retain only the public signal semantics from the actual model input."""
    if not isinstance(value, dict):
        return {}
    result = _select(value, ("message", "source", "reason", "trigger"))
    if isinstance(value.get("sos"), dict):
        result["sos"] = _select(value["sos"], (
            "received", "room", "name", "target_ref", "message", "direction", "captured_at", "received_at"))
    if isinstance(value.get("teammate_message"), dict):
        result["teammate_message"] = _message(value["teammate_message"])
    if isinstance(value.get("events"), list):
        result["events"] = []
        for event in value["events"]:
            if not isinstance(event, dict):
                continue
            item = _select(event, ("trigger", "message", "source", "reason", "other_agent_info"))
            if isinstance(event.get("context"), dict):
                item["context"] = _context(event["context"])
            result["events"].append(item)
    return result


def _action(value: Any) -> Any:
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = _select(value, ("name", "subtype"))
    if isinstance(value.get("parameters"), dict):
        result["parameters"] = _select(value["parameters"], ("target", "room", "direction", "distance"))
    return result


def _execution_feedback(value: Any) -> dict[str, Any]:
    result = _select(value, ("action_id", "step", "status", "reason", "error", "observation_error"))
    if isinstance(value, dict):
        if "action" in value:
            result["action"] = _action(value["action"])
        if isinstance(value.get("data"), dict):
            result["data"] = _select(value["data"], _FEEDBACK_DATA_FIELDS)
    return result


def summarize_input(runtime_input: dict[str, Any]) -> dict[str, Any]:
    """Detach a strict selection of an already assembled model input."""
    observation = runtime_input.get("current_observation") or {}
    signals = runtime_input.get("signal_channel") or {}
    result = _select(runtime_input, ("step", "agent_id", "task_time"))
    result["agent_state"] = _select(runtime_input.get("agent_state"), (
        "location", "orientation", "room", "following_npcs", "resource_state"))
    result["trigger"] = observation.get("decision_trigger")
    result["observation_ref"] = _select(observation, ("batch_id", "action_id", "latest_tick"))
    if isinstance(observation.get("trigger_context"), dict):
        result["trigger_context"] = _context(observation["trigger_context"])
    result["signal_channel"] = {
        "teammate_messages": [_message(item) for item in signals.get("teammate_messages", [])
                              if isinstance(item, dict)],
        "environment_signals": [_context(item) for item in signals.get("environment_signals", [])
                                if isinstance(item, dict)],
        "super_agent_assignment": (None if signals.get("super_agent_assignment") is None else
            _select(signals["super_agent_assignment"], ("task_id", "goal", "target", "room", "message", "priority"))),
    }
    result["previous_execution_feedback"] = _execution_feedback(runtime_input.get("previous_execution_feedback"))
    invalid = runtime_input.get("previous_decision_feedback")
    if isinstance(invalid, dict):
        result["previous_decision_feedback"] = _select(invalid, ("step", "status", "action_executed"))
        if isinstance(invalid.get("errors"), list):
            result["previous_decision_feedback"]["errors"] = [
                _select(error, ("code", "path", "message")) for error in invalid["errors"]
                if isinstance(error, dict)]
    else:
        result["previous_decision_feedback"] = None
    return result


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)) or "agent"


def session_context_path(log_dir: Path, agent_id: str, session_id: str) -> Path:
    return Path(log_dir) / f"decision_context_{_safe_id(agent_id)}_{_safe_id(session_id)}.jsonl"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DecisionContextLogger:
    """One append-only context stream per agent/session, with one metadata row."""

    def __init__(self, log_dir: Path, agent_id: str, session_id: str) -> None:
        self.path = session_context_path(log_dir, agent_id, session_id)
        self.agent_id = str(agent_id)
        self.session_id = str(session_id)
        self._metadata_written = False
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                self._metadata_written = any(json.loads(line).get("event") == "metadata" for line in stream if line.strip())

    def _append(self, event: str, **fields: Any) -> None:
        row = {"schema_version": SCHEMA_VERSION, "run_id": self.session_id,
               "agent_id": self.agent_id, "recorded_at": datetime.now().isoformat(),
               "event": event, **fields}
        serialized = _canonical(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def record_input(self, runtime_input: dict[str, Any], *, system_prompt: str,
                     model: str | None = None, config: dict[str, Any] | None = None) -> None:
        summary = summarize_input(runtime_input)
        if not self._metadata_written:
            selected_config = _select(config, _CONFIG_FIELDS)
            self._append("metadata", system_prompt_hash=_hash(system_prompt), model=model,
                         config=selected_config, config_hash=_hash(_canonical(selected_config)))
            self._metadata_written = True
        self._append("input", **summary)

    def record_outcome(self, step: int, status: str, *, reason: str | None = None,
                       **details: Any) -> None:
        if status not in OUTCOME_STATUSES:
            raise ValueError(f"Unsupported decision-context outcome: {status!r}")
        fields = _select(details, ("action", "execution_status", "end_reason", "error_type",
                                  "event_revision", "action_id", "llm_elapsed_sec"))
        if "action" in fields:
            fields["action"] = _action(fields["action"])
        self._append("outcome", step=step, status=status, reason=reason, **fields)
