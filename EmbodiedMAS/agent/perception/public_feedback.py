"""Whitelist projection of internal action feedback for both model modalities."""
from __future__ import annotations

import copy
import math
import re
from numbers import Real
from typing import Any
from uuid import UUID

from EmbodiedMAS.agent.perception.oracle import OracleNames
from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.agent.spatial_units import finite_xy, world_xy_m


_FOLLOWER_FIELDS = {
    "reported_following_npc_ids": "reported_following_npcs",
    "previous_following_npc_ids": "previous_following_npcs",
    "newly_following_npc_ids": "newly_following_npcs",
    "no_longer_reported_npc_ids": "no_longer_reported_npcs",
}
_PUBLIC_FOLLOWERS = frozenset(_FOLLOWER_FIELDS.values()) | {"following_npcs"}
_REASON_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def public_self_xy(value: Any, *, coordinate_unit: str = "uu") -> list[float] | None:
    """Project a caller-identified self pose to meters, respecting public units."""
    if isinstance(value, dict) and "location" in value:
        value = value["location"]
    if coordinate_unit == "m":
        return finite_xy(value)
    if coordinate_unit == "uu":
        return world_xy_m(value)
    return None


def public_reason(value: Any) -> str:
    """Keep machine reason codes, not exception details containing private data."""
    code = str(value).split(":", 1)[0].strip()
    return code if _REASON_CODE.fullmatch(code) else "execution_error"


def _public_reference(value: Any, names: OracleNames) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        identity = UUID(value)
    except ValueError:
        return value
    return names.assign(str(identity), "object") if identity.int else None


def _follower_names(identifiers: Any, names: OracleNames) -> list[str] | None:
    if not isinstance(identifiers, (list, tuple, dict)):
        return None
    identities = [stable_entity_id({"id": value}) for value in identifiers]
    if any(identity is None for identity in identities):
        return None
    return [names.assign(identity, "NPC") for identity in identities]


def _public_followers(value: Any, names: OracleNames) -> list[str] | None:
    if isinstance(value, dict):
        return _follower_names(value, names)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        return None
    return [_public_reference(v, names) for v in value]


def public_action_params(value: Any, names: OracleNames) -> dict[str, Any] | None:
    """Keep named targets and commanded relative travel, never resolved geometry."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ("subtype", "room", "direction")
              if isinstance(value.get(key), str)}
    if isinstance(value.get("target"), str):
        result["target"] = _public_reference(value["target"], names)
    elif isinstance(value.get("target"), dict):
        from EmbodiedMAS.agent.actions.image_targets import ImageTargetError, validate_image_target
        try:
            result["target"] = validate_image_target(value["target"])
        except ImageTargetError:
            pass
    if value.get("subtype") == "relative" and _finite_number(value.get("distance")):
        result["distance"] = value["distance"]
    return result


def _public_data(value: Any, names: OracleNames) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("source", "room", "route_kind", "phase", "verification", "observed_effect", "feedback_protocol"):
        if isinstance(value.get(key), str):
            result[key] = value[key]
    if isinstance(value.get("observed_effect"), dict):
        result["observed_effect"] = public_feedback(value["observed_effect"], names)
    for key in ("execution_feedback", "effect_feedback"):
        if key in value:
            result[key] = public_feedback(value[key], names)
    for key in ("effect_success", "burning_state", "closed_loop", "execution_started", "destroyed", "partial_exploration"):
        if key in value and (value[key] is None or type(value[key]) is bool):
            result[key] = value[key]
    for key in ("duration_sec", "observation_tick", "completed_segments", "next_segment",
                "visited_waypoint_count", "skipped_waypoint_count", "alternative_waypoints_used", "projected_waypoints_used"):
        if key in value and (value[key] is None or _finite_number(value[key])):
            result[key] = value[key]
    for key in ("target", "target_ref"):
        if key in value and (value[key] is None or isinstance(value[key], str)):
            result[key] = _public_reference(value[key], names)
    self_position_fields = ("actual", "actual_end_position", "last_observed_position")
    source_unit = value.get("coordinate_unit", "uu")
    for key in self_position_fields:
        if key in value:
            result[key] = public_self_xy(value[key], coordinate_unit=source_unit)
    if "coordinate_unit" in value or any(key in value for key in self_position_fields):
        # The unit marker also makes a repeated privacy projection idempotent.
        result["coordinate_unit"] = "m"
    if "stop_reason" in value:
        result["stop_reason"] = public_reason(value["stop_reason"])
    if isinstance(value.get("candidate_rejections"), list):
        result["candidate_rejections"] = [public_reason(v) for v in value["candidate_rejections"]]
    completed = value.get("completed_waypoints")
    if isinstance(completed, list) and all(type(v) is int and v >= 0 for v in completed):
        result["completed_waypoints"] = list(completed)
    elif type(completed) is int and completed >= 0:
        result["completed_waypoints"] = completed
    skipped = value.get("skipped_waypoints")
    if isinstance(skipped, list) and all(type(v) is int and v >= 0 for v in skipped):
        result["skipped_waypoints"] = list(skipped)
    if isinstance(value.get("skipped_waypoint_reasons"), list):
        result["skipped_waypoint_reasons"] = [
            {"waypoint_index": item["waypoint_index"], "reason": public_reason(item.get("reason", "unknown"))}
            for item in value["skipped_waypoint_reasons"] if isinstance(item, dict)
            and type(item.get("waypoint_index")) is int and item["waypoint_index"] >= 0]
    for key in _PUBLIC_FOLLOWERS:
        if key in value:
            result[key] = _public_followers(value[key], names)
    for internal, public in _FOLLOWER_FIELDS.items():
        if internal in value and public not in value:
            result[public] = _follower_names(value[internal], names)
    return result


def public_feedback(value: Any, names: OracleNames) -> Any:
    """Detach allowed results/history; internal RPC data and geometry stay private.

    Self-position samples expose XY in meters with a unit marker. Raw history
    remains in UU; projecting public meters again does not rescale. Only
    action_params for navigate/relative retain a commanded distance. Effect checks never expose target coordinates,
    distances to targets, paths, raw execution results, or exception payloads.
    """
    if isinstance(value, (list, tuple)):
        return [public_feedback(entry, names) for entry in value]
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("action", "status", "decision_basis"):
        if key in value and (value[key] is None or isinstance(value[key], str)):
            result[key] = value[key]
    for key in ("step", "action_id"):
        if key in value and (value[key] is None or type(value[key]) is int):
            result[key] = value[key]
    if "reason" in value:
        result["reason"] = public_reason(value["reason"])
    if "data" in value:
        result["data"] = _public_data(value["data"], names)
        if result.get("reason") == "target_destroyed" and result["data"].get("destroyed") is True:
            target = result["data"].get("target_ref", "target")
            # Emit the semantic explanation, never a raw engine exception.
            result["data"]["message"] = f"Target {target!r} has been destroyed and no longer exists."
    if "action_params" in value:
        result["action_params"] = public_action_params(value["action_params"], names)
    if "action_result" in value:
        result["action_result"] = public_feedback(value["action_result"], names)
    if isinstance(value.get("teammate_messages"), list):
        # These are exact snapshots of messages already supplied to the model,
        # not engine feedback; do not reinterpret their content or coordinates.
        result["teammate_messages"] = copy.deepcopy(value["teammate_messages"])
    for key in ("error", "observation_error"):
        if key in value:
            result[key] = public_reason(value[key])
    return result


def public_model_feedback(value: Any, names: OracleNames) -> Any:
    """Use the output action shape in model inputs, after the privacy projection.

    Internal logs retain the legacy action/action_params representation. This
    adapter moves subtype out of parameters only in the model-facing history.
    """
    def canonical(entry: Any) -> Any:
        if isinstance(entry, list):
            return [canonical(item) for item in entry]
        if not isinstance(entry, dict):
            return entry
        if "action" in entry:
            name = entry["action"]
            params = entry.pop("action_params", None) or {}
            if isinstance(name, str):
                subtype = params.pop("subtype", None)
                entry["action"] = {"name": name, "parameters": params}
                if subtype:
                    entry["action"]["subtype"] = subtype
        if "action_result" in entry:
            entry["action_result"] = canonical(entry["action_result"])
        return entry

    return canonical(public_feedback(value, names))
