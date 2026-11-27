"""Pure action-effect checks using only allowed observations and state readback.

These checks deliberately receive no command/RPC result: an interrupted or
failed command can still reach its goal, and an accepted command can miss it.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .results import action_result


_UNITS_PER_METER = 100.0


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    compact = value.replace("-", "")
    if len(compact) == 32 and all(char in "0123456789abcdefABCDEF" for char in compact):
        return compact.lower()
    return value.casefold()


def fire_effect(observation: Any, target_ref: str, *, modality: str) -> dict[str, Any]:
    """Use the target's latest post-action text observation, never periodic history.

    The latest matching sighting owns the verdict even when its burning state is
    missing. Stable normalized aliases and internal IDs are accepted. Vision has
    no supported fire-state confirmation and must remain unknown.
    """
    data: dict[str, Any] = {"target_ref": target_ref, "effect_success": None}
    if modality != "text":
        return action_result("unknown", "fire_state_unavailable", data)
    identity = _identity(target_ref)
    if identity is None:
        return action_result("unknown", "target_not_observed", data)
    latest: tuple[tuple[float, float, int], Any, Any] | None = None
    for index, frame in enumerate(_field(observation, "surroundings", ()) or ()):
        entities = _field(frame, "entities", ()) or ()
        timestamp = _field(frame, "captured_at_monotonic", _field(frame, "captured_at", index))
        tick = _field(frame, "tick", index)
        try:
            order = (float(timestamp), float(tick), index)
            if not all(math.isfinite(item) for item in order[:2]):
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        matches = [entity for entity in entities if isinstance(entity, dict)
                   and any(_identity(entity.get(key)) == identity
                           for key in ("target_ref", "entity_id", "id", "guid"))]
        if not matches or (latest is not None and order <= latest[0]):
            continue
        states = [entity.get("burning_state") for entity in matches]
        # Conflicting duplicate identities in one frame cannot establish a state.
        state = states[0] if all(type(item) is bool and item == states[0] for item in states) else None
        latest = (order, state, tick)
    if latest is None:
        return action_result("unknown", "target_not_observed", data)
    _, state, tick = latest
    data.update(burning_state=state, observation_tick=tick)
    if type(state) is not bool:
        return action_result("unknown", "burning_state_unknown", data)
    data["effect_success"] = not state
    return action_result("failed" if state else "completed",
                         "target_still_burning" if state else "fire_extinguished", data)


def _reported_followers(state: Any) -> dict[str, Any] | None:
    reported = state.get("following_NPC") if isinstance(state, dict) else None
    if not isinstance(reported, dict):
        return None
    if any(_identity(key) in (None, "0" * 32) for key in reported):
        return None
    return copy.deepcopy(reported)


def follow_effect(self_state: dict[str, Any] | None, *,
                  previous_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report followers from GetSelfState; a follow RPC supplies only a bool.

    Compare self-state snapshots to identify additions without inventing a
    recipient list. An unchanged nonempty list confirms existing followers only.
    Health labels are reported verbatim, not interpreted as physical movement,
    survival, or successful evacuation.
    """
    before = _reported_followers(previous_state)
    after = _reported_followers(self_state)
    data: dict[str, Any] = {
        "source": "GetSelfState.following_NPC",
        "reported_following_npc_ids": list(after) if after is not None else None,
        "following_npcs": after,
        "previous_following_npc_ids": list(before) if before is not None else None,
        "newly_following_npc_ids": None,
        "no_longer_reported_npc_ids": None,
        "effect_success": None,
    }
    if after is None:
        return action_result("unknown", "following_state_unavailable", data)
    if before is not None:
        previous_ids = {_identity(key) for key in before}
        current_ids = {_identity(key) for key in after}
        data["newly_following_npc_ids"] = [key for key in after if _identity(key) not in previous_ids]
        data["no_longer_reported_npc_ids"] = [key for key in before if _identity(key) not in current_ids]
    data["effect_success"] = bool(after)
    if not after:
        return action_result("failed", "no_followers_reported", data)
    reason = ("following_reported" if before is None else
              "follow_confirmed" if data["newly_following_npc_ids"] else "following_unchanged")
    return action_result("completed", reason, data)


def _coordinates(value: Any) -> list[float] | None:
    """Validate XY; retain optional finite Z for raw diagnostics only."""
    if isinstance(value, dict):
        if not all(axis in value for axis in ("x", "y")):
            value = value.get("location")
        if isinstance(value, dict):
            value = [value.get(axis) for axis in ("x", "y", "z")]
    if not isinstance(value, (list, tuple)):
        value = [getattr(value, axis, None) for axis in ("x", "y", "z")]
    if len(value) < 2 or any(isinstance(item, bool) for item in value[:2]):
        return None
    try:
        result = [float(item) for item in value[:2]]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    if len(value) > 2:
        try:
            height = float(value[2])
            if not isinstance(value[2], bool) and math.isfinite(height):
                result.append(height)
        except (TypeError, ValueError, OverflowError):
            pass
    return result


def navigation_effect(position: Any, target_location: Any, tolerance_m: float) -> dict[str, Any]:
    """Arrival depends only on final XY distance within the metre tolerance."""
    actual, target = _coordinates(position), _coordinates(target_location)
    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError, OverflowError):
        tolerance = float("nan")
    valid_tolerance = not isinstance(tolerance_m, bool) and math.isfinite(tolerance) and tolerance >= 0.0
    data: dict[str, Any] = {
        "actual": actual, "target": target, "coordinate_unit": "uu",
        "distance_m": None,
        "tolerance_m": tolerance if valid_tolerance else None, "effect_success": None,
    }
    if not valid_tolerance:
        return action_result("unknown", "invalid_navigation_tolerance", data)
    if target is None:
        return action_result("unknown", "navigation_target_unavailable", data)
    if actual is None:
        return action_result("unknown", "position_unavailable", data)
    distance = math.hypot(actual[0] - target[0], actual[1] - target[1]) / _UNITS_PER_METER
    success = distance <= tolerance or math.isclose(distance, tolerance, rel_tol=0.0, abs_tol=1e-9)
    data.update(distance_m=distance, effect_success=success)
    return action_result("completed" if success else "failed",
                         "target_reached" if success else "target_not_reached", data)
