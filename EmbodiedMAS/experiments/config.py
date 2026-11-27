"""Validate the frozen review scene and preburn clock."""
from __future__ import annotations

import math
from typing import Any

from EmbodiedMAS.environment.safety import validate_safe_areas


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _number(value: Any, label: str, *, minimum: float = 0, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < minimum or (positive and value <= 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)


_UNSET = object()


def resolve_burn_time(scene: dict, explicit: Any = _UNSET, *, label: str = "burn_time") -> float:
    """Use the UE phase transition as the single preburn clock.

    Legacy Python overrides are accepted only when they agree with the final
    merged scene. This is the delay before agent decisions, not robot spawn time.
    """
    scene = _object(scene, "SceneConfig")
    delay = _number(scene.get("FireRateDelayTime"), "SceneConfig.FireRateDelayTime")
    if explicit is _UNSET:
        return delay
    duration = _number(explicit, label)
    if not math.isclose(duration, delay, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{label} ({duration:g}s) must equal SceneConfig.FireRateDelayTime "
                         f"({delay:g}s); omit burn_time to derive it from the scene, "
                         "or update both values together")
    return delay


def resolve_task_burn_time(task: dict) -> float:
    return resolve_burn_time(task.get("scene_config"), task.get("burn_time", _UNSET))


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


SCENE_ROOM_FIELDS = {
    "room_generate": "RoomGenerate",
    "fire_room": "FireRoom",
    "npc_room": "NPCRoom",
}
LEGACY_SCENE_FIELDS = {"SceneID", "FireNum", "NPCNum"}


def _room_array(items: Any, field: str) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError(f"SceneConfig.{field} must be a nonempty room-indexed array")
    for index, item in enumerate(items):
        if field == "RoomGenerate":
            _integer(item, f"SceneConfig.{field}[{index}]")
        elif type(item) is not int or item not in (0, 1):
            raise ValueError(f"SceneConfig.{field}[{index}] must be an integer 0 or 1")


def validate_scene_config(value: Any) -> None:
    """Validate the UE room-indexed schema without changing the source JSON.

    RoomGenerate contains opaque nonnegative house-type codes; occupancy arrays
    are flags, not entity counts. Array slots include any rooms absent from the
    public measured map, so no length is inferred from RoomConfig polygons.
    """
    scene = _object(value, "SceneConfig")
    legacy = set(scene) & LEGACY_SCENE_FIELDS
    if legacy:
        raise ValueError("SceneConfig contains removed fields " + ", ".join(sorted(legacy))
                         + "; use RoomGenerate, FireRoom and NPCRoom from the new UE package")
    for field in SCENE_ROOM_FIELDS.values():
        _room_array(scene.get(field), field)
    if len({len(scene[field]) for field in SCENE_ROOM_FIELDS.values()}) != 1:
        raise ValueError("SceneConfig.RoomGenerate, FireRoom and NPCRoom must have the same length")
    safe_area = _object(scene.get("SafeArea"), "SceneConfig.SafeArea")
    validate_safe_areas([safe_area])
    for key, item in scene.items():
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"SceneConfig.{key} must be finite")


