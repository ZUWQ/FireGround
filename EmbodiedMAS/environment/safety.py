"""Configured XY-only safety facts for legally observed NPC positions."""
from __future__ import annotations

from numbers import Real
import math
from typing import Any


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def validate_safe_areas(value: Any) -> list[dict[str, dict[str, float]]]:
    """Validate and detach world-unit AABBs; Z is retained but never tested."""
    if not isinstance(value, list) or not value:
        raise ValueError("SafeArea must be a nonempty list")
    result = []
    for area in value:
        if not isinstance(area, dict):
            raise ValueError("SafeArea entries must contain min and max bounds")
        bounds = {}
        for end in ("min", "max"):
            raw = area.get(end)
            if not isinstance(raw, dict) or any(axis not in raw for axis in ("x", "y", "z")):
                raise ValueError(f"SafeArea.{end} must contain x, y and z")
            bounds[end] = {axis: _finite(raw[axis], f"SafeArea.{end}.{axis}")
                           for axis in ("x", "y", "z")}
        if any(bounds["min"][axis] > bounds["max"][axis] for axis in ("x", "y", "z")):
            raise ValueError("SafeArea min bounds must not exceed max bounds")
        result.append(bounds)
    return result


def npc_is_safe(location: Any, safe_areas: list[dict[str, dict[str, float]]]) -> bool:
    """Return current XY membership, including boundaries and ignoring Z.

    Coordinates and bounds use UE world units. Invalid NPC XY is a capture/data
    error, not evidence that a civilian is outside the area.
    """
    try:
        if isinstance(location, dict):
            x, y = location["x"], location["y"]
        elif isinstance(location, (list, tuple)):
            if len(location) not in (2, 3):
                raise ValueError("NPC coordinates must contain XY or XYZ")
            x, y = location[0], location[1]
        else:
            x, y = location.x, location.y
        x, y = _finite(x, "NPC location.x"), _finite(y, "NPC location.y")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("NPC is_safe requires valid observed XY coordinates") from exc
    return any(area["min"]["x"] <= x <= area["max"]["x"]
               and area["min"]["y"] <= y <= area["max"]["y"] for area in safe_areas)
