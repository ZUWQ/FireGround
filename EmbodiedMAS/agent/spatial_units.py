"""Spatial units at the model boundary; execution and raw records stay in UE UU."""
from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any

UE_UNITS_PER_METER = 100.0
_POSE_FIELDS = frozenset({"x", "y", "forward", "room", "following_npcs"})


def finite_xy(value: Any) -> list[float] | None:
    """Detach a finite XY pair without changing its units or retaining Z."""
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    elif hasattr(value, "x") and hasattr(value, "y"):
        value = [value.x, value.y]
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        return None
    if any(isinstance(component, bool) or not isinstance(component, Real) for component in value[:2]):
        return None
    try:
        xy = [float(component) for component in value[:2]]
    except (TypeError, ValueError, OverflowError):
        return None
    return xy if all(math.isfinite(component) for component in xy) else None


def world_xy_m(value: Any) -> list[float] | None:
    """Convert one internal UE world position to model-visible meters, once."""
    xy = finite_xy(value)
    return None if xy is None else [component / UE_UNITS_PER_METER for component in xy]


def public_pose_m(pose: dict[str, Any]) -> dict[str, Any]:
    """Copy permitted frame metadata while converting only position to meters."""
    result = {key: copy.deepcopy(value) for key, value in pose.items() if key in _POSE_FIELDS}
    xy = world_xy_m(pose)
    for index, axis in enumerate(("x", "y")):
        if axis in result:
            result[axis] = None if xy is None else xy[index]
    return result
