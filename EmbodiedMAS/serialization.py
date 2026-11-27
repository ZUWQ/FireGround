"""Shared data conversion helpers, independent of prompts and simulator clients."""
from __future__ import annotations

import json
from typing import Any


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        return {
            "x": float(getattr(value, "x")),
            "y": float(getattr(value, "y")),
            "z": float(getattr(value, "z")),
        }
    return str(value)


def compact_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, separators=(", ", ": "))


def location_from_perception(value: Any) -> list[float]:
    if value is None:
        raise ValueError("location is None")
    coords = json_safe(value)
    if not isinstance(coords, dict):
        raise ValueError(f"unsupported location type: {type(value)!r}")
    return [float(coords["x"]), float(coords["y"]), float(coords["z"])]
