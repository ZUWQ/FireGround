"""Raw camera input for visual processors; processed frames use records.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualFrame:
    rgb_path: str
    depth_path: str | None
    agent_position: dict[str, Any]
    world_position: tuple[float, float, float]
    yaw_degrees: float
    captured_at_monotonic: float
    view: str = "front"
    batch_id: str | None = None
    capture_kind: str = "front"
    camera_metadata: dict[str, Any] | None = None
