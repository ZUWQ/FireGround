"""Owned observation snapshots shared by text and vision.

A ready package is published only after all four views and orientation recovery
succeed. Consumers never mutate its frames or pop them from a sensor queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VIEWS = ("front", "left", "back", "right")


@dataclass(frozen=True)
class ObservationFrame:
    tick: int
    agent_position: dict[str, Any]
    entities: tuple[dict[str, Any], ...]
    captured_at_monotonic: float
    view: str = "front"
    batch_id: str | None = None
    action_id: int | None = None
    capture_kind: str = "front"
    rgb_path: str | None = None
    depth_path: str | None = None
    masked_rgb_path: str | None = None
    targets: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    vision_backend: str = "grounded_sam"
    frame_ref: str | None = None
    camera_metadata: dict[str, Any] | None = field(default=None, repr=False)

    global_metadata: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.view not in (*VIEWS, "global") or self.capture_kind not in {"front", "surround", "global"}:
            raise ValueError("Invalid observation view or capture kind")
        if (self.view == "global") != (self.capture_kind == "global"):
            raise ValueError("Global captures require the global view and capture kind")
        if self.capture_kind == "front" and self.view != "front":
            raise ValueError("Periodic observations must face forward")
        if self.capture_kind == "surround" and self.batch_id is None:
            raise ValueError("Surround observations require a batch ID")

    @property
    def image_paths(self) -> tuple[str, ...]:
        """Route RGB only; ending RGB followed by a distinct annotation if present."""
        if self.rgb_path is None:
            return ()
        if (self.vision_backend == "raw_vlm" or self.capture_kind != "surround"
                or not self.masked_rgb_path or self.masked_rgb_path == self.rgb_path):
            return (self.rgb_path,)
        return (self.rgb_path, self.masked_rgb_path)

    @property
    def image_path(self) -> str | None:
        paths = self.image_paths
        return paths[-1] if paths else None


@dataclass(frozen=True)
class DecisionObservation:
    batch_id: str
    reason: str
    action_id: int | None
    surroundings: tuple[ObservationFrame, ...]
    agent_position: dict[str, Any]
    during_action: tuple[ObservationFrame, ...] = ()
    scope: str = "local"

    def __post_init__(self) -> None:
        if self.scope not in {"local", "global"}:
            raise ValueError("Observation scope must be local or global")
        expected_views = VIEWS if self.scope == "local" else ("global",)
        if tuple(frame.view for frame in self.surroundings) != expected_views:
            raise ValueError("A local decision requires exactly front/left/back/right; a global decision requires one global snapshot")
        boundary_kind = "surround" if self.scope == "local" else "global"
        if any(frame.batch_id != self.batch_id or frame.capture_kind != boundary_kind
               or frame.action_id != self.action_id for frame in self.surroundings):
            raise ValueError("Surround views must belong to the same action and batch")
        periodic_kind = "front" if self.scope == "local" else "global"
        if any(frame.action_id != self.action_id or frame.capture_kind != periodic_kind
               for frame in self.during_action):
            raise ValueError("Periodic frames must belong to the completed action")

    @property
    def frames(self) -> tuple[ObservationFrame, ...]:
        return self.during_action + self.surroundings
