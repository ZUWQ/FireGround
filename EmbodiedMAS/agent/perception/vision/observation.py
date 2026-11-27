"""Capture one complete visual frame without queues or target caches."""
from __future__ import annotations

import copy
import math
import time
from typing import Any

from EmbodiedMAS.environment.perception_api import PerceptionInfo
from EmbodiedMAS.agent.perception.records import ObservationFrame
from EmbodiedMAS.agent.visual_backends import uses_detections, supports_image_targets
from .frame import VisualFrame
from .backends import create_backend


def default_detection_labels():
    """Load the detection vocabulary only when the detector backend requests it."""
    from EmbodiedMAS.agent.perception.vision_pipeline.vision_recognition import default_detection_labels as labels
    return labels()


class VisionPerceptionModule:
    def __init__(self, context: Any, *, interval_sec: float = 2.0,
                 backend: Any = "grounded_sam") -> None:
        self._context = context
        self.backend = create_backend(backend) if isinstance(backend, str) else backend
        self._perception_info = PerceptionInfo(context)
        self.interval_sec = float(interval_sec)

    async def capture(
        self, agent_id: Any, *, get_agent_position: Any, save_images: Any,
        tick: int, name_suffix: str | None = None, timeout: float = 5.0,
        view: str = "front", batch_id: str | None = None,
        action_id: int | None = None, capture_kind: str = "front",
    ) -> ObservationFrame:
        images = await save_images(save_rgb=True, save_depth=True)
        if not isinstance(images, dict) or not images.get("rgb"):
            raise RuntimeError("Current visual capture did not produce an RGB image")
        rgb_path, depth_path = images["rgb"], images.get("depth")
        camera_metadata = copy.deepcopy(images.get("camera_metadata"))
        if uses_detections(getattr(self.backend, "name", None)):
            if not depth_path:
                raise RuntimeError("Grounded-SAM visual capture requires paired depth")
            mode = (camera_metadata or {}).get("depth_mode")
            if mode is not None and (isinstance(mode, bool) or mode not in (1, 3)):
                raise ValueError("Visual grounding supports only linear (1) or view-space Z (3) depth")
        captured_at = float((camera_metadata or {}).get("captured_at_monotonic", time.monotonic()))
        agent_position = await get_agent_position()
        if not isinstance(agent_position, dict):
            raise RuntimeError("Visual capture did not produce an agent pose")
        forward = agent_position.get("forward")
        if forward is None:
            raise RuntimeError("Visual projection requires the capture heading")
        fx, fy = ((forward["x"], forward["y"]) if isinstance(forward, dict)
                  else (forward.x, forward.y))
        frame = VisualFrame(
            rgb_path=rgb_path, depth_path=depth_path,
            agent_position=copy.deepcopy(agent_position),
            world_position=tuple(float(agent_position[key]) for key in ("x", "y", "z")),
            yaw_degrees=math.degrees(math.atan2(fy, fx)),
            captured_at_monotonic=captured_at, view=view,
            batch_id=batch_id, capture_kind=capture_kind, camera_metadata=camera_metadata,
        )
        backend_name = getattr(self.backend, "name", None)
        if uses_detections(backend_name):
            labels = default_detection_labels()
        else:
            labels = []
        frame_ref = None
        if supports_image_targets(backend_name):
            from .image_grounding import validate_camera_metadata
            validate_camera_metadata(camera_metadata)
            if not depth_path:
                raise RuntimeError("Pixel-capable visual capture requires paired depth")
            owner = agent_id.hex() if isinstance(agent_id, bytes) else str(agent_id)
            frame_ref = f"{owner}/{batch_id or f'action-{action_id}'}/{view}/t{tick}"
        visibility_metadata = dict(modality="vision", tick=tick, view=view,
            batch_id=batch_id, action_id=action_id, capture_kind=capture_kind,
            image_captured_at_monotonic=captured_at, rgb_path=str(rgb_path))
        # Evaluation visibility is an independent side channel. Backend flags
        # cannot make current Oracle labels part of visual perception.
        await self._perception_info.capture_evaluation_visibility(
            agent_id, timeout=timeout, **visibility_metadata)
        # The processor uses this exact suffix in both image annotations and targets.
        suffix = f"{name_suffix or ''}_t{tick}"
        recognition = await self.backend.process(frame, labels=labels, name_suffix=suffix)
        targets = {name: tuple(float(value) for value in coordinate)
                   for name, coordinate in recognition.detected_entities.items()}
        if any(not name.endswith(suffix) for name in targets):
            raise RuntimeError("Visual processor returned targets without the frame reference")
        objects = getattr(recognition, "objects", ())
        if objects:
            entities = tuple(copy.deepcopy(item) for item in objects)
            for entity in entities:
                ref = entity["target_ref"]
                if not ref.endswith(suffix):
                    raise RuntimeError("Visual processor returned objects without the frame reference")
                if ref in targets:
                    entity["location"] = list(targets[ref])
        else:
            entities = tuple({"entity_name": name, "target_ref": name,
                              "location": list(targets[name])} for name in sorted(targets))
        return ObservationFrame(
            tick=tick, agent_position=copy.deepcopy(agent_position), entities=entities,
            captured_at_monotonic=captured_at, view=view, batch_id=batch_id,
            action_id=action_id, capture_kind=capture_kind, rgb_path=rgb_path,
            depth_path=depth_path, masked_rgb_path=recognition.masked_rgb_path,
            targets=targets, vision_backend=backend_name or "grounded_sam", frame_ref=frame_ref,
            camera_metadata=camera_metadata if supports_image_targets(backend_name) else None,
        )
