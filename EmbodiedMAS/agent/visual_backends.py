"""Visual capabilities shared without importing detector dependencies."""
from __future__ import annotations

VISUAL_BACKENDS = ("grounded_sam", "raw_vlm", "grounded_sam_pixel")


def uses_detections(backend: str | None) -> bool:
    return backend in {"grounded_sam", "grounded_sam_pixel"}


def supports_image_targets(backend: str | None) -> bool:
    return backend in {"raw_vlm", "grounded_sam_pixel"}


def image_target_metadata(frame, *, actionable: bool) -> dict:
    """Public pixel contract; calibration and depth remain execution-private."""
    if not supports_image_targets(frame.vision_backend):
        return {}
    metadata = frame.camera_metadata or {}
    return {"frame_ref": frame.frame_ref, "width": metadata.get("width"),
            "height": metadata.get("height"), "actionable": bool(actionable)}
