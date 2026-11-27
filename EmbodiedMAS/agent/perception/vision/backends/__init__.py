"""Select an image processor before touching model or oracle dependencies."""
from __future__ import annotations

from EmbodiedMAS.agent.visual_backends import uses_detections


def create_backend(name: str):
    normalized = str(name).strip().lower()
    if normalized == "raw_vlm":
        from .raw_vlm import RawVLMBackend
        return RawVLMBackend()
    if not uses_detections(normalized):
        raise ValueError(f"Unknown visual backend: {name!r}")
    from .grounded_sam import GroundedSamBackend
    return GroundedSamBackend(name=normalized)
