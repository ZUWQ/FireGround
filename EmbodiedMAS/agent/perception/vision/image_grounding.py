"""Decision-scoped RGB point to paired-depth grounding, without semantic models.

A valid depth is geometric evidence only. It does not certify that a chosen point
is ground, a person, a burning surface, or reachable by the navigation controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from EmbodiedMAS.agent.actions.image_targets import ImageTargetError
from EmbodiedMAS.agent.perception.records import DecisionObservation
from EmbodiedMAS.agent.visual_backends import supports_image_targets
from EmbodiedMAS.agent.perception.vision_pipeline.Location.coordinate_projection import (
    load_depth, pixel_to_world_unreal, read_rgb_hw,
)


class ImageGroundingError(ImageTargetError):
    """Expected image-target failure; public reason contains no world geometry."""


@dataclass(frozen=True)
class ImageGroundingConfig:
    window_size: int = 5
    min_valid_fraction: float = 0.5
    max_depth_uu: float = 10000.0
    max_relative_iqr: float = 0.10
    min_iqr_tolerance_uu: float = 20.0

    def __post_init__(self):
        if (isinstance(self.window_size, bool) or not isinstance(self.window_size, int)
                or self.window_size < 1 or self.window_size % 2 != 1):
            raise ValueError("window_size must be a positive odd integer")
        for key in ("min_valid_fraction", "max_depth_uu", "max_relative_iqr", "min_iqr_tolerance_uu"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.min_valid_fraction > 1:
            raise ValueError("min_valid_fraction must be at most one")


@dataclass(frozen=True)
class ImageGroundingResult:
    frame_ref: str
    pixel_xy: tuple[int, int]
    purpose: str
    world_point: tuple[float, float, float]
    execution_point: tuple[float, float, float]
    depth_median_uu: float
    depth_iqr_uu: float
    valid_count: int
    sample_count: int
    quality_config: dict[str, Any]

    def audit(self) -> dict[str, Any]:
        return asdict(self)


def validate_camera_metadata(metadata: Any) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Require actual snapshot calibration; never synthesize it from robot pose."""
    try:
        if not isinstance(metadata, dict):
            raise ValueError
        if isinstance(metadata.get("depth_mode"), bool) or metadata.get("depth_mode") not in (1, 3):
            raise ImageGroundingError("unsupported_depth_mode", "Image targets require linear or view-space Z depth.")
        width, height = metadata["width"], metadata["height"]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height)):
            raise ValueError
        intrinsics = metadata["intrinsics"]
        fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            raise ValueError
        position = np.asarray(metadata["world_position"], dtype=np.float64).reshape(3)
        rotation = np.asarray(metadata["world_rotation"], dtype=np.float64).reshape(3)
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return width, height, matrix, position, rotation
    except ImageGroundingError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ImageGroundingError("invalid_camera_metadata", "The selected image has no valid snapshot calibration.") from None


def ground_image_target(
    observation: DecisionObservation, target: dict[str, Any], *, purpose: str,
    navigation_z: float = 1000.0, config: ImageGroundingConfig | None = None,
) -> ImageGroundingResult:
    """Resolve a pixel-capable original RGB in the current ending observation.

    Depth quality uses valid sample fraction and interquartile range. The pixel
    itself is never shifted, replaced by a neighbour or clamped into the image.
    Navigation uses the existing scene's actor-root height convention; suppression
    retains the sampled surface height. Neither operation infers object identity.
    """
    if purpose not in {"navigate", "suppress_fire"}:
        raise ValueError("purpose must be navigate or suppress_fire")
    if not isinstance(target, dict) or set(target) != {"frame_ref", "pixel_xy"}:
        raise ImageGroundingError("invalid_image_target", "Image target requires frame_ref and pixel_xy.")
    reference, pixel = target["frame_ref"], target["pixel_xy"]
    if (not isinstance(reference, str) or not reference or reference != reference.strip()
            or not isinstance(pixel, (tuple, list)) or len(pixel) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in pixel)):
        raise ImageGroundingError("invalid_image_target", "Use an exact frame_ref and two integer original-image pixels.")
    frames = [frame for frame in observation.surroundings if frame.frame_ref == reference]
    if len(frames) != 1:
        raise ImageGroundingError("frame_not_actionable", "Choose an actionable image from the current ending observation.")
    frame = frames[0]
    if (not supports_image_targets(frame.vision_backend) or frame.capture_kind != "surround"
            or frame.batch_id != observation.batch_id):
        raise ImageGroundingError("frame_not_actionable", "Choose an actionable original RGB image from the current ending observation.")
    width, height, matrix, position, rotation = validate_camera_metadata(frame.camera_metadata)
    u, v = pixel
    if not 0 <= u < width or not 0 <= v < height:
        raise ImageGroundingError("pixel_out_of_bounds", "The selected pixel is outside the original image.")
    if not frame.rgb_path or not frame.depth_path:
        raise ImageGroundingError("missing_depth", "The selected image has no paired depth.")
    try:
        actual_height, actual_width = read_rgb_hw(frame.rgb_path)
        if (actual_width, actual_height) != (width, height):
            raise ImageGroundingError("image_size_mismatch", "Image dimensions do not match the snapshot calibration.")
        if not Path(frame.depth_path).is_file():
            raise ImageGroundingError("missing_depth", "The selected image has no paired depth.")
        depth = load_depth(frame.rgb_path, frame.depth_path)
    except ImageGroundingError:
        raise
    except (ValueError, OSError, RuntimeError):
        raise ImageGroundingError("invalid_depth", "The paired depth cannot be read at the original image dimensions.") from None
    quality = config or ImageGroundingConfig()
    radius = quality.window_size // 2
    patch = depth[max(0, v-radius):min(height, v+radius+1), max(0, u-radius):min(width, u+radius+1)]
    valid = patch[np.isfinite(patch) & (patch > 0) & (patch < quality.max_depth_uu)]
    if valid.size == 0 or valid.size / patch.size < quality.min_valid_fraction:
        raise ImageGroundingError("invalid_depth", "Too few valid depth samples near the selected pixel.")
    median = float(np.median(valid))
    q25, q75 = np.percentile(valid, [25, 75])
    iqr = float(q75 - q25)
    if iqr > max(quality.min_iqr_tolerance_uu, quality.max_relative_iqr * median):
        raise ImageGroundingError("unstable_depth", "Depth is discontinuous near the selected pixel; observe again or select another point.")
    world = tuple(float(value) for value in pixel_to_world_unreal(u, v, median, matrix, position, rotation))
    if not np.isfinite(world).all():
        raise ImageGroundingError("invalid_depth", "The selected pixel did not produce a finite world point.")
    if purpose == "navigate":
        if isinstance(navigation_z, bool) or not math.isfinite(navigation_z):
            raise ValueError("navigation_z must be finite")
        execution = (world[0], world[1], float(navigation_z))
    else:
        execution = world
    return ImageGroundingResult(reference, (u, v), purpose, world, execution, median,
        iqr, int(valid.size), int(patch.size), asdict(quality))
