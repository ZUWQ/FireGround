"""Small image-target protocol shared without importing visual model dependencies."""
from __future__ import annotations


class ImageTargetError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.public_message = message
        super().__init__(message)


def validate_image_target(value):
    if not isinstance(value, dict) or set(value) != {"frame_ref", "pixel_xy"}:
        raise ImageTargetError("invalid_image_target", "Image target requires only frame_ref and pixel_xy.")
    reference, point = value["frame_ref"], value["pixel_xy"]
    if not isinstance(reference, str) or not reference.strip() or reference != reference.strip():
        raise ImageTargetError("invalid_frame_ref", "Copy a non-empty frame_ref exactly from the observation.")
    if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int for v in point):
        raise ImageTargetError("invalid_pixel", "pixel_xy must contain two original-image integer coordinates.")
    if any(v < 0 for v in point):
        raise ImageTargetError("pixel_out_of_bounds", "Pixel coordinates must be within the referenced image.")
    return {"frame_ref": reference, "pixel_xy": list(point)}
