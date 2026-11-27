"""Visual capture API, independent of the selected image processor."""
from .frame import VisualFrame
from .observation import VisionPerceptionModule

__all__ = ["VisualFrame", "VisionPerceptionModule"]
