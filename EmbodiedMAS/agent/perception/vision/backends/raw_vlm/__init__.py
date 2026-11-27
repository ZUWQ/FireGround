"""Raw RGB backend: no detector, segmenter, vocabulary or learned inference."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawVisionResult:
    masked_rgb_path: None = None
    detected_entities: dict = field(default_factory=dict)
    objects: tuple = ()


class RawVLMBackend:
    name = "raw_vlm"
    requires_oracle_labels = False

    async def process(self, frame, *, labels=(), name_suffix=None):
        return RawVisionResult()
