"""The existing DINO/SAM/depth pipeline is one replaceable visual processor."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
import numpy as np

from ..frame import VisualFrame
from EmbodiedMAS.agent.perception.vision_pipeline.vision_recognition import (
    recognize_labels_in_agent_view,
)


# All backend instances share SAM's stateful predictor, including set_image().
_PREDICTOR_LOCK = threading.Lock()


def _recognize_serialized(**kwargs):
    with _PREDICTOR_LOCK:
        return recognize_labels_in_agent_view(**kwargs)


class GroundedSamBackend:
    name = "grounded_sam"
    requires_oracle_labels = False

    def __init__(self, *, name: str = "grounded_sam") -> None:
        if name not in {"grounded_sam", "grounded_sam_pixel"}:
            raise ValueError(f"Not a SAM backend: {name!r}")
        self.name = name

    async def process(
        self, frame: VisualFrame, *, labels: list[str], name_suffix: str | None = None
    ):
        worker = asyncio.create_task(asyncio.to_thread(
            _recognize_serialized,
            rgb_path=frame.rgb_path,
            depth_path=frame.depth_path,
            labels=labels,
            agent_world_pos=np.array(frame.world_position, dtype=np.float32),
            agent_yaw_deg=frame.yaw_degrees,
            mask_output_dir=Path(frame.rgb_path).parent,
            entity_id_suffix=name_suffix,
            camera_metadata=frame.camera_metadata,
        ))
        # A completion future also owns failures during cancellation; shielding
        # a bare failing task logs those failures on Python 3.14.
        completion = asyncio.gather(worker, return_exceptions=True)
        cancelled = None
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as exc:
                # A running thread cannot be cancelled. Keep ownership through
                # repeated cancellation until it releases the shared predictor.
                cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled
        return worker.result()
