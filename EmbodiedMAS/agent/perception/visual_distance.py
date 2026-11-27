"""Model-visible visual entity projection and approximate planar distances.

Full target geometry stays private. SAM+Pixel may expose the distance from its
capture observer to the already-grounded representative point, never raw depth.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from EmbodiedMAS.agent.spatial_units import finite_xy, world_xy_m

_VISUAL_ENTITY_FIELDS = frozenset({
    "entity_name", "target_ref", "label", "confidence", "bbox_xyxy",
    "grounding_available", "category", "attached_to", "grounding_source",
})


def planar_distance_m(target_m: Any, observer_m: Any) -> float | None:
    """Approximate horizontal distance rounded to 0.1 m; inputs are meters."""
    target = finite_xy(target_m)
    observer = finite_xy(observer_m)
    if target is None or observer is None:
        return None
    distance = math.hypot(target[0] - observer[0], target[1] - observer[1])
    return round(distance, 1) if math.isfinite(distance) else None


def distance_xy_m(target_uu: Any, observer_uu: Any) -> float | None:
    """Convert two internal UE positions once and estimate horizontal meters."""
    return planar_distance_m(world_xy_m(target_uu), world_xy_m(observer_uu))


def project_visual_entity(entity: dict[str, Any], *, observer_uu: Any = None,
                          include_distance: bool = False) -> dict[str, Any]:
    """Detach the public detection schema, optionally adding capture distance.

    The observer must be from the same captured frame as entity.location.
    Grounding availability is required even if a private location is present.
    A missing or invalid position remains unknown, not zero or a guessed depth.
    """
    public = {key: copy.deepcopy(value) for key, value in entity.items()
              if key in _VISUAL_ENTITY_FIELDS}
    if include_distance:
        public["distance_xy_m"] = (
            distance_xy_m(entity.get("location"), observer_uu)
            if entity.get("grounding_available") is True else None)
    return public
