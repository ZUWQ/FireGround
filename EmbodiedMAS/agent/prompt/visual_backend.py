"""Backend-specific prompt overlays; source action configs remain unchanged."""
from __future__ import annotations

import copy
from typing import Any

from EmbodiedMAS.agent.visual_backends import VISUAL_BACKENDS


def resolve_visual_backend(modality: str, backend: str | None = None) -> str:
    selected = str(backend or "grounded_sam").strip().lower()
    if modality in {"vision", "vision_observation"} and selected not in VISUAL_BACKENDS:
        raise ValueError(f"Unknown visual backend: {backend!r}")
    return selected


def action_space_for_backend(action_space: dict[str, Any], *, raw: bool = False, hybrid: bool = False) -> dict[str, Any]:
    if not (raw or hybrid):
        return action_space
    result = copy.deepcopy(action_space)
    image_target = '{"frame_ref":"<actionable frame_ref>","pixel_xy":[u,v]}'
    for action in result.get("allowed_actions", []):
        if action.get("name") == "navigate":
            action["description"] = "Choose a known room anchor, a relative offset, or an image ground point; the executor plans the path."
            if hybrid:
                action["description"] = "Choose a current grounded entity, a known room anchor, a relative offset, or an image ground point; the executor plans the path."
            for subtype in action.get("subtypes", []):
                if raw and subtype.get("name") == "entity":
                    subtype.update(description="Navigate to a known room_name anchor.", parameters={"target": "room_name"})
            action["subtypes"].append({"name": "image", "description": "Approach a visible target by selecting a nearby standable ground point in an actionable frame.", "parameters": {"target": image_target}})
        elif action.get("name") == "suppress_fire":
            action.update(description="Aim and spray from the current position at a visible burning solid surface selected in an actionable frame. This does not navigate. Approach when needed and check subsequent feedback and images.", parameters={"target": image_target})
            if hybrid:
                action["description"] = "Aim and spray from the current position at a current grounded target_ref or a visible burning solid surface selected by image pixels. This does not navigate. Approach when needed and check subsequent feedback and images."
                action["parameters"]["target"] = "current grounded target_ref OR " + image_target
    return result
