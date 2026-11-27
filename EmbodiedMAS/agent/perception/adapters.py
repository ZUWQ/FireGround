"""Pure presentation of an explicitly owned observation package."""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from numbers import Real
from typing import Any

from EmbodiedMAS.agent.config.environment_config import load_environment_config
from EmbodiedMAS.environment.safety import npc_is_safe
from EmbodiedMAS.agent.perception.records import DecisionObservation
from EmbodiedMAS.agent.visual_backends import image_target_metadata
from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.agent.perception.visual_distance import project_visual_entity
from EmbodiedMAS.agent.spatial_units import public_pose_m, world_xy_m


def _fire_state(entity: dict[str, Any]) -> bool | None:
    """Unknown fire state remains distinct from a confirmed false state."""
    value = entity.get("burning_state")
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    return None


def _remaining_percent(entity: dict[str, Any]) -> float | None:
    value = entity.get("burned_percentage")
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if not math.isfinite(value) or not 0 <= value <= 100:
        return None
    return 100 - value


def _entity_xy(entity: dict[str, Any]) -> list[float] | None:
    """Expose observed world XY in meters; the internal position remains in UU."""
    return world_xy_m(entity.get("location"))


def _is_npc(entity: dict[str, Any]) -> bool:
    return entity.get("entity_type") == "npc" or entity.get("kind") == "npc_info"


def _text_entities_by_frame(observation: DecisionObservation, safe_areas: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Keep first sightings, fire-state changes and NPC safety changes within an action.

    Position, room, remaining material, and NPC health are carried by retained records,
    but changes to these fields alone do not emit another record.
    """
    last_state_by_id: dict[str, tuple[Any, ...]] = {}
    result: list[list[dict[str, Any]]] = []
    # The ending four views continue the same action-local state as route frames.
    for frame in observation.frames:
        entities = []
        for entity in frame.entities:
            identity = stable_entity_id(entity)
            if identity is not None:
                state = ((_fire_state(entity), npc_is_safe(entity.get("location"), safe_areas))
                         if _is_npc(entity) else (_fire_state(entity),))
                if identity in last_state_by_id and last_state_by_id[identity] == state:
                    continue
                last_state_by_id[identity] = state
            # Selection is read-only; copy only the public projection below.
            entities.append(entity)
        result.append(entities)
    return result


def _text_rooms(entities: list[dict[str, Any]], safe_areas: list[dict[str, Any]]) -> dict[str, Any]:
    """Project normalized identities into the compact model-visible schema."""
    rooms: dict[str, Any] = {}
    for entity in entities:
        name = entity.get("name") or entity.get("entity_name") or "unknown"
        room = entity.get("room") or "unknown"
        members = rooms.setdefault(room, {"objects": {}, "npcs": {}})
        if _is_npc(entity):
            members["npcs"][name] = {
                "location": _entity_xy(entity),
                "health": copy.deepcopy(entity.get("health")),
                "fire_affected": _fire_state(entity),
                "is_safe": npc_is_safe(entity.get("location"), safe_areas),
            }
        else:
            members["objects"][name] = {
                "location": _entity_xy(entity),
                "burning": _fire_state(entity),
                "remaining_percent": _remaining_percent(entity),
            }
    return rooms


class ObservationAdapter:
    def __init__(self, modality: str, *, observation_scope: str = "local", record_mode: str | None = None):
        if modality not in {"text", "vision"}:
            raise ValueError(f"Unsupported modality: {modality!r}")
        if observation_scope not in {"local", "global"}:
            raise ValueError("observation_scope must be local or global")
        if modality == "vision" and observation_scope == "global":
            raise NotImplementedError("global vision requires a defined global visual observation provider")
        self._safe_areas = copy.deepcopy(load_environment_config()["SafeArea"]) if modality == "text" else []
        self.modality = modality
        self.observation_scope = observation_scope
        self.record_mode = record_mode or ("full_snapshot" if observation_scope == "global" or modality == "vision" else "action_first_seen_or_fire_state_change")
        if self.record_mode not in {"full_snapshot", "action_first_seen_or_fire_state_change"}:
            raise ValueError("Unsupported record_mode")
        if modality == "vision" and self.record_mode != "full_snapshot":
            raise ValueError("Vision observations require full_snapshot records")
        if observation_scope == "global" and self.record_mode != "full_snapshot":
            raise ValueError("Global oracle requires full_snapshot records")

    def prepare_observation(self, observation: DecisionObservation) -> dict[str, Any]:
        """Project one complete batch without decision-specific metadata."""
        if self.modality == "vision" and any(f.rgb_path is None for f in observation.frames):
            raise ValueError("Every visual observation must have its own current image")
        if observation.scope != self.observation_scope:
            raise ValueError("Observation scope does not match the configured adapter")
        # Also enforce boundary-only input for reconstructed/offline packages.
        if self.observation_scope == "global" and observation.during_action:
            observation = replace(observation, during_action=())
        text_entities = None
        if self.modality == "text":
            text_entities = (_text_entities_by_frame(observation, self._safe_areas) if self.record_mode != "full_snapshot"
                             else [list(frame.entities) for frame in observation.frames])

        def present(index, frame):
            result = {"tick": frame.tick, "view": frame.view,
                      "captured_at": frame.captured_at_monotonic,
                      "agent_position": public_pose_m(frame.agent_position)}
            if self.modality == "vision" and frame.vision_backend != "raw_vlm":
                # World coordinates belong to execution, not the visual prompt.
                result["entities"] = [
                    project_visual_entity(entity, observer_uu=frame.agent_position,
                        include_distance=frame.vision_backend == "grounded_sam_pixel")
                    for entity in frame.entities
                ]
            elif self.modality == "text":
                result["rooms"] = _text_rooms(text_entities[index], self._safe_areas)
            if self.modality == "vision":
                result.update(image_target_metadata(frame, actionable=
                    frame in observation.surroundings and frame.capture_kind == "surround"))
            if self.observation_scope == "global" and frame.global_metadata is not None:
                result["destroyed_entities"] = copy.deepcopy(frame.global_metadata.get("destroyed_entities", []))
                result["sampling"] = {key: copy.deepcopy(value) for key, value in frame.global_metadata.items()
                                      if key in {"source", "requested_at_monotonic", "completed_at_monotonic"}}
            return result

        presented = [present(index, frame) for index, frame in enumerate(observation.frames)]
        during_count = len(observation.during_action)
        payload = {
            "batch_id": observation.batch_id, "action_id": observation.action_id,
            "observation_scope": observation.scope,
            "reason": observation.reason,
            "surroundings": presented[during_count:],
            "during_action": presented[:during_count],
            "latest_tick": observation.surroundings[-1].tick,
            "raw_visual_reference": ([f.image_path for f in observation.frames]
                                     if self.modality == "vision" else None),
        }
        if self.modality == "text":
            payload["entity_record_mode"] = self.record_mode
        else:
            payload["visual_images"] = [
                {"path": path, "image_role": role, "tick": frame.tick, "view": frame.view,
                 "captured_at": frame.captured_at_monotonic,
                 **image_target_metadata(frame, actionable=role == "raw" and
                     frame in observation.surroundings and frame.capture_kind == "surround"),
                 **({"annotation_available": len(frame.image_paths) > 1}
                    if frame.vision_backend != "raw_vlm" else {})}
                for frame in observation.frames
                for role, path in zip(("raw", "annotated"), frame.image_paths)
            ]
        return payload

    def build_observation(self, observation: DecisionObservation, *, trigger: str,
                          context: dict[str, Any] | None = None):
        """Present an independent batch outside the runtime's prepared lifecycle."""
        payload = self.prepare_observation(observation)
        payload["decision_trigger"] = trigger
        if context:
            payload["trigger_context"] = copy.deepcopy(context)
        return observation.agent_position, payload


def create_observation_adapter(modality: str, *, observation_scope: str = "local",
                               record_mode: str | None = None) -> ObservationAdapter:
    return ObservationAdapter(modality, observation_scope=observation_scope, record_mode=record_mode)
