"""Pure action validation and routing shared by runtime acceptance and execution."""
from __future__ import annotations

from typing import Any

from .geometry import direction_to_angle, validate_relative_distance
from .results import normalize_action_result
from .image_targets import validate_image_target


class ActionExecutor:
    def __init__(self, robot: Any):
        self.robot = robot

    def validate(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized copy, without invoking actions or mutating state."""
        if not isinstance(action_payload, dict):
            raise ValueError("Action payload must be an object")
        name = action_payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Action requires a non-empty name")
        action_name = name.strip()
        parameters = action_payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Action parameters must be an object")
        parameters = dict(parameters)
        subtype = str(action_payload.get("subtype") or "").strip().lower()

        def image_target() -> None:
            if not getattr(self.robot, "supports_image_targets", False):
                raise ValueError("Image targets require the raw_vlm or grounded_sam_pixel backend")
            parameters["target"] = validate_image_target(parameters.get("target"))

        def target() -> None:
            value = parameters.get("target")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{action_name} requires parameters.target (entity reference)")
            parameters["target"] = value.strip()

        if action_name == "navigate":
            if subtype == "entity":
                target()
            elif subtype == "image":
                image_target()
            elif subtype == "relative":
                if "distance" not in parameters or "direction" not in parameters:
                    raise ValueError("navigate/relative requires parameters.distance and parameters.direction")
                distance = validate_relative_distance(parameters["distance"])
                direction = parameters["direction"]
                if not isinstance(direction, str):
                    raise ValueError("navigate/relative requires a direction string")
                direction = direction.strip().lower()
                direction_to_angle(direction)
                parameters.update(distance=distance, direction=direction)
            else:
                raise ValueError(f"Unknown navigate subtype {subtype!r}. Expected 'entity' or 'relative'.")
        elif action_name == "instruct_civilian":
            if subtype not in {"follow", "stop_following"}:
                raise ValueError("instruct_civilian requires subtype 'follow' or 'stop_following'")
        elif action_name == "wait":
            if subtype or parameters:
                raise ValueError("wait takes no subtype or parameters; its duration is fixed at 10 seconds")
        elif action_name == "explore":
            if subtype or set(parameters) != {"room"}:
                raise ValueError("explore requires only parameters.room and no subtype")
            room = parameters["room"]
            if not isinstance(room, str) or not room.strip():
                raise ValueError("explore requires a non-empty room name")
            parameters["room"] = room.strip()
            validate_room = getattr(self.robot, "validate_exploration_room", None)
            if callable(validate_room):
                validate_room(parameters["room"])
        elif action_name == "suppress_fire":
            if not getattr(self.robot, "supports_fire", False):
                raise NotImplementedError(f"Action {action_name!r} is not supported by {type(self.robot).__name__}")
            if isinstance(parameters.get("target"), dict):
                image_target()
            else:
                target()
        else:
            raise NotImplementedError(f"Action {action_name!r} is not implemented")

        normalized = dict(action_payload, name=action_name, parameters=parameters)
        if "subtype" in normalized or subtype:
            normalized["subtype"] = subtype
        return normalized

    async def execute(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        payload = self.validate(action_payload)
        name, parameters = payload["name"], payload["parameters"]
        if name == "navigate":
            if payload["subtype"] == "entity":
                raw = await self.robot.navigate_entity(parameters["target"])
            elif payload["subtype"] == "image":
                raw = await self.robot.navigate_image(parameters["target"])
            else:
                raw = await self.robot.navigate_relative(parameters["distance"], parameters["direction"])
        elif name == "instruct_civilian":
            raw = await self.robot.instruct_civilian(payload["subtype"])
        elif name == "wait":
            raw = await self.robot.wait()
        elif name == "explore":
            raw = await self.robot.explore_room(parameters["room"])
        else:
            raw = await self.robot.suppress_fire(parameters["target"])
        return normalize_action_result(name, raw)
