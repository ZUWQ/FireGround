"""Strict decision protocol with two deterministic, content-preserving repairs.

This module never consults observations, fills missing decisions, or calls a model.
The audit is returned even for valid output and attached to expected failures.
"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from .actions.image_targets import validate_image_target, ImageTargetError
from .memory import DEFAULT_NOTE_MAX_CHARS

PROTOCOL_VERSION = "decision-output-v2"
_ACTION_SUBTYPES = {
    "navigate": frozenset({"entity", "relative", "image"}),
    "instruct_civilian": frozenset({"follow", "stop_following"}),
}
_DIRECTIONS = frozenset({"forward", "backward", "left", "right", "front_left", "front_right"})
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z", re.IGNORECASE)


class InvalidDecisionOutput(ValueError):
    """Expected model-output failure, with a serializable audit and safe errors."""

    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        self.errors = audit["errors"]
        super().__init__("; ".join(error["message"] for error in self.errors))


class _DecodeFailure(ValueError):
    def __init__(self, code: str, message: str):
        self.error = {"code": code, "path": "$", "message": message}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DecodeFailure("duplicate_key", "JSON object keys must be unique.")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _DecodeFailure("non_finite_number", "All JSON numbers must be finite.")
    return value


def _reject_constant(_text: str) -> None:
    raise _DecodeFailure("non_finite_number", "All JSON numbers must be finite.")


def _decode(text: str) -> Any:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise _DecodeFailure("invalid_unicode", "All JSON strings and keys must contain valid Unicode.") from None
    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_float=_finite_float,
                          parse_constant=_reject_constant)
    except _DecodeFailure:
        raise
    except (ValueError, RecursionError):
        raise _DecodeFailure("invalid_json", "Return exactly one complete JSON object.") from None


def _contains_invalid_unicode(payload: Any) -> bool:
    """Inspect keys and values iteratively, including arbitrary memory metadata."""
    pending = [payload]
    visited = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                return True
        elif isinstance(value, (dict, list)) and id(value) not in visited:
            visited.add(id(value))
            if isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            else:
                pending.extend(value)
    return False


def validate_payload(payload: Any, *, note_max_chars: int = DEFAULT_NOTE_MAX_CHARS) -> list[dict[str, str]]:
    """Check protocol shape without modifying values or judging action strategy.

    Role capabilities, coordination mode, target resolution and world-state
    preconditions belong to the runtime. Communication routing metadata is kept.
    """
    if isinstance(note_max_chars, bool) or not isinstance(note_max_chars, int) or note_max_chars <= 0:
        raise ValueError("note_max_chars must be a positive integer")
    if _contains_invalid_unicode(payload):
        return [{"code": "invalid_unicode", "path": "$",
                 "message": "All JSON strings and keys must contain valid Unicode."}]
    errors: list[dict[str, str]] = []

    def error(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    def shape(value: Any, required: set[str], path: str) -> bool:
        if not isinstance(value, dict):
            error("expected_object", path, f"{path} must be an object.")
            return False
        for key in sorted(required - value.keys()):
            error("missing_field", f"{path}.{key}", f"{path}.{key} is required.")
        if value.keys() - required:
            error("unexpected_field", path, f"{path} contains unsupported fields.")
        return True

    def nonempty_string(value: Any, path: str) -> None:
        if not isinstance(value, str) or not value.strip():
            error("expected_nonempty_string", path, f"{path} must be a non-empty string.")

    if not shape(payload, {"decision_basis", "action", "communication", "memory_update"}, "$"):
        return errors
    if "decision_basis" in payload:
        nonempty_string(payload["decision_basis"], "$.decision_basis")

    action = payload.get("action")
    if not isinstance(action, dict):
        if "action" in payload:
            error("expected_object", "$.action", "$.action must be an object.")
    else:
        name = action.get("name")
        if not isinstance(name, str) or name not in {*_ACTION_SUBTYPES, "suppress_fire", "explore", "wait"}:
            error("invalid_action", "$.action.name",
                  "Action name must be navigate, suppress_fire, instruct_civilian, wait or explore.")
            shape(action, {"name", "parameters"} | ({"subtype"} if "subtype" in action else set()), "$.action")
        else:
            shape(action, {"name", "parameters"} | ({"subtype"} if name in _ACTION_SUBTYPES else set()), "$.action")
            subtype = action.get("subtype")
            valid_subtype = isinstance(subtype, str) and subtype in _ACTION_SUBTYPES.get(name, ())
            if name in _ACTION_SUBTYPES and not valid_subtype:
                allowed = ", ".join(sorted(_ACTION_SUBTYPES[name]))
                error("invalid_subtype", "$.action.subtype", f"$.action.subtype must be one of: {allowed}.")
            parameters = action.get("parameters")
            expected = None
            if name == "navigate" and valid_subtype:
                expected = {"target"} if subtype in {"entity", "image"} else {"distance", "direction"}
            elif name == "suppress_fire":
                expected = {"target"}
            elif name == "explore":
                expected = {"room"}
            elif name in {"instruct_civilian", "wait"}:
                expected = set()
            if expected is not None and shape(parameters, expected, "$.action.parameters"):
                for key in expected & {"target", "room"}:
                    if key in parameters:
                        reference = parameters[key]
                        image_target = key == "target" and (subtype == "image" or
                            (name == "suppress_fire" and isinstance(reference, dict)))
                        if image_target:
                            try:
                                validate_image_target(reference)
                            except ImageTargetError as exc:
                                error(exc.code, f"$.action.parameters.{key}", exc.public_message)
                            continue
                        nonempty_string(reference, f"$.action.parameters.{key}")
                        if isinstance(reference, str) and reference != reference.strip():
                            error("reference_whitespace", f"$.action.parameters.{key}",
                                  f"$.action.parameters.{key} must copy the reference exactly, without surrounding whitespace.")
                if "distance" in expected and "distance" in parameters:
                    distance = parameters["distance"]
                    try:
                        finite = (not isinstance(distance, bool) and isinstance(distance, (int, float))
                                  and math.isfinite(distance) and distance > 0 and math.isfinite(distance * 100.0))
                    except OverflowError:
                        finite = False
                    if not finite:
                        error("invalid_distance", "$.action.parameters.distance",
                              "$.action.parameters.distance must be a positive finite number in meters.")
                if "direction" in expected and "direction" in parameters:
                    direction = parameters["direction"]
                    if not isinstance(direction, str) or direction not in _DIRECTIONS:
                        error("invalid_direction", "$.action.parameters.direction",
                              "Direction must be forward, backward, left, right, front_left or front_right.")
            elif expected is None and not isinstance(parameters, dict):
                error("expected_object", "$.action.parameters", "$.action.parameters must be an object.")

    if "communication" in payload and payload["communication"] is not None:
        communication = payload["communication"]
        if not isinstance(communication, dict):
            error("invalid_communication", "$.communication", "$.communication must be null or an object.")
        else:
            for key, allowed in (
                ("type", {"discovery", "request_help", "status_update"}),
                ("priority", {"urgent", "important", "useful"}),
            ):
                value = communication.get(key)
                if not isinstance(value, str) or value not in allowed:
                    error("invalid_message_" + key, "$.communication." + key,
                          f"$.communication.{key} must be one of: {', '.join(sorted(allowed))}.")
            if not isinstance(communication.get("message"), str):
                error("invalid_message_text", "$.communication.message", "$.communication.message must be a string.")

    if "memory_update" in payload:
        update = payload["memory_update"]
        if shape(update, {"note"}, "$.memory_update") and "note" in update:
            note = update["note"]
            if note is not None and not isinstance(note, str):
                error("invalid_memory_note", "$.memory_update.note", "$.memory_update.note must be a string or null.")
            elif isinstance(note, str) and len(note) > note_max_chars:
                error("memory_note_too_long", "$.memory_update.note",
                      f"$.memory_update.note must contain at most {note_max_chars} characters.")
    return errors


def process_response(raw: str | None, *, note_max_chars: int = DEFAULT_NOTE_MAX_CHARS) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate, apply only the whitelist, and validate again; never infer content.

    InvalidDecisionOutput is the only expected failure. Its audit preserves the
    original response, attempted repairs and final candidate, including failures.
    """
    audit: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "raw_response": raw,
                             "status": "invalid", "original_errors": [], "repairs": [],
                             "normalized_response": None, "errors": []}
    if not isinstance(raw, str) or not raw.strip():
        audit["errors"] = [{"code": "empty_response", "path": "$", "message": "Return a non-empty JSON object."}]
        audit["original_errors"] = copy.deepcopy(audit["errors"])
        raise InvalidDecisionOutput(audit)

    try:
        payload = _decode(raw)
    except _DecodeFailure as exc:
        audit["original_errors"].append(exc.error)
        fence = _FENCE.fullmatch(raw.strip())
        if fence is None:
            audit["errors"] = copy.deepcopy(audit["original_errors"])
            raise InvalidDecisionOutput(audit) from None
        candidate_text = fence.group("body")
        audit["repairs"].append({"rule_id": "remove_complete_json_fence", "changes": [
            {"path": "$", "before": raw, "after": candidate_text}]})
        try:
            payload = _decode(candidate_text)
        except _DecodeFailure as inner:
            audit["errors"] = [inner.error]
            raise InvalidDecisionOutput(audit) from None

    original_errors = validate_payload(payload, note_max_chars=note_max_chars)
    audit["original_errors"].extend(copy.deepcopy(original_errors))
    try:
        candidate = copy.deepcopy(payload)
    except RecursionError:
        error = {"code": "structure_too_deep", "path": "$",
                 "message": "The JSON structure is too deeply nested to process."}
        audit["original_errors"].append(copy.deepcopy(error))
        audit["errors"] = [error]
        raise InvalidDecisionOutput(audit) from None
    if isinstance(candidate, dict):
        action = candidate.get("action")
        if isinstance(action, dict) and "subtype" not in action:
            name, parameters = action.get("name"), action.get("parameters")
            if isinstance(name, str) and name in _ACTION_SUBTYPES and isinstance(parameters, dict):
                subtype = parameters.get("subtype")
                if isinstance(subtype, str) and subtype in _ACTION_SUBTYPES[name]:
                    before = copy.deepcopy(action)
                    action["subtype"] = parameters.pop("subtype")
                    audit["repairs"].append({"rule_id": "move_action_parameter_subtype", "changes": [
                        {"from": "$.action.parameters.subtype", "to": "$.action.subtype", "value": subtype}],
                        "before": before, "after": copy.deepcopy(action)})

    audit["normalized_response"] = copy.deepcopy(candidate)
    audit["errors"] = validate_payload(candidate, note_max_chars=note_max_chars)
    if audit["errors"]:
        raise InvalidDecisionOutput(audit)
    audit["status"] = "repaired" if audit["repairs"] else "valid"
    return candidate, audit


__all__ = ["PROTOCOL_VERSION", "InvalidDecisionOutput", "process_response", "validate_payload"]
