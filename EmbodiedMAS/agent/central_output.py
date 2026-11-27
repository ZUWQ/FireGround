"""CMAS-only team protocol, reusing individual action validation and repairs."""
from __future__ import annotations

import copy
import json
from typing import Any, Iterable

from EmbodiedMAS.agent.decision_output import (
    InvalidDecisionOutput, _DecodeFailure, _FENCE, _contains_invalid_unicode,
    _decode, process_response as process_action_response, validate_payload as _validate_action_payload,
)

from EmbodiedMAS.agent.memory import DEFAULT_NOTE_MAX_CHARS

PROTOCOL_VERSION = "central-output-v2"


def _error(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _path_errors(errors: list[dict[str, str]], robot_id: str) -> list[dict[str, str]]:
    prefix = "$.robot_actions[" + json.dumps(robot_id, ensure_ascii=True) + "]"
    return [{**error, "path": error["path"].replace("$.action", prefix, 1)} for error in errors]


def validate_payload(payload: Any, robot_ids: Iterable[str], *,
                     note_max_chars: int = DEFAULT_NOTE_MAX_CHARS) -> list[dict[str, str]]:
    """Pure whole-team validation, also usable for injected central policies."""
    expected = tuple(robot_ids)
    try:
        # Injected policies must obey the same JSON-value boundary as a model.
        _decode(json.dumps(payload, ensure_ascii=True, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        return [_error("invalid_json_value", "$", "The team payload must contain finite JSON values.")]
    if _contains_invalid_unicode(payload):
        return [_error("invalid_unicode", "$", "All JSON strings and keys must contain valid Unicode.")]
    if not isinstance(payload, dict) or set(payload) != {"decision_basis", "robot_actions", "memory_update"}:
        return [_error("invalid_team_shape", "$", "Use exactly decision_basis, robot_actions and memory_update.")]
    errors = _validate_action_payload({"decision_basis": payload["decision_basis"],
        "action": {"name": "wait", "parameters": {}}, "communication": None,
        "memory_update": payload["memory_update"]}, note_max_chars=note_max_chars)
    actions = payload["robot_actions"]
    if not isinstance(actions, dict):
        return errors + [_error("robot_membership", "$.robot_actions", "Provide one action for every controlled robot.")]
    if set(actions) != set(expected):
        errors.append(_error("robot_membership", "$.robot_actions", "Provide exactly the controlled robot IDs."))
    for robot_id in expected:
        if robot_id not in actions:
            continue
        action = actions[robot_id]
        if isinstance(action, dict) and action.get("name") == "continue":
            if set(action) != {"name", "parameters"} or action.get("parameters") != {}:
                errors.append(_error("invalid_continue", "$.robot_actions[" + json.dumps(robot_id) + "]",
                    "continue requires only name and empty parameters, with no subtype."))
        else:
            errors.extend(_path_errors(_validate_action_payload({"decision_basis": "Validate the supplied action.",
                "action": action, "communication": None, "memory_update": {"note": None}}, note_max_chars=note_max_chars), robot_id))
    return errors


def process_response(raw: str | None, robot_ids: Iterable[str],
                     finish_reason: str | None = None, *,
                     note_max_chars: int = DEFAULT_NOTE_MAX_CHARS) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one complete team reply; no action or memory is committed here.

    The only repairs are a whole-object JSON fence and the established unambiguous
    subtype move. Individual robot repairs retain their exact team-scoped paths.
    """
    expected = tuple(robot_ids)
    if not expected or len(set(expected)) != len(expected) or any(not isinstance(key, str) or not key for key in expected):
        raise ValueError("robot_ids must contain unique nonempty names")
    audit: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "raw_response": raw,
        "status": "invalid", "original_errors": [], "repairs": [],
        "normalized_response": None, "errors": [], "finish_reason": finish_reason}
    if not isinstance(raw, str) or not raw.strip():
        audit["errors"] = [_error("empty_response", "$", "Return one non-empty team JSON object.")]
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
        body = fence.group("body")
        audit["repairs"].append({"rule_id": "remove_complete_json_fence", "changes": [
            {"path": "$", "before": raw, "after": body}]})
        try:
            payload = _decode(body)
        except _DecodeFailure as inner:
            audit["errors"] = [inner.error]
            raise InvalidDecisionOutput(audit) from None
    if _contains_invalid_unicode(payload):
        audit["errors"] = [_error("invalid_unicode", "$", "All JSON strings and keys must contain valid Unicode.")]
    elif not isinstance(payload, dict) or set(payload) != {"decision_basis", "robot_actions", "memory_update"}:
        audit["errors"] = [_error("invalid_team_shape", "$", "Use exactly decision_basis, robot_actions and memory_update.")]
    else:
        # Use the established contract for basis and memory, without changing it.
        envelope = {"decision_basis": payload["decision_basis"], "action": {"name": "wait", "parameters": {}},
                    "communication": None, "memory_update": payload["memory_update"]}
        audit["errors"].extend(_validate_action_payload(envelope, note_max_chars=note_max_chars))
        actions = payload["robot_actions"]
        if not isinstance(actions, dict) or set(actions) != set(expected):
            audit["errors"].append(_error("robot_membership", "$.robot_actions",
                "Provide exactly one action for every controlled robot, with no missing or extra robot IDs."))
        if isinstance(actions, dict):
            try:
                candidate = copy.deepcopy(payload)
            except RecursionError:
                audit["errors"].append(_error("structure_too_deep", "$", "The JSON structure is too deeply nested."))
                candidate = None
            if candidate is not None:
                for robot_id in expected:
                    if robot_id not in actions:
                        continue
                    action = actions[robot_id]
                    if isinstance(action, dict) and action.get("name") == "continue":
                        if set(action) != {"name", "parameters"} or action.get("parameters") != {}:
                            audit["errors"].append(_error("invalid_continue", "$.robot_actions[" + json.dumps(robot_id) + "]",
                                "continue requires only name and empty parameters, with no subtype."))
                        continue
                    wrapper = {"decision_basis": "Validate the supplied action.", "action": action,
                               "communication": None, "memory_update": {"note": None}}
                    try:
                        normalized, action_audit = process_action_response(json.dumps(wrapper, ensure_ascii=True, allow_nan=False),
                                                                          note_max_chars=note_max_chars)
                    except InvalidDecisionOutput as exc:
                        action_audit = exc.audit
                        audit["errors"].extend(_path_errors(action_audit["errors"], robot_id))
                    else:
                        candidate["robot_actions"][robot_id] = normalized["action"]
                    audit["original_errors"].extend(_path_errors(action_audit["original_errors"], robot_id))
                    for repair in action_audit["repairs"]:
                        repair = copy.deepcopy(repair)
                        repair["robot_id"] = robot_id
                        prefix = "$.robot_actions[" + json.dumps(robot_id, ensure_ascii=True) + "]"
                        for change in repair.get("changes", []):
                            for field in ("path", "from", "to"):
                                if field in change:
                                    change[field] = change[field].replace("$.action", prefix, 1)
                        audit["repairs"].append(repair)
                audit["normalized_response"] = candidate
    if finish_reason is not None and finish_reason != "stop":
        audit["errors"].append(_error("incomplete_response", "$", "The team response did not finish normally."))
    if audit["errors"]:
        audit["original_errors"].extend(copy.deepcopy(audit["errors"]))
        raise InvalidDecisionOutput(audit)
    audit["status"] = "repaired" if audit["repairs"] else "valid"
    return copy.deepcopy(audit["normalized_response"]), audit


__all__ = ["PROTOCOL_VERSION", "InvalidDecisionOutput", "process_response", "validate_payload"]
