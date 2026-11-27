"""Shared result envelope for command records and post-action effect verdicts."""
from __future__ import annotations

import copy
from typing import Any

STATUSES = frozenset({"completed", "failed", "interrupted", "unknown"})


def action_result(status: str, reason: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(f"Invalid action status: {status!r}")
    return {"status": status, "reason": str(reason), "data": copy.deepcopy(data or {})}


def normalize_action_result(action: str, raw: Any) -> dict[str, Any]:
    """Also accept legacy low-level RPC values and injected test bodies."""
    if isinstance(raw, dict) and "status" in raw:
        if (isinstance(raw["status"], str) and raw["status"] in STATUSES and isinstance(raw.get("reason"), str)
                and isinstance(raw.get("data"), dict)):
            return action_result(raw["status"], raw["reason"], raw["data"])
        return action_result("failed", "invalid_result", {"execution_result": raw})
    data = {"execution_result": raw}
    if action == "suppress_fire":
        data["observed_effect"] = "unknown"
    if raw is None:
        return action_result("failed", "no_result", data)
    if action == "instruct_civilian" and isinstance(raw, list):
        return action_result("completed" if raw else "failed",
                             "instruction_sent" if raw else "no_recipients", data)
    if isinstance(raw, dict) and isinstance(raw.get("success"), bool):
        success = raw["success"]
        return action_result("completed" if success else "failed",
                             "executed" if success else "execution_failed", data)
    if isinstance(raw, bool):
        return action_result("completed" if raw else "failed",
                             "executed" if raw else "execution_failed", data)
    if isinstance(raw, str):
        if raw.lower() == "success":
            return action_result("completed", "executed", data)
        return action_result("failed", raw if raw.lower().startswith("fail") else "invalid_result", data)
    return action_result("failed", "invalid_result", data)


def navigation_feedback(execution: dict | None, observed: dict) -> dict:
    """Use verified XY arrival, retaining cancellation and pre-execution errors."""
    if execution is not None:
        if execution.get("status") == "interrupted" or execution.get("data", {}).get("execution_started") is False:
            return copy.deepcopy(execution)
        if observed.get("status") == "unknown" and execution.get("status") == "failed":
            result = copy.deepcopy(execution)
            result["data"]["observed_effect"] = copy.deepcopy(observed)
            return result
    return copy.deepcopy(observed)
