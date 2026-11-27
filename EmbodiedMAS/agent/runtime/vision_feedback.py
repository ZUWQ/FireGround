"""Versioned, episode-local command/effect receipts; no inference or RPCs."""
from __future__ import annotations

import copy
from typing import Any

PROTOCOL = "execution_effect_v2"
LEGACY = "legacy"
_FIELDS = ("feedback_protocol", "execution_feedback", "effect_feedback")


def validate_protocol(value: Any) -> str:
    if not isinstance(value, str) or value not in {LEGACY, PROTOCOL}:
        raise ValueError("vision_feedback_protocol must be legacy or execution_effect_v2")
    return value


def _receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(result)
    for key in _FIELDS:
        receipt.get("data", {}).pop(key, None)
    return receipt


class VisionActionFeedback:
    """One active action, with completed records retained by existing memory.

    A verification result may keep a failed command primary and attach an
    independent observed_effect. Neither stage establishes causation for the
    other. Cross-frame object identities and success are never inferred here.
    """

    def __init__(self) -> None:
        self.action_id: int | None = None
        self.execution: dict[str, Any] | None = None
        self.effect: dict[str, Any] | None = None

    def record(self, action_id: int, result: dict[str, Any], *, phase: str) -> None:
        if phase not in {"execution", "verification"}:
            raise ValueError("unsupported action-feedback phase")
        if self.action_id != action_id:
            self.action_id, self.execution, self.effect = action_id, None, None
        if phase == "execution":
            self.execution, self.effect = _receipt(result), None
        else:
            observed = result.get("data", {}).get("observed_effect")
            self.effect = _receipt(observed if isinstance(observed, dict) else result)

    def enrich(self, action_id: int, result: dict[str, Any]) -> dict[str, Any]:
        enriched = _receipt(result)
        if self.action_id == action_id:
            enriched["data"].update(
                feedback_protocol=PROTOCOL,
                execution_feedback=copy.deepcopy(self.execution),
                effect_feedback=copy.deepcopy(self.effect),
            )
        return enriched
