"""Single-agent payload contracts; public action results use a shared envelope."""
from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class Action(TypedDict):
    name: str
    subtype: NotRequired[str]
    parameters: NotRequired[dict[str, Any]]


class Observation(TypedDict):
    perception_ticks: list[dict[str, Any]]
    latest_tick: int | None
    raw_visual_reference: list[str] | None
    decision_trigger: str
    trigger_context: NotRequired[dict[str, Any]]


class MemoryUpdate(TypedDict):
    note: str | None


class Decision(TypedDict):
    action: str
    action_payload: Action
    decision_basis: str
    memory_update: MemoryUpdate


class ExecutionFeedback(TypedDict):
    action: str | None
    action_params: dict[str, Any] | None
    status: NotRequired[str]
    reason: NotRequired[str]
    data: NotRequired[dict[str, Any]]


class ActionResult(TypedDict):
    status: str
    reason: str
    data: dict[str, Any]
