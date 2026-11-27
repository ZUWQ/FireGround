"""LLM runtime text formatting, JSON helpers, and label normalization."""

from __future__ import annotations

import json
from typing import Any

from EmbodiedMAS.serialization import json_safe, compact_json


def _format_entity_list_for_llm(entities: Any) -> str:
    if isinstance(entities, list) and entities and all(isinstance(item, dict) for item in entities):
        formatted_items = ",\n".join(f"    {compact_json(item)}" for item in entities)
        return f"[\n{formatted_items}\n  ]"
    return compact_json(entities)


def format_observation_for_llm(
    observation: dict[str, Any], *, input_modality: str = "text",
) -> str:
    """Keep complete snapshots and temporal provenance from the adapter."""
    return compact_json(observation)


def format_memory_state_for_llm(memory_state: dict[str, Any]) -> str:
    """Preserve every public memory field, including configured note limits."""
    lines = []
    for key, value in memory_state.items():
        formatted = _format_entity_list_for_llm(value) if key == "short_term" else compact_json(value)
        lines.append(f"  {compact_json(key)}: {formatted}")
    return "{\n" + ",\n".join(lines) + "\n}" if lines else "{}"


def format_runtime_for_llm(
    runtime_input: dict[str, Any],
    *,
    input_modality: str = "text",
) -> str:
    data = json_safe(runtime_input)
    keys = list(data.keys())
    lines = ["{"]
    for index, key in enumerate(keys):
        value = data[key]
        trailing = "," if index < len(keys) - 1 else ""
        if key == "current_observation" and isinstance(value, dict):
            formatted = format_observation_for_llm(value, input_modality=input_modality)
        elif key == "memory_state" and isinstance(value, dict):
            formatted = format_memory_state_for_llm(value)
        else:
            formatted = compact_json(value)
        lines.append(f'  "{key}": {formatted}{trailing}')
    lines.append("}")
    return "\n".join(lines)


def parse_json_from_response(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def format_memory_log_line(mem: dict[str, Any]) -> str:
    parts = [
        f"step={mem.get('step')}",
        f"action={mem.get('action')}",
    ]
    params = mem.get("action_params")
    if params:
        parts.append(f"params={params}")
    if "action_result" in mem:
        parts.append(f"result={mem['action_result']!r}")
    else:
        # Older session records only contain these per-action summaries.
        for key, label in (
            ("navigate_success", "nav_ok"),
            ("suppress_fire_result", "suppress"),
            ("instruct_civilian_result", "instruct"),
            ("instruct_follow_received", "follow_rx"),
        ):
            if key in mem:
                parts.append(f"{label}={mem[key]!r}")
    if mem.get("decision_basis"):
        parts.append(f"basis={mem['decision_basis']!r}")
    return "  " + " | ".join(parts)
