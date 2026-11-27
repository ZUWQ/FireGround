from __future__ import annotations

from typing import Any

from EmbodiedMAS.agent.memory import DEFAULT_NOTE_MAX_CHARS

from EmbodiedMAS.agent.prompt.visual_backend import action_space_for_backend, resolve_visual_backend
from EmbodiedMAS.agent.prompt.src.config_loader import ConfigLoader
from EmbodiedMAS.agent.prompt.src.initial_prompt_builder import PromptBuilder


_SCENARIO_CONFIG_FILE = "scenario_example.json"

_INPUT_MODALITY_FILES = {
    "vision": "vision_observation.json",
    "text": "text_observation.json",
}

_AGENT_PROFILE_FILES = {
    "RR": "rescue_agent.json",
    "FR": "firefighting_agent.json",
    "ER": "exploration_agent.json",
    "HF": "human_firefighter.json",
}

_COOPERATION_MODE_FILES = {
    "none": "none.json",
    "C": "centralized.json",
    "D": "decentralized.json",
    "H": "hybrid.json",
}


def build_initial_prompt(
    input_modality: str,
    agent_class: str,
    cooperation_type: str | None,
    *,
    observation_scope: str = "local",
    record_mode: str | None = None,
    episode_briefing: dict[str, Any] | None = None,
    vision_backend: str | None = None,
) -> str:
    """Build the selected English prompt; the original three arguments still work.

    input_modality is text/vision; agent_class is RR/FR/ER/HF; cooperation_type
    is none/D/C/H (None also means none). Text-local defaults to action-local
    fire deltas; global and visual observations use full_snapshot.
    """
    loader = ConfigLoader()
    builder = PromptBuilder(loader)
    profile = loader.load_agent_profile(
        _resolve_label(agent_class, _AGENT_PROFILE_FILES, "agent_class"))
    coordination = loader.load_coordination_mode(
        _resolve_label("none" if cooperation_type is None else cooperation_type,
                       _COOPERATION_MODE_FILES, "cooperation_type"))
    selected_backend = resolve_visual_backend(input_modality, vision_backend)
    raw = _normalize_label(input_modality) == "vision" and selected_backend == "raw_vlm"
    hybrid = _normalize_label(input_modality) == "vision" and selected_backend == "grounded_sam_pixel"
    modality = loader.load_input_modality("sam_pixel_observation.json" if hybrid else "raw_vlm_observation.json" if raw else
        _resolve_label(input_modality, _INPUT_MODALITY_FILES, "input_modality"))
    return builder.build_initial_prompt(
        scenario_config=loader.load_scenario_config(_SCENARIO_CONFIG_FILE),
        agent_profile=profile,
        coordination_mode=coordination,
        action_space=action_space_for_backend(loader.load_action_space(
            f"{builder.action_space_name_for_agent(profile['agent_type'])}.json"), raw=raw, hybrid=hybrid),
        input_modality=modality,
        observation_scope=observation_scope,
        record_mode=record_mode,
        episode_briefing=episode_briefing,
        output_protocol_path="sam_pixel_output_protocol.md" if hybrid else "raw_vlm_output_protocol.md" if raw else "output_protocol.md",
    )


def build_runtime_input(
    *,
    step: int,
    agent_id: str,
    agent_state: dict[str, Any],
    current_observation: dict[str, Any],
    teammate_messages: list[dict[str, Any]] | None = None,
    environment_signals: list[dict[str, Any]] | None = None,
    memory_state: dict[str, Any] | None = None,
    previous_execution_feedback: dict[str, Any] | None = None,
    previous_decision_feedback: dict[str, Any] | None = None,
    super_agent_assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-step runtime input package.

    Required inputs are already-processed runtime values:
    step is an int timestep; agent_id is a runtime agent id string;
    agent_state and current_observation are dicts.
    Optional communication, signal, memory, feedback, and assignment inputs are
    lists or dicts matching the runtime JSON structure.
    """
    return {
        "step": step,
        "agent_id": agent_id,
        "agent_state": agent_state,
        "current_observation": current_observation,
        "signal_channel": {
            "teammate_messages": teammate_messages or [],
            "environment_signals": environment_signals or [],
            "super_agent_assignment": super_agent_assignment,
        },
        "memory_state": memory_state if memory_state is not None else {"note": "", "note_max_chars": DEFAULT_NOTE_MAX_CHARS, "short_term": []},
        "previous_execution_feedback": previous_execution_feedback
        or {"action": None},
        "previous_decision_feedback": previous_decision_feedback,
    }


def _resolve_label(label: str, mapping: dict[str, str], label_name: str) -> str:
    normalized = _normalize_label(label)
    candidates = (normalized, normalized.lower(), normalized.upper())
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]

    expected = ", ".join(_public_labels(mapping))
    raise ValueError(f"Unknown {label_name} label {label!r}. Expected one of: {expected}.")


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().replace("-", "_").replace(" ", "_")
    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    return normalized


def _public_labels(mapping: dict[str, str]) -> list[str]:
    labels: list[str] = []
    seen_files: set[str] = set()
    for label, file_name in mapping.items():
        if file_name in seen_files:
            continue
        labels.append(label)
        seen_files.add(file_name)
    return labels


__all__ = ["build_initial_prompt", "build_runtime_input"]
