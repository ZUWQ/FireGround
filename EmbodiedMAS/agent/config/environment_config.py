"""Agent runtime settings loaded from environment_config.json."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from EmbodiedMAS.serialization import json_safe
from EmbodiedMAS.environment.safety import validate_safe_areas
from EmbodiedMAS.agent.memory import DEFAULT_NOTE_MAX_CHARS, validate_memory_limits

_CONFIG_PATH = Path(__file__).parent / "environment_config.json"
_cached_config: dict[str, Any] | None = None
_cached_config_path: Path | None = None

EXPLORATION_DEFAULTS = {
    "inset_m": 1.0,
    "acceptance_radius_m": 0.5,
    "endpoint_tolerance_m": 2.0,
    "segment_timeout_sec": 30.0,
    "total_timeout_sec": 120.0,
}


def exploration_options(value: Any = None) -> dict[str, float]:
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - (set(EXPLORATION_DEFAULTS) | {"projection_tolerance_m"}):
        raise ValueError("exploration must contain only supported exploration options")
    options = dict(EXPLORATION_DEFAULTS)
    for key, supplied in value.items():
        if isinstance(supplied, bool):
            raise ValueError(f"exploration.{key} must be a positive finite number")
        try:
            number = float(supplied)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"exploration.{key} must be a positive finite number") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"exploration.{key} must be a positive finite number")
        # Accept legacy frozen configs without retaining their old projection
        # distance gate. Room/path membership now decides projection validity.
        if key != "projection_tolerance_m":
            options[key] = number
    if options["acceptance_radius_m"] > options["endpoint_tolerance_m"]:
        raise ValueError("exploration acceptance radius must not exceed endpoint tolerance")
    return options



DEFAULT_VISION_MAX_TOKENS = 8192


def _positive_token_budget(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def output_token_budget(modality: str, runtime_config: dict[str, Any], *,
                        explicit: int | None = None, text_default: int = 1000) -> int:
    """Resolve the completion budget without consulting model credentials.

    Reasoning and final JSON share this limit on the experiment gateway. An
    explicit constructor override takes precedence over the visual config.
    """
    if explicit is not None:
        return _positive_token_budget(explicit, "max_tokens")
    if modality == "vision":
        return _positive_token_budget(
            runtime_config.get("vision_max_tokens", DEFAULT_VISION_MAX_TOKENS),
            "vision_max_tokens")
    return _positive_token_budget(runtime_config.get("text_max_tokens", text_default), "text_max_tokens")


def perception_interval(modality: str, config: dict[str, Any]) -> float:
    """Vision has its own sampling interval; text retains the common setting."""
    key = "vision_perception_interval_sec" if modality == "vision" else "perception_interval_sec"
    value = config.get(key, 4.0 if modality == "vision" else 2.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return float(value)


def load_environment_config(*, reload: bool = False) -> dict[str, Any]:
    global _cached_config, _cached_config_path
    # Batch subprocesses receive a complete private config; the shared project
    # file remains the default for interactive runs. Cache by source path too.
    configured_path = os.environ.get("EMBODIED_ENVIRONMENT_CONFIG")
    config_path = Path(configured_path).expanduser().resolve() if configured_path else _CONFIG_PATH.resolve()
    if _cached_config is not None and _cached_config_path == config_path and not reload:
        return _cached_config

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"[EnvironmentConfig] cannot read {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"[EnvironmentConfig] invalid config format in {config_path}")

    prompt_env = data.get("prompt_environment")
    if not isinstance(prompt_env, dict):
        raise ValueError(
            f"[EnvironmentConfig] missing or invalid prompt_environment in {config_path}"
        )

    navigation_tolerance = data.get("navigation_tolerance_m", 2.0)
    if isinstance(navigation_tolerance, bool) or not isinstance(navigation_tolerance, (int, float)) or not math.isfinite(navigation_tolerance) or navigation_tolerance <= 0:
        raise ValueError("navigation_tolerance_m must be a positive finite number")
    scope = str(data.get("observation_scope", "local")).lower()
    if scope not in {"local", "global"}:
        raise ValueError("observation_scope must be local or global")
    record_mode = data.get("record_mode")
    if record_mode not in {None, "full_snapshot", "action_first_seen_or_fire_state_change"}:
        raise ValueError("unsupported record_mode")
    coordination = str(data.get("coordination_mode", "none"))
    if coordination not in {"none", "D", "C", "H"}:
        raise ValueError("coordination_mode must be none, D, C or H")
    time_limit = data.get("time_limit_sec")
    if time_limit is not None and (isinstance(time_limit, bool) or
            not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit_sec must be a positive finite number or null")
    history_size = data["max_memory_size"]
    note_max_chars = data.get("memory_note_max_chars", DEFAULT_NOTE_MAX_CHARS)
    validate_memory_limits(history_size, note_max_chars)
    session_id = data.get("session_id")
    if type(data.get("vision_grace_pause", False)) is not bool:
        raise ValueError("vision_grace_pause must be a boolean")
    if "vision_feedback_protocol" in data:
        from EmbodiedMAS.agent.runtime.vision_feedback import validate_protocol
        validate_protocol(data["vision_feedback_protocol"])
    _cached_config = {
        "SafeArea": validate_safe_areas(data.get("SafeArea")),
        "observation_scope": scope,
        "record_mode": record_mode,
        "coordination_mode": coordination,
        "time_limit_sec": float(time_limit) if time_limit is not None else None,
        "navigation_tolerance_m": float(navigation_tolerance),
        "exploration": exploration_options(data.get("exploration")),
        "prompt_environment": {str(k): json_safe(v) for k, v in prompt_env.items()},
        "session_id": str(session_id) if session_id is not None else None,
        "pause_during_llm": bool(data["pause_during_llm"]),
        "max_memory_size": history_size,
        "memory_note_max_chars": note_max_chars,
        "burn_time": float(data.get("burn_time", 10.0)),
        "perception_interval_sec": perception_interval("text", data),
        "vision_perception_interval_sec": perception_interval("vision", data),
        "vision_backend": str(data.get("vision_backend", "grounded_sam")),
        "vision_max_tokens": _positive_token_budget(
            data.get("vision_max_tokens", DEFAULT_VISION_MAX_TOKENS), "vision_max_tokens"),
        "perception_capture_timeout_sec": float(data.get("perception_capture_timeout_sec", 60.0)),
    }
    if "vision_grace_pause" in data:
        _cached_config["vision_grace_pause"] = data["vision_grace_pause"]
    if "vision_feedback_protocol" in data:
        _cached_config["vision_feedback_protocol"] = data["vision_feedback_protocol"]
    if "text_max_tokens" in data:
        _cached_config["text_max_tokens"] = _positive_token_budget(data["text_max_tokens"], "text_max_tokens")
    _cached_config_path = config_path
    return _cached_config


def get_burn_time() -> float:
    return load_environment_config()["burn_time"]
