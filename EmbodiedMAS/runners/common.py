"""Shared preparation and lifecycle helpers for SA and team runners.

Runners own episodes. Individual runtimes own perception, cognition and actions;
future centralized runners can reuse preparation without creating a LLM per body.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import tongsim as ts

from EmbodiedMAS.agent.visual_backends import VISUAL_BACKENDS
from EmbodiedMAS.agent.config.environment_config import load_environment_config
from EmbodiedMAS.agent.prompt.emas_prompt import build_initial_prompt
from EmbodiedMAS.agent.perception.adapters import ObservationAdapter
from EmbodiedMAS.environment.entities import format_agent_name

Z_HEIGHT = 1000
ROLES = ("FR", "RR", "HF", "ER")
CAPABILITIES = {role: ["navigate", "instruct_civilian", "wait", "explore"] +
                (["suppress_fire"] if role in {"FR", "HF"} else []) for role in ROLES}
DEFAULT_POSITIONS = (
    {"x": -300, "y": 100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    {"x": -300, "y": -100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    {"x": 300, "y": 100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    {"x": 300, "y": -100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
)


def validate_max_steps(max_steps: int | None) -> None:
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps must be a nonnegative integer or None")


def validate_options(context: Any, *, mode: str, agent_types: list[str],
                     max_steps: int | None = None, require_budget: bool = True,
                     require_provider: bool = True, **runtime_options: Any) -> dict[str, Any]:
    """Reject unsupported runs before spawning bodies or touching UE state."""
    cfg = load_environment_config()
    if mode not in {"text", "vision"}:
        raise ValueError("mode must be text or vision")
    if any(role not in ROLES for role in agent_types):
        raise ValueError("agent_types must use FR/RR/HF/ER")
    validate_max_steps(max_steps)
    options = dict(runtime_options)
    scope = options.get("observation_scope") or cfg["observation_scope"]
    record_mode = options.get("record_mode") or cfg.get("record_mode")
    cooperation = options.get("cooperation_type") or cfg["coordination_mode"]
    backend = options.get("vision_backend") or cfg.get("vision_backend", "grounded_sam")
    if mode == "vision" and backend not in VISUAL_BACKENDS:
        raise ValueError("vision_backend must be one of: grounded_sam, raw_vlm, grounded_sam_pixel")
    limit = options.get("time_limit_sec")
    if limit is None:
        limit = cfg.get("time_limit_sec")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("time_limit_sec must be positive and finite")
    if cfg.get("pause_during_llm", False) and not (
            limit is not None and getattr(context, "_emas_t1_pause_authorized", False)
            and mode == "text" and scope == "global" and agent_types == ["FR"]):
        raise ValueError("Timed model pause requires an explicitly validated T1 task")
    if require_budget and limit is None and max_steps is None:
        raise ValueError("run requires a task time limit or decision cap")
    if mode == "vision" and scope == "global":
        raise ValueError("global vision is not defined; use local vision or global text")
    ObservationAdapter(mode, observation_scope=scope, record_mode=record_mode)
    from EmbodiedMAS.agent.runtime.vision_protocol import validate_mode
    vision_pause = options.get('vision_grace_pause', cfg.get('vision_grace_pause', False))
    validate_mode(enabled=vision_pause, mode=mode, scope=scope, duration=limit,
                  pause_during_llm=cfg.get('pause_during_llm', False))
    if vision_pause or 'vision_grace_pause' in options:
        options['vision_grace_pause'] = vision_pause
    if require_provider and scope == "global":
        from EmbodiedMAS.agent.perception.global_oracle import ensure_global_provider
        ensure_global_provider(context)
    for role in agent_types:
        build_initial_prompt(mode, role, cooperation, observation_scope=scope,
                             record_mode=record_mode, episode_briefing=options.get("episode_briefing"), vision_backend=backend)
    options.update(observation_scope=scope, record_mode=record_mode,
                   cooperation_type=cooperation, time_limit_sec=limit, vision_backend=backend)
    return options


def validate_position(position: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(position, dict):
        raise ValueError("agent position must be an object")
    result = dict(position)
    for key in ("x", "y", "z"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"position.{key} must be a finite number in UU")
    rotation = result.get("rotation")
    if rotation is not None:
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rotation
        ):
            raise ValueError("position.rotation must contain three finite Euler angles in degrees")
        result["rotation"] = tuple(rotation)
    return result


def preload_vision(mode: str, vision_backend: str | None = None) -> None:
    if mode != "vision":
        return
    backend = vision_backend or load_environment_config().get("vision_backend", "grounded_sam")
    if mode == "vision" and backend == "raw_vlm":
        return
    if mode == "vision":
        if backend not in VISUAL_BACKENDS:
            raise ValueError("vision_backend must be one of: grounded_sam, raw_vlm, grounded_sam_pixel")
        from EmbodiedMAS.agent.perception.vision_pipeline.perception_preload import preload_grounded_sam2_if_vision
        preload_grounded_sam2_if_vision(mode)


async def deploy_runtime(runtime: Any, *, mode: str, role: str, index: int,
                         position: dict[str, Any]) -> None:
    position = validate_position(position)
    rotation = position.get("rotation")
    await runtime.deploy(
        name=format_agent_name(modality=mode, agent_type=role, index=index),
        location=ts.Vector3(position["x"], position["y"], position["z"]),
        rotation=(ts.math.euler_to_quaternion(ts.Vector3(rotation), is_degree=True)
                  if rotation is not None else None),
    )


async def close_runtimes(runtimes: list[Any]) -> None:
    """Attempt every cleanup even when one body fails to close."""
    async def close_one(runtime: Any) -> None:
        errors = []
        try:
            await runtime.close()
        except BaseException as exc:
            errors.append(exc)
        # Bodies distinguish actors they spawned from externally bound bodies.
        destroy = getattr(getattr(runtime, "agent", None), "destroy_spawned_actor", None)
        if callable(destroy):
            try:
                await destroy()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Runtime or owned actor cleanup failed", errors)

    outcomes = await asyncio.gather(*(close_one(runtime) for runtime in runtimes), return_exceptions=True)
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if errors:
        raise BaseExceptionGroup("Agent cleanup failed", errors)


async def run_episode(runtimes: list[Any], *, max_steps: int | None = None,
                      retry_interval_sec: float = 0.25, monitor: Any = None) -> None:
    """Run each runtime once, with a shared start and bounded transport retries.

    All runtimes must be deployed and registered before entry. The caller owns
    close(); this function always drains its run/retry tasks, including on error.
    """
    validate_max_steps(max_steps)
    if isinstance(retry_interval_sec, bool) or not isinstance(retry_interval_sec, (int, float)) or not math.isfinite(retry_interval_sec) or retry_interval_sec <= 0:
        raise ValueError("retry_interval_sec must be positive and finite")
    if max_steps != 0:
        starts = [getattr(runtime, "_episode_started_at", None) for runtime in runtimes]
        if any(start is not None for start in starts):
            if any(start is None for start in starts) or len(set(starts)) != 1:
                raise ValueError("Team members must share one episode clock")
            # A segmented SA continuation keeps its original clock and deadline.
        else:
            started_at = time.monotonic()
            for runtime in runtimes:
                runtime.start_episode(started_at=started_at)
    controller = (getattr(getattr(runtimes[0], 'context', None), '_emas_vision_pause_controller', None)
                  if runtimes else None)
    if controller is not None and max_steps != 0:
        for runtime in runtimes:
            if runtime._input_modality != 'vision' or runtime._vision_pause_controller is not controller:
                raise ValueError('A paused-vision episode cannot mix world owners or Oracle runtimes')
            controller.register_runtime(runtime)
        controller.start(runtimes[0]._episode_started_at, runtimes[0]._time_limit_sec)
    if monitor is not None:
        if max_steps == 0:
            await monitor.capture_snapshot("initial")
        elif monitor.started_at is None:
            await monitor.start(started_at=runtimes[0]._episode_started_at,
                                deadline=runtimes[0]._episode_deadline)
    runs = [asyncio.create_task(runtime.run(max_decisions=max_steps, agent_idx=index),
                                name=f"emas-agent-{index}") for index, runtime in enumerate(runtimes)]

    async def retry_messages() -> None:
        while True:
            await asyncio.sleep(retry_interval_sec)
            for runtime, task in zip(runtimes, runs):
                if (not task.done() and runtime.outgoing_messages
                        and (runtime.task_time["remaining_sec"] is None or runtime.task_time["remaining_sec"] > 0)):
                    await runtime.flush_outgoing_messages()

    retry = asyncio.create_task(retry_messages(), name="emas-message-retry")
    # Monitor the retry worker too, so failures are never lost.
    try:
        pending = set(runs)
        while pending:
            done, _ = await asyncio.wait(pending | {retry}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            pending.difference_update(done)
    finally:
        for task in [*runs, retry]:
            if not task.done():
                task.cancel()
        # Resume before draining orientation/control cleanup which may itself
        # wait for the shared physical resume barrier. Normal segmented SA runs
        # retain their clock and controller for the next decision-limited call.
        import sys
        abort = sys.exc_info()[0] is not None
        try:
            if controller is not None and (abort or any(r.end_reason == 'time_limit' for r in runtimes)):
                await controller.close()
        finally:
            await asyncio.gather(*runs, retry, return_exceptions=True)
