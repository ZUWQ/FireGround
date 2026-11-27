"""Explicit, vision-only grace-pause protocol; legacy paths remain opt-out."""
from __future__ import annotations

import math
from typing import Any

PROTOCOL = 'vision_grace_pause_v1'
GRACE_ACTIVE_SEC = 20.0
MODEL_ATTEMPT_WALL_SEC = 180.0
MODEL_MAX_ATTEMPTS = 2



def derive_wall_timeout(active_limit: float, callers: int, base_timeout: float = 900.0,
                        *, grace_sec: float = GRACE_ACTIVE_SEC,
                        pause_rpc_timeout_sec: float = 5.0) -> float:
    """A finite hang insurance covering every legal shared pause in an episode.

    Callers are sequential within one agent. Each can exhaust its grace at
    most ceil(active_limit / grace) times plus a boundary allowance. Adding
    each caller's full HTTP/retry/cleanup hold overestimates overlapping pauses.
    Business deadlines must still use the active environment clock.
    """
    for name, value in (("active_limit", active_limit), ("base_timeout", base_timeout),
                        ("grace_sec", grace_sec), ("pause_rpc_timeout_sec", pause_rpc_timeout_sec)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if isinstance(callers, bool) or not isinstance(callers, int) or callers < 1:
        raise ValueError("callers must be a positive integer")
    from EmbodiedMAS.agent.config.vision_llm_transport import (
        CLIENT_CLOSE_TIMEOUT_SEC, MAX_RETRY_DELAY_SEC,
    )
    crossings = callers * (math.ceil(active_limit / grace_sec) + 1)
    hold = (MODEL_MAX_ATTEMPTS * MODEL_ATTEMPT_WALL_SEC
            + (MODEL_MAX_ATTEMPTS - 1) * MAX_RETRY_DELAY_SEC
            + CLIENT_CLOSE_TIMEOUT_SEC + 2 * pause_rpc_timeout_sec)
    result = max(base_timeout, active_limit) + crossings * hold
    if not math.isfinite(result):
        raise ValueError("Derived vision wall timeout must be finite")
    return float(result)


def validate_mode(*, enabled: Any, mode: str, scope: str, duration: Any,
                  pause_during_llm: bool = False) -> bool:
    if type(enabled) is not bool:
        raise ValueError('vision_grace_pause must be a boolean')
    if not enabled:
        return False
    if mode != 'vision' or scope != 'local':
        raise ValueError('vision_grace_pause requires local vision; Oracle is unchanged')
    if pause_during_llm:
        raise ValueError('vision_grace_pause cannot be combined with the T1 pause switch')
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError('vision_grace_pause requires a positive finite task time limit')
    return True


def validate_task(task: dict) -> None:
    enabled = validate_mode(enabled=task.get('vision_grace_pause', False),
                            mode=task.get('mode'), scope=task.get('observation_scope'),
                            duration=task.get('time_limit_sec'),
                            pause_during_llm=task.get('pause_during_llm', False))
    if enabled:
        if task.get('runner') not in {'SA', 'DMAS', 'CMAS'}:
            raise ValueError('vision_grace_pause requires an agent episode')
        if not isinstance(task.get('llm_profile'), str) or not task['llm_profile'].strip():
            raise ValueError('Explicit model profile required for vision_grace_pause; no model fallback')


def bind_controller(context: Any, *, enabled: bool, mode: str, scope: str,
                    duration: Any, pause_during_llm: bool = False):
    validate_mode(enabled=enabled, mode=mode, scope=scope, duration=duration,
                  pause_during_llm=pause_during_llm)
    controller = getattr(context, '_emas_vision_pause_controller', None)
    if controller is not None:
        if not enabled or mode != 'vision':
            raise ValueError('An owned paused-vision world cannot mix protocols or Oracle agents')
        return controller
    if not enabled:
        return None
    if getattr(context, '_emas_t1_pause_controller', None) is not None:
        raise ValueError('Cannot combine vision and T1 pause controllers')
    if getattr(context, '_emas_monitor', None) is not None:
        raise RuntimeError('A vision pause controller must be bound before its episode monitor')
    from .vision_pause import VisionPauseController
    controller = VisionPauseController(context, grace_sec=GRACE_ACTIVE_SEC)
    context._emas_vision_pause_controller = controller
    return controller


async def release_controller(context: Any) -> None:
    controller = getattr(context, '_emas_vision_pause_controller', None)
    if controller is not None:
        try:
            await controller.close()
        finally:
            if getattr(context, '_emas_vision_pause_controller', None) is controller:
                delattr(context, '_emas_vision_pause_controller')


async def close_and_cleanup(context: Any, cleanup, *, release: bool = True) -> None:
    """Try world resume and every local cleanup even when either side fails."""
    controller = getattr(context, '_emas_vision_pause_controller', None)
    if controller is None:
        await cleanup()
        return
    errors = []
    try:
        await controller.close()
    except BaseException as exc:
        errors.append(exc)
    try:
        await cleanup()
    except BaseException as exc:
        errors.append(exc)
    finally:
        if release and getattr(context, '_emas_vision_pause_controller', None) is controller:
            delattr(context, '_emas_vision_pause_controller')
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup('Vision resume and local cleanup failed', errors)
