"""Opt-in action business time for a vision episode's shared pause clock.

Legacy text and vision actions retain their existing clocks. Long-running UE
navigation gets a separate, finite wall-clock transport insurance; its business
budget remains active environment time.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any


def controller_for(actions: Any):
    if getattr(actions, "_modality", None) != "vision":
        return None
    context = getattr(actions, "_context", None)
    controller = getattr(context, "_emas_vision_pause_controller", None)
    # Runners install the shared controller before deployment/visual warm-up.
    # Those preparation calls are not inside the formal episode clock yet.
    if controller is not None and getattr(controller, "started", True) is False:
        return None
    return controller


async def wait_running(actions: Any) -> None:
    controller = controller_for(actions)
    if controller is not None:
        await controller.wait_running()


async def wait_for_cleanup(actions: Any) -> None:
    """Restore owned orientation even after the episode's active budget ends."""
    controller = controller_for(actions)
    if controller is not None:
        resumed = getattr(controller, "wait_resumed", None)
        if callable(resumed):
            await resumed()
        else:
            await controller.wait_running()


def active_now(actions: Any) -> float:
    controller = controller_for(actions)
    return controller.active_now() if controller is not None else asyncio.get_running_loop().time()


def elapsed_now(actions: Any) -> float:
    controller = controller_for(actions)
    return controller.active_now() if controller is not None else time.perf_counter()


async def sleep_active(actions: Any, seconds: float) -> None:
    controller = controller_for(actions)
    if controller is None:
        await asyncio.sleep(seconds)
    else:
        await controller.sleep_active(seconds)
        await controller.wait_running()


@asynccontextmanager
async def timeout_active(actions: Any, seconds: float):
    controller = controller_for(actions)
    if controller is None:
        async with asyncio.timeout(seconds):
            yield
    else:
        async with controller.timeout_active(seconds):
            yield
