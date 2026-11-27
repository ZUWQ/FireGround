"""Opt-in, single-agent model-call pause bookkeeping.

This clock is active wall time, not an engine simulation clock. Only confirmed
pause/resume intervals are excluded. The caller owns the T1-only scope gate.
No global clocks, existing profiles, or default runtime behavior are modified.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

PauseRPC = Callable[[bool], Awaitable[None]]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class T1PauseBudgetExhausted(TimeoutError):
    """The T1 effective-time budget expired before a model request started."""


class T1PauseController:
    """Pause one owned UE world around one model call at a time.

    Runtime integration supplies ``_active_decision_timeout`` while inside its
    asyncio timeout context. Network/API timeouts remain ordinary wall time.
    """

    clock = "active_wall_seconds"

    def __init__(self, runtime: Any, *, pause_rpc: PauseRPC | None = None,
                 wall_clock: Callable[[], float] = time.monotonic,
                 rpc_timeout_sec: float = 5.0) -> None:
        if not _finite(rpc_timeout_sec) or rpc_timeout_sec <= 0:
            raise ValueError("rpc_timeout_sec must be finite and positive")
        self.runtime = runtime
        self._pause_rpc = pause_rpc or self._default_pause_rpc
        self._wall_clock = wall_clock
        self.rpc_timeout_sec = float(rpc_timeout_sec)
        self._intervals: list[dict[str, float]] = []
        self._pause_start: float | None = None
        self._busy = False
        self._broken = False
        self._ledger_complete = True
        self._pause_uncertain = False
        self._state = "running"
        self._timeout: Any = None
        self._timeout_was_disabled = False
        self._original_timeout_when: float | None = None

    def _now(self) -> float:
        value = self._wall_clock()
        if not _finite(value):
            raise RuntimeError("Pause clock returned an invalid wall timestamp")
        return float(value)

    async def _default_pause_rpc(self, pause: bool) -> None:
        import tongsim as ts
        fn = ts.UnaryAPI.pause_scene
        # The SDK decorator can turn a failed RPC into None, which is also the
        # successful void return. Always retain the strict RPC error semantics.
        await getattr(fn, "__wrapped__", fn)(self.runtime.context.conn, pause)

    async def _rpc(self, pause: bool) -> None:
        async with asyncio.timeout(self.rpc_timeout_sec):
            await self._pause_rpc(pause)

    @property
    def paused(self) -> bool:
        return self._pause_start is not None

    @property
    def total_paused_sec(self) -> float:
        return self.paused_duration_at(self._now())

    def paused_duration_at(self, wall_time: float) -> float:
        """Map raw wall timestamps without rewriting their original evidence."""
        if not _finite(wall_time):
            raise ValueError("wall_time must be finite")
        duration = sum(max(0.0, min(float(wall_time), item["end_wall"]) - item["start_wall"])
                       for item in self._intervals)
        if self._pause_start is not None:
            duration += max(0.0, float(wall_time) - self._pause_start)
        return duration

    def snapshot(self, at_wall: float | None = None) -> dict[str, Any]:
        now = self._now() if at_wall is None else float(at_wall)
        if not _finite(now):
            raise ValueError("at_wall must be finite")
        total = self.paused_duration_at(now)
        started = getattr(self.runtime, "_episode_started_at", None)
        elapsed = max(0.0, now - started - total) if _finite(started) else 0.0
        limit = getattr(self.runtime, "_time_limit_sec", None)
        if not _finite(limit):
            deadline = getattr(self.runtime, "_episode_deadline", None)
            closed_duration = sum(item["duration_sec"] for item in self._intervals)
            limit = deadline - started - closed_duration if _finite(deadline) and _finite(started) else None
        intervals: list[dict[str, Any]] = [dict(item) for item in self._intervals]
        if self._pause_start is not None:
            intervals.append({"start_wall": self._pause_start, "end_wall": None,
                              "duration_sec": max(0.0, now - self._pause_start)})
        return {"clock": self.clock, "paused": self.paused, "pause_state": self._state,
                "ledger_complete": self._ledger_complete,
                "total_paused_sec": total, "elapsed_sec": elapsed,
                "remaining_sec": max(0.0, float(limit) - elapsed) if _finite(limit) else None,
                "pause_intervals": intervals}

    def _monitor(self) -> Any:
        return getattr(getattr(self.runtime, "context", None), "_emas_monitor", None)

    async def _notify(self) -> None:
        callback = getattr(self._monitor(), "on_pause_changed", None)
        if callable(callback):
            result = callback(self)
            if inspect.isawaitable(result):
                async with asyncio.timeout(self.rpc_timeout_sec):
                    await result

    def _disable_timeout(self) -> None:
        self._timeout = getattr(self.runtime, "_active_decision_timeout", None)
        self._timeout_was_disabled = False
        self._original_timeout_when = None
        if self._timeout is None:
            return
        expired = getattr(self._timeout, "expired", None)
        if callable(expired) and expired():
            raise TimeoutError("Episode timeout already expired before model pause")
        when = getattr(self._timeout, "when", None)
        self._original_timeout_when = when() if callable(when) else None
        self._timeout.reschedule(None)
        self._timeout_was_disabled = True

    def _restore_timeout(self) -> None:
        if not self._timeout_was_disabled:
            return
        self._timeout_was_disabled = False
        expired = getattr(self._timeout, "expired", None)
        if callable(expired) and expired():
            return
        deadline = getattr(self.runtime, "_episode_deadline", None)
        target = (asyncio.get_running_loop().time() + max(0.0, float(deadline) - self._now())
                  if _finite(deadline) else self._original_timeout_when)
        self._timeout.reschedule(target)

    def _guard_entry(self) -> None:
        if self._broken:
            raise RuntimeError("Pause controller failed; the episode cannot continue")
        if self._busy:
            raise RuntimeError("T1 model pauses must not overlap")
        started = getattr(self.runtime, "_episode_started_at", None)
        deadline = getattr(self.runtime, "_episode_deadline", None)
        if _finite(started) and _finite(deadline) and self._now() >= deadline:
            raise T1PauseBudgetExhausted("T1 task budget exhausted before model pause")
        control = getattr(self.runtime, "control", None)
        if control is None:
            control = getattr(self.runtime, "_control", None)
        running = getattr(control, "is_running", None)
        if callable(running) and running():
            raise RuntimeError("Cannot pause a T1 model call while an action is running")

    async def _finish(self, *, resume_required: bool) -> None:
        """Restore the world, persist its state, and restore the episode timeout."""
        errors: list[BaseException] = []
        try:
            if resume_required:
                self._state = "resume_requested"
                await self._rpc(False)
                ended = self._now()
                if self._pause_start is not None:
                    if ended < self._pause_start:
                        raise RuntimeError("Wall clock moved backwards during pause")
                    duration = ended - self._pause_start
                    self._intervals.append({"start_wall": self._pause_start,
                                            "end_wall": ended, "duration_sec": duration})
                    self._pause_start = None
                    for owner, name in ((self.runtime, "_episode_deadline"), (self._monitor(), "deadline")):
                        deadline = getattr(owner, name, None)
                        if _finite(deadline):
                            setattr(owner, name, float(deadline) + duration)
            self._state = "pause_uncertain" if self._pause_uncertain else "running"
        except BaseException as exc:
            self._broken = True
            self._state = ("resume_failed" if self._pause_start is not None else
                           "pause_uncertain" if self._pause_uncertain else "failed")
            errors.append(exc)
        try:
            # Persist the failed/open state too; observers must not assume that
            # cleanup acknowledged a resume when its RPC actually failed.
            await self._notify()
        except BaseException as exc:
            self._broken = True
            self._state = ("resume_failed" if self._pause_start is not None else
                           "pause_uncertain" if self._pause_uncertain else "failed")
            errors.append(exc)
        try:
            self._restore_timeout()
        except BaseException as exc:
            self._broken = True
            self._state = ("resume_failed" if self._pause_start is not None else
                           "pause_uncertain" if self._pause_uncertain else "failed")
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("T1 pause cleanup failed", errors)

    async def _finish_shielded(self, *, resume_required: bool) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        task = asyncio.create_task(self._finish(resume_required=resume_required), name="t1-pause-resume")
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(task)
                return None, cancellation
            except asyncio.CancelledError as exc:
                if task.done():
                    if task.cancelled():
                        return exc, cancellation
                    cancellation = exc
                    try:
                        task.result()
                    except BaseException as error:
                        return error, cancellation
                    return None, cancellation
                cancellation = exc
            except BaseException as exc:
                return exc, cancellation

    @asynccontextmanager
    async def before_model(self):
        self._guard_entry()
        self._busy = True
        resume_required = False
        error: BaseException | None = None
        budget_exhausted = False
        try:
            self._disable_timeout()
            resume_required = True  # A timed-out pause RPC may already have paused UE.
            self._state = "pause_requested"
            await self._rpc(True)
            self._pause_start = self._now()
            self._state = "paused"
            await self._notify()
            remaining = self.snapshot()["remaining_sec"]
            if remaining is not None and remaining <= 0:
                budget_exhausted = True
                raise T1PauseBudgetExhausted("T1 task budget exhausted while confirming pause")
            yield self
        except BaseException as exc:
            error = exc
            if resume_required and self._pause_start is None:
                # The RPC may have paused UE before losing its acknowledgment.
                # A later successful resume cannot reconstruct the missing start.
                self._ledger_complete = False
                self._pause_uncertain = True
                self._broken = True
                self._state = "pause_uncertain"
        finally:
            cleanup_error, cancellation = await self._finish_shielded(resume_required=resume_required)
            self._busy = False
        errors = [item for item in (error, cleanup_error) if item is not None]
        expired = getattr(self._timeout, "expired", None)
        # Rescheduling an exhausted episode deadline cancels this task while
        # shielded cleanup returns. That is the same budget error, not a second
        # failure; avoid turning the normal timeout into an exception group.
        derived_budget_cancel = (budget_exhausted and cleanup_error is None
                                 and callable(expired) and expired())
        if (cancellation is not None and not derived_budget_cancel
                and not any(isinstance(item, asyncio.CancelledError) for item in errors)):
            errors.append(cancellation)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("T1 model call and pause cleanup failed", errors)
