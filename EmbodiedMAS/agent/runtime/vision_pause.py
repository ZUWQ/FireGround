"""Episode-owned vision grace pauses and an auditable active-wall clock.

Only a runner that explicitly opts in installs this controller. UE/network
control operations retain real-time limits; business time excludes the union
of acknowledged global pause intervals. No model or UE connection is created.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class VisionPauseBudgetExhausted(TimeoutError):
    """No new decision/action may be dispatched after the active deadline."""


@dataclass
class _Decision:
    owner: asyncio.Task
    active_start: float
    done: bool = False
    watcher: asyncio.Task | None = None
    error: BaseException | None = None


class VisionPauseController:
    clock = "active_wall_seconds"
    protocol = "vision_grace_pause_v1"

    def __init__(self, context: Any, *, grace_sec: float = 20.0,
                 pause_rpc: Callable[[bool], Awaitable[None]] | None = None,
                 wall_clock: Callable[[], float] = time.monotonic,
                 rpc_timeout_sec: float = 5.0) -> None:
        for name, value in (("grace_sec", grace_sec), ("rpc_timeout_sec", rpc_timeout_sec)):
            if not _finite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        self.context = context
        self.grace_sec, self.rpc_timeout_sec = float(grace_sec), float(rpc_timeout_sec)
        self._pause_rpc = pause_rpc or self._default_pause_rpc
        self._wall_clock = wall_clock
        self._started_at: float | None = None
        self._duration: float | None = None
        self._pause_start: float | None = None
        self._intervals: list[dict[str, float]] = []
        self._holders: set[str] = set()
        self._decisions: dict[str, _Decision] = {}
        self._seen_ids: set[str] = set()
        self._runtimes: list[Any] = []
        self._timeouts: dict[asyncio.Timeout, float] = {}
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._state = "running"
        self._ledger_complete = True
        self._failure: BaseException | None = None
        self._closed = False
        self._close_task: asyncio.Task | None = None
        self._resume_required = False

    @property
    def rpc_wall_timeout_sec(self) -> float:
        # A navigation transport can span many sequential global pauses.
        # Keep its real-time insurance above the whole valid episode bound;
        # timeout_active separately enforces the action's business budget.
        if self._duration is None:
            return 3600.0
        from .vision_protocol import derive_wall_timeout
        return max(3600.0, derive_wall_timeout(
            self._duration, max(1, len(self._runtimes)), grace_sec=self.grace_sec,
            pause_rpc_timeout_sec=self.rpc_timeout_sec))

    def _now(self) -> float:
        now = self._wall_clock()
        if not _finite(now):
            raise RuntimeError("Vision pause wall clock is invalid")
        return float(now)

    def start(self, started_at: float, duration_sec: float) -> None:
        if not _finite(started_at) or not _finite(duration_sec) or duration_sec <= 0:
            raise ValueError("Vision pause start and duration must be finite; duration must be positive")
        if self._closed:
            raise RuntimeError("Vision pause episode is closed")
        if self._started_at is not None:
            if self._started_at == float(started_at) and self._duration == float(duration_sec):
                return
            raise RuntimeError("Vision pause episode has different existing clock settings")
        self._started_at, self._duration = float(started_at), float(duration_sec)
        self._sync_deadlines()
        self._signal()

    def register_runtime(self, runtime: Any) -> None:
        if all(item is not runtime for item in self._runtimes):
            self._runtimes.append(runtime)
        self._sync_deadlines()

    @property
    def started(self) -> bool:
        return self._started_at is not None

    @property
    def paused(self) -> bool:
        return self._pause_start is not None

    @property
    def total_paused_sec(self) -> float:
        return self.paused_duration_at(self._now())

    def paused_duration_at(self, wall_time: float) -> float:
        if not _finite(wall_time):
            raise ValueError("wall_time must be finite")
        total = sum(max(0.0, min(wall_time, i["end_wall"]) - i["start_wall"])
                    for i in self._intervals)
        if self._pause_start is not None:
            total += max(0.0, wall_time - self._pause_start)
        return total

    def active_now(self) -> float:
        now = self._now()
        return now - self.paused_duration_at(now)

    def snapshot(self, at_wall: float | None = None) -> dict[str, Any]:
        now = self._now() if at_wall is None else at_wall
        if not _finite(now):
            raise ValueError("at_wall must be finite")
        total = self.paused_duration_at(now)
        elapsed = max(0.0, now - total - self._started_at) if self._started_at is not None else 0.0
        intervals = [dict(i) for i in self._intervals]
        if self._pause_start is not None:
            intervals.append(dict(start_wall=self._pause_start, end_wall=None,
                                  duration_sec=max(0.0, now - self._pause_start)))
        deadline = (self._started_at + self._duration + total
                    if self._started_at is not None else None)
        return dict(clock=self.clock, protocol=self.protocol, grace_sec=self.grace_sec,
                    paused=self.paused, pause_state=self._state,
                    ledger_complete=self._ledger_complete and self._failure is None,
                    total_paused_sec=total, elapsed_sec=elapsed, limit_sec=self._duration,
                    remaining_sec=max(0.0, self._duration - elapsed) if self._duration is not None else None,
                    pause_intervals=intervals, pause_holders=sorted(self._holders),
                    pending_decisions=len(self._decisions), deadline_monotonic=deadline,
                    closed=self._closed)

    def _sync_deadlines(self) -> None:
        if self._started_at is None:
            return
        deadline = self._started_at + self._duration + self.total_paused_sec
        for runtime in self._runtimes:
            runtime._episode_deadline = deadline
        monitor = getattr(self.context, "_emas_monitor", None)
        if monitor is not None:
            monitor.deadline = deadline

    def _signal(self) -> None:
        old, self._changed = self._changed, asyncio.Event()
        old.set()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for timeout, active_deadline in tuple(self._timeouts.items()):
            if not timeout.expired():
                timeout.reschedule(None if self.paused or self._state != "running" else
                                   loop.time() + max(0.0, active_deadline - self.active_now()))

    async def _notify(self, event: str, **fields: Any) -> None:
        self._sync_deadlines()
        self._signal()
        monitor = getattr(self.context, "_emas_monitor", None)
        callback = getattr(monitor, "on_pause_changed", None)
        if callable(callback):
            result = callback(self)
            if inspect.isawaitable(result):
                async with asyncio.timeout(self.rpc_timeout_sec):
                    await result
        record = getattr(monitor, "record", None)
        if callable(record):
            record("lifecycle", dict(event=event, **fields, **self.snapshot()))

    def _check(self, *, budget: bool = True) -> None:
        if self._failure is not None:
            raise RuntimeError("Vision shared pause controller failed") from self._failure
        if self._closed:
            raise RuntimeError("Vision shared pause controller is closed")
        if self._started_at is None:
            raise RuntimeError("Vision pause episode has not started")
        if budget and self.snapshot()["remaining_sec"] <= 0:
            raise VisionPauseBudgetExhausted("Vision episode active-time budget exhausted")

    async def _default_pause_rpc(self, pause: bool) -> None:
        import tongsim as ts
        fn = ts.UnaryAPI.pause_scene
        await getattr(fn, "__wrapped__", fn)(self.context.conn, pause)

    async def _rpc(self, pause: bool) -> None:
        async with asyncio.timeout(self.rpc_timeout_sec):
            await self._pause_rpc(pause)

    def _break(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        self._ledger_complete = False
        for decision in tuple(self._decisions.values()):
            if not decision.done and not decision.owner.done():
                decision.owner.cancel()
        self._signal()

    async def wait_running(self) -> None:
        while True:
            changed = self._changed
            self._check()
            if self._state == "running" and not self.paused and not self._holders:
                return
            await changed.wait()

    async def wait_resumed(self) -> None:
        """Allow owned cleanup after the task deadline, but never during pause.

        Normal close may already have completed; unlike new work, camera/body
        cleanup may still need to restore a pose after the episode budget.
        """
        while True:
            changed = self._changed
            if self._failure is not None:
                raise RuntimeError("Vision shared pause controller failed") from self._failure
            if self._state == "running" and not self.paused and not self._resume_required:
                return
            await changed.wait()

    async def _wait_until_active(self, deadline: float, *, budget: bool = False) -> None:
        while True:
            changed = self._changed
            self._check(budget=budget)
            remaining = deadline - self.active_now()
            if remaining <= 0:
                return
            wait = None if self.paused or self._state != "running" else remaining
            try:
                async with asyncio.timeout(wait):
                    await changed.wait()
            except TimeoutError:
                pass

    async def sleep_active(self, seconds: float) -> None:
        if not _finite(seconds) or seconds < 0:
            raise ValueError("Active sleep must be finite and nonnegative")
        self._check()
        await self._wait_until_active(self.active_now() + seconds)
        await self.wait_running()

    @asynccontextmanager
    async def timeout_active(self, seconds: float):
        if not _finite(seconds) or seconds < 0:
            raise ValueError("Active timeout must be finite and nonnegative")
        self._check()
        deadline = self.active_now() + seconds
        timeout = asyncio.timeout(None)
        async with timeout:
            self._timeouts[timeout] = deadline
            self._signal()
            try:
                yield timeout
            finally:
                self._timeouts.pop(timeout, None)

    async def _acquire(self, logical_id: str, decision: _Decision) -> None:
        async with self._lock:
            if decision.done or self._decisions.get(logical_id) is not decision:
                return
            self._check()  # The 300-second deadline wins over the grace timer.
            self._holders.add(logical_id)
            if self.paused:
                await self._notify("vision_pause_holder_added", logical_decision_id=logical_id)
                return
            self._state, self._resume_required = "pause_requested", True
            await self._notify("vision_pause_requested", logical_decision_id=logical_id)
            try:
                await self._rpc(True)
                self._pause_start = self._now()
                self._state = "paused"
                await self._notify("vision_pause_confirmed", logical_decision_id=logical_id,
                                   grace_overshoot_sec=max(0.0, self.active_now() - decision.active_start - self.grace_sec))
                if self.snapshot()["remaining_sec"] <= 0:
                    raise VisionPauseBudgetExhausted("Vision active budget expired while confirming pause")
            except BaseException:
                if self._pause_start is None:
                    self._state = "pause_uncertain"
                    self._ledger_complete = False
                raise

    async def _watch(self, logical_id: str, decision: _Decision) -> None:
        try:
            end = min(decision.active_start + self.grace_sec,
                      self._started_at + self._duration)
            while not decision.done:
                changed = self._changed
                self._check()
                remaining = end - self.active_now()
                if remaining <= 0:
                    await self._acquire(logical_id, decision)
                    return
                wait = None if self.paused or self._state != "running" else remaining
                try:
                    async with asyncio.timeout(wait):
                        await changed.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            decision.error = error
            if not isinstance(error, VisionPauseBudgetExhausted):
                self._break(error)
            if not decision.done and not decision.owner.done():
                decision.owner.cancel()

    @staticmethod
    def _raise_errors(message: str, errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(message, errors)

    async def _resume_locked(self) -> None:
        if not self._resume_required:
            return
        errors: list[BaseException] = []
        self._state = "resume_requested"
        try:
            await self._notify("vision_resume_requested")
        except BaseException as error:
            self._break(error)
            errors.append(error)
        # Logging failure must never skip the best-effort physical resume.
        try:
            await self._rpc(False)
            ended = self._now()
            if self._pause_start is not None:
                if ended < self._pause_start:
                    raise RuntimeError("Vision pause wall clock moved backwards")
                self._intervals.append(dict(start_wall=self._pause_start, end_wall=ended,
                                            duration_sec=ended-self._pause_start))
                self._pause_start = None
            self._resume_required = False
            self._state = "running" if self._failure is None else "failed"
        except BaseException as error:
            self._state = "resume_failed" if self.paused else "pause_uncertain"
            self._break(error)
            errors.append(error)
        try:
            await self._notify("vision_resume_confirmed" if not self._resume_required else
                               "vision_resume_failed")
        except BaseException as error:
            self._break(error)
            errors.append(error)
        self._raise_errors("Vision resume and audit failed", errors)

    async def _end(self, logical_id: str, decision: _Decision) -> None:
        errors: list[BaseException] = []
        async with self._lock:
            self._holders.discard(logical_id)
            self._decisions.pop(logical_id, None)
            try:
                await self._notify("vision_pause_holder_released", logical_decision_id=logical_id)
            except BaseException as error:
                self._break(error)
                errors.append(error)
            if not self._holders:
                try:
                    await self._resume_locked()
                except BaseException as error:
                    errors.append(error)
        # A watcher inside a bounded pause RPC must not be cancelled midway:
        # _lock ensured its acknowledgement was settled before release above.
        if decision.watcher is not None:
            try:
                await decision.watcher
            except BaseException as error:
                errors.append(error)
        self._raise_errors("Vision decision cleanup failed", errors)

    async def _shield_cleanup(self, coroutine) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        task = asyncio.create_task(coroutine)
        cancelled = None
        while True:
            try:
                await asyncio.shield(task)
                return None, cancelled
            except asyncio.CancelledError as error:
                if task.done():
                    try:
                        task.result()
                    except BaseException as cleanup_error:
                        return cleanup_error, cancelled
                    return None, error
                cancelled = error
            except BaseException as error:
                return error, cancelled

    @asynccontextmanager
    async def model_wait(self, logical_id: str):
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("A nonempty logical decision ID is required")
        await self.wait_running()
        if logical_id in self._seen_ids:
            raise ValueError("Logical decision IDs must not be reused within an episode")
        self._seen_ids.add(logical_id)
        decision = _Decision(asyncio.current_task(), self.active_now())
        self._decisions[logical_id] = decision
        decision.watcher = asyncio.create_task(self._watch(logical_id, decision),
                                               name="vision-grace-" + logical_id)
        error: BaseException | None = None
        try:
            yield self
        except BaseException as caught:
            error = caught
        finally:
            decision.done = True
            self._signal()
            cleanup_error, extra_cancel = await self._shield_cleanup(self._end(logical_id, decision))
        if decision.error is not None and (error is None or isinstance(error, asyncio.CancelledError)):
            error = decision.error
        if cleanup_error is not None:
            if error is None or isinstance(error, asyncio.CancelledError):
                raise cleanup_error
            raise BaseExceptionGroup("Vision model and pause cleanup failed", [error, cleanup_error])
        if error is not None:
            raise error
        if extra_cancel is not None:
            raise extra_cancel
        await self.wait_running()  # Release before waiting: no all-ready deadlock.

    async def _close(self) -> None:
        self._closed = True
        self._signal()
        for decision in tuple(self._decisions.values()):
            decision.done = True
            if not decision.owner.done() and decision.owner is not asyncio.current_task():
                decision.owner.cancel()
        async with self._lock:
            self._holders.clear()
            await self._resume_locked()
        self._signal()

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="vision-pause-close")
        error, cancelled = await self._shield_cleanup(self._close_task_wait())
        if error is not None:
            raise error
        if cancelled is not None:
            raise cancelled

    async def _close_task_wait(self) -> None:
        await self._close_task
