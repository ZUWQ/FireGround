"""In-memory step timing (no file persistence)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimingRecord:
    name: str
    duration_sec: float
    step: Optional[int] = None
    action: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class StepTimingCache:
    """Records timing for the current step and the full session."""

    def __init__(self) -> None:
        self._step_records: list[TimingRecord] = []
        self._session_records: list[TimingRecord] = []

    def clear_step(self) -> None:
        self._step_records.clear()

    @property
    def step_records(self) -> list[TimingRecord]:
        return list(self._step_records)

    @property
    def session_records(self) -> list[TimingRecord]:
        return list(self._session_records)

    def duration(self, name: str, *, step: Optional[int] = None) -> Optional[float]:
        records = self._step_records if step is None else [
            rec for rec in self._session_records if rec.step == step
        ]
        for rec in reversed(records):
            if rec.name == name:
                return rec.duration_sec
        return None

    def _record(self, rec: TimingRecord) -> None:
        self._step_records.append(rec)
        self._session_records.append(rec)


async def track(
    name: str,
    coro_factory: Callable[[], Awaitable[T]],
    cache: StepTimingCache,
    **fields: Any,
) -> T:
    t0 = time.monotonic()
    try:
        return await coro_factory()
    finally:
        cache._record(
            TimingRecord(
                name=name,
                duration_sec=round(time.monotonic() - t0, 6),
                step=fields.pop("step", None),
                action=fields.pop("action", None),
                extra=fields,
            )
        )


async def track_llm_thinking(
    coro_factory: Callable[[], Awaitable[T]],
    cache: StepTimingCache,
    *,
    pause_conn: Any = None,
) -> T:
    if pause_conn is None:
        return await track("llm_thinking", coro_factory, cache)

    import tongsim as ts

    await ts.UnaryAPI.pause_scene(pause_conn, True)
    try:
        return await track("llm_thinking", coro_factory, cache)
    finally:
        await ts.UnaryAPI.pause_scene(pause_conn, False)
