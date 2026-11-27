"""Own the lifetime of one asynchronous action and its cancellation waiter."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

ExecuteFn = Callable[[dict[str, Any]], Awaitable[None]]


class ControlModule:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task[str]] = None
        self._cancel_event: Optional[asyncio.Event] = None
        self._done_event = asyncio.Event()
        self._done_event.set()

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def execute(self, decision: dict[str, Any], *, execute_fn: ExecuteFn) -> str:
        self.cancel()
        await self.wait_until_done()
        self._cancel_event = asyncio.Event()
        self._done_event.clear()
        cancel_event = self._cancel_event

        async def run() -> str:
            if cancel_event.is_set():
                return "interrupted"
            action_task = asyncio.create_task(execute_fn(decision))
            cancel_wait = asyncio.create_task(cancel_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {action_task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_wait in done:
                    return "interrupted"
                await action_task
                return "completed"
            finally:
                # This path also runs when the outer execute task is cancelled.
                # Both child tasks are owned here and cannot outlive the action.
                for task in (action_task, cancel_wait):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(action_task, cancel_wait, return_exceptions=True)

        self._task = asyncio.create_task(run())
        try:
            return await self._task
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                raise
            return "interrupted"
        finally:
            self._done_event.set()

    def cancel(self) -> None:
        """Cooperatively signal an interrupt without cancelling the owning loop."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    async def wait_until_done(self) -> None:
        await self._done_event.wait()
