from __future__ import annotations

import asyncio
import copy
import math
import time
from uuid import UUID
from typing import Any, Awaitable, Callable, Optional

from tongsim.core.world_context import WorldContext

from EmbodiedMAS.environment.perception_api import PerceptionInfo
from EmbodiedMAS.agent.spatial_units import finite_xy

OnSOSCallback = Callable[[dict[str, Any]], Awaitable[None]]
OnSOSEvent = Callable[[str, dict[str, Any]], None]


def build_sos_trigger_context(
    sos: dict[str, Any],
    estimated_location: Any = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {"sos": sos}
    if estimated_location is not None:
        if hasattr(estimated_location, "x"):
            context["estimated_location"] = {
                "x": float(estimated_location.x),
                "y": float(estimated_location.y),
                "z": float(estimated_location.z),
            }
        else:
            context["estimated_location"] = estimated_location
    return context


def public_sos_context(context: Any, names: Any = None) -> dict[str, Any]:
    """Expose signal semantics, never raw bearings, ranges or inferred coordinates.

    The engine SOS packet identifies the listening agent and carries directional
    samples, not a verified civilian identity. Do not invent a source or room.
    Explicit semantic fields from richer sensors may pass through. The optional
    names argument keeps a common projection-call signature; IDs are not used.
    """
    if not isinstance(context, dict):
        return {}
    result: dict[str, Any] = {
        key: context[key] for key in ("message", "source", "reason", "trigger")
        if isinstance(context.get(key), str)
    }
    sos = context.get("sos")
    if isinstance(sos, dict):
        signal: dict[str, Any] = {"received": True}
        for key in ("room", "name", "target_ref", "message", "direction"):
            value = sos.get(key)
            if not isinstance(value, str):
                continue
            if key in {"name", "target_ref"}:
                try:
                    UUID(value)
                except ValueError:
                    pass
                else:
                    continue
            if key == "direction" and value not in {
                    "forward", "left", "right", "backward", "front_left", "front_right", "unknown"}:
                continue
            signal[key] = value
        for key in ("captured_at", "received_at"):
            value = sos.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                signal[key] = value
        result["sos"] = signal
    teammate = context.get("teammate_message")
    if isinstance(teammate, dict):
        result["teammate_message"] = {
            key: teammate[key] for key in ("sender", "type", "priority", "message", "content")
            if isinstance(teammate.get(key), str)
        }
    if isinstance(context.get("events"), (list, tuple)):
        events = []
        for event in context["events"]:
            if not isinstance(event, dict):
                continue
            public_event: dict[str, Any] = {
                key: event[key] for key in ("message", "source", "reason")
                if isinstance(event.get(key), str)
            }
            if isinstance(event.get("trigger"), str):
                public_event["trigger"] = event["trigger"]
            if isinstance(event.get("context"), dict):
                public_event["context"] = public_sos_context(event["context"], names)
            if isinstance(event.get("other_agent_info"), str):
                public_event["other_agent_info"] = event["other_agent_info"]
            events.append(public_event)
        result["events"] = events
    return result


def _receiver_key(value: Any) -> str | None:
    """Canonical listening-agent identity; never interpret it as an NPC ID."""
    if isinstance(value, bytes):
        return str(UUID(bytes_le=value)) if len(value) == 16 and any(value) else None
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        guid = UUID(value)
        return str(guid) if guid.int else None
    except ValueError:
        return value  # Custom/offline receivers may have opaque identities.


class SOSHub:
    """Own UE's single SOS stream for all receivers in one WorldContext/loop.

    Membership changes drain the old RPC before opening the new union. Only
    queue insertion happens in the reader; each listener owns its callback task.
    A context must be shared by agents connected to the same UE instance.
    """

    def __init__(self, context: WorldContext) -> None:
        self._perception = PerceptionInfo(context)
        self._perception._sos_event_callback = self._transport_event
        self._packet_sequence = 0
        self._transport_audit_listeners: dict[str, tuple[SOSListener, ...]] = {}
        self._transport_stats = dict(raw_responses=0, empty_responses=0, raw_signals=0,
                                     routed_signals=0, unmatched_signals=0, invalid_packets=0)
        self._listeners: dict[str, set[SOSListener]] = {}
        self._changed = asyncio.Event()
        self._revision = 0
        self._applied_revision = 0
        self._applied = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def register(self, listener: SOSListener) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("SOSHub must be used on its WorldContext event loop")
        self._loop = loop
        self._listeners.setdefault(listener._receiver, set()).add(listener)
        self._revision += 1
        self._changed.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._supervise(), name="emas-sos-hub")

    def unregister(self, listener: SOSListener) -> int:
        listeners = self._listeners.get(listener._receiver)
        if listeners is not None:
            listeners.discard(listener)
            if not listeners:
                del self._listeners[listener._receiver]
        self._revision += 1
        self._changed.set()
        return self._revision

    def _mark_applied(self) -> None:
        self._applied_revision = self._revision
        self._applied.set()
        self._applied = asyncio.Event()

    async def wait_reconfigured(self, revision: int) -> None:
        while self._applied_revision < revision:
            event = self._applied
            await event.wait()

    def _notify(self, event: str, **details: Any) -> None:
        for listeners in tuple(self._listeners.values()):
            for listener in tuple(listeners):
                listener._notify(event, **details)

    @property
    def transport_stats(self) -> dict[str, int]:
        """Context-wide transport counts, not per-receiver delivery counts."""
        return dict(self._transport_stats)

    def _transport_event(self, event: str, details: dict[str, Any]) -> None:
        if event == "raw_response":
            self._transport_stats["raw_responses"] += 1
            self._transport_stats["empty_responses"] += int(details.get("empty_response", False))
        elif event == "raw_signal":
            self._transport_stats["raw_signals"] += 1
        active = {listener for group in self._listeners.values() for listener in group}
        stream_id = details.get("sos_stream_id")
        if event == "rpc_created" and isinstance(stream_id, str):
            self._transport_audit_listeners[stream_id] = tuple(active)
        recipients = set(active)
        if event in {"rpc_cancelled", "rpc_error", "rpc_closed", "rpc_close_timeout"}:
            # unregister() removes packet recipients before draining the RPC.
            # Keep audit-only recipients until that stream's terminal event so
            # the final receiver can still record cancellation and closure.
            recipients.update(self._transport_audit_listeners.get(stream_id, ()))
        for listener in recipients:
            listener._notify(event, **details, transport_counters=self.transport_stats)
        if event in {"rpc_closed", "rpc_close_timeout"}:
            self._transport_audit_listeners.pop(stream_id, None)

    async def _read(self, agent_ids: list[bytes | str]) -> None:
        stream = self._perception.receive_sos(agent_ids)
        try:
            async for sos in stream:
                self._packet_sequence += 1
                if not isinstance(sos, dict):
                    self._transport_stats["invalid_packets"] += 1
                    self._notify("packet_rejected", reason="not_an_object",
                                 transport_counters=self.transport_stats)
                    continue
                diagnostics = getattr(self._perception, "sos_diagnostics_for", None)
                trace = diagnostics(sos) if callable(diagnostics) else {}
                trace = trace or {"signal_id": f"hub:{id(self):x}:{self._packet_sequence}",
                                  "trace_source": "hub_only"}
                receiver = _receiver_key(sos.get("agent_id"))
                listeners = tuple(self._listeners.get(receiver, ()))
                if not listeners:
                    self._transport_stats["unmatched_signals"] += 1
                    self._notify("unmatched_receiver", **trace, receiver=receiver,
                                 reason="invalid_receiver" if receiver is None else "receiver_not_registered",
                                 transport_counters=self.transport_stats)
                    continue
                self._transport_stats["routed_signals"] += 1
                self._notify("routed", **trace, receiver=receiver, listener_count=len(listeners),
                             transport_counters=self.transport_stats)
                for listener in listeners:
                    listener._offer(sos, trace=trace)
        finally:
            await stream.aclose()

    async def _supervise(self) -> None:
        reader = changed = None
        try:
            while True:
                self._changed.clear()
                # Any previous reader has been fully drained at this point.
                self._mark_applied()
                if not self._listeners:
                    return
                receivers = [next(iter(group)) for group in self._listeners.values()]
                ids = [listener._agent_guid for listener in receivers]
                delay = min(listener._retry_delay_sec for listener in receivers)
                self._notify("subscription_attempt", receiver_count=len(ids),
                             receivers=[listener._receiver for listener in receivers], generation=self._revision)
                reader = asyncio.create_task(self._read(ids), name="emas-sos-stream")
                changed = asyncio.create_task(self._changed.wait())
                try:
                    done, _ = await asyncio.wait({reader, changed}, return_when=asyncio.FIRST_COMPLETED)
                    if changed not in done:
                        try:
                            await reader
                        except Exception as exc:
                            self._notify("reconnecting", reason=type(exc).__name__, detail=str(exc))
                        else:
                            self._notify("reconnecting", reason="stream_ended")
                        # Membership changes wake a failed stream's retry delay.
                        try:
                            await asyncio.wait_for(self._changed.wait(), delay)
                        except TimeoutError:
                            pass
                finally:
                    for task in (reader, changed):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(reader, changed, return_exceptions=True)
                    reader = changed = None
        finally:
            for task in (reader, changed):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (reader, changed) if task is not None),
                                 return_exceptions=True)
            self._mark_applied()
            self._task = None


def sos_hub_for_context(context: WorldContext) -> SOSHub:
    hub = getattr(context, "_emas_sos_hub", None)
    if hub is None:
        hub = SOSHub(context)
        context._emas_sos_hub = hub
    return hub


class SOSLocationMemory:
    """Remember acoustic XY estimates for one deployed body, in meters."""

    radius_m = 0.5

    def __init__(self) -> None:
        self._locations: list[tuple[float, float]] = []

    def clear(self) -> None:
        self._locations.clear()

    def remember(self, location_m: Any) -> bool:
        """Return True only when outside every recorded point's 0.5 m radius."""
        xy = finite_xy(location_m)
        if xy is None:
            raise ValueError("SOS location must contain finite XY in meters")
        if any(math.dist(xy, previous) <= self.radius_m + 1e-9 for previous in self._locations):
            return False
        # Keep the first anchor: repeated noisy samples cannot move it gradually.
        self._locations.append(tuple(xy))
        return True


class SOSListener:
    """A receiver's lifetime subscription and queue; runtime filters locations.

    A running task proves local listening only. subscription_attempt is not a
    server acknowledgment; received is evidence of an actual matching packet.
    """

    def __init__(
        self,
        context: WorldContext,
        agent_guid: bytes | str,
        on_sos: OnSOSCallback,
        *,
        retry_delay_sec: float = 1.0,
        on_event: OnSOSEvent | None = None,
    ) -> None:
        self._hub = sos_hub_for_context(context)
        self._agent_guid = agent_guid
        self._receiver = _receiver_key(agent_guid)
        if self._receiver is None:
            raise ValueError("SOS listener requires a valid receiver identity")
        self._on_sos = on_sos
        self._on_event = on_event
        self._retry_delay_sec = max(0.01, float(retry_delay_sec))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active_trace: dict[str, Any] = {}
        self._task: Optional[asyncio.Task[None]] = None
        self._stats = dict(received=0, delivered=0, callback_errors=0)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def current_trace(self) -> dict[str, Any]:
        return dict(self._active_trace)

    @property
    def transport_stats(self) -> dict[str, int]:
        return self._hub.transport_stats

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _notify(self, event: str, **details: Any) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event, dict(details, counters=self.stats))
            except Exception as exc:
                # A full/unwritable audit directory must not disable SOS.
                print(f"[SOSListener] Event log unavailable: {exc}")
        elif event in {"reconnecting", "callback_failed"}:
            print(f"[SOSListener] {event}: {details}")

    def start(self) -> None:
        if self.is_running:
            return
        self._queue = asyncio.Queue()
        self._hub.register(self)
        self._task = asyncio.create_task(self._consume(), name=f"emas-sos-{self._receiver}")
        self._notify("started")

    async def stop(self) -> None:
        if self._task is None:
            return
        task = self._task
        # Detach before yielding: a later start() owns its new registration and
        # consumer. This stop must never erase or cancel that replacement.
        self._task = None
        revision = self._hub.unregister(self)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self._hub.wait_reconfigured(revision)
        if self._task is None:
            self._notify("stopped")

    def _offer(self, sos: dict[str, Any], *, trace: dict | None = None) -> None:
        self._stats["received"] += 1
        metadata = dict(trace or {}, queued_at_monotonic=time.monotonic())
        self._queue.put_nowait((copy.deepcopy(sos), metadata))
        self._notify("received", **metadata, pending_count=self._queue.qsize())

    async def _consume(self) -> None:
        while True:
            sos, trace = await self._queue.get()
            self._active_trace = trace
            started = time.monotonic()
            self._notify("callback_enter", **trace, pending_count=self._queue.qsize(),
                         queue_wait_sec=started - trace["queued_at_monotonic"])
            try:
                await self._on_sos(sos)
                self._stats["delivered"] += 1
                self._notify("callback_exit", **trace, status="returned",
                             elapsed_sec=time.monotonic() - started)
            except asyncio.CancelledError:
                self._notify("callback_exit", **trace, status="cancelled",
                             elapsed_sec=time.monotonic() - started)
                raise
            except Exception as exc:
                self._stats["callback_errors"] += 1
                self._notify("callback_failed", **trace, reason=type(exc).__name__, detail=str(exc))
                self._notify("callback_exit", **trace, status="failed",
                             elapsed_sec=time.monotonic() - started)
            finally:
                self._active_trace = {}


__all__ = [
    "SOSHub",
    "SOSListener",
    "SOSLocationMemory",
    "sos_hub_for_context",
    "build_sos_trigger_context",
    "public_sos_context",
]
