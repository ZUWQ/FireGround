"""Episode-local transport for explicit teammate messages, with retry receipts."""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any

from EmbodiedMAS.agent.runtime.messages import normalize_message


class MessageDeliveryError(RuntimeError):
    """At least one recipient did not acknowledge a broadcast."""

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = dict(failures)
        super().__init__("Teammate delivery failed: " + ", ".join(failures))


@dataclass
class _Delivery:
    message: dict[str, Any]
    recipients: tuple[str, ...]
    acknowledged: set[str] = field(default_factory=set)


class EpisodeMessageBus:
    """Register deployed runtimes before starting their shared episode.

    A fresh instance owns each episode; it never reads agent observations or
    memory. Only explicit messages pass through on_other_agent_info, which owns
    priority/interrupt scheduling. Confirmed recipients are skipped on retry.
    """

    def __init__(self, *, on_event: Any = None) -> None:
        self._on_event = on_event
        self._members: dict[str, Any] = {}
        self._callbacks: dict[str, Any] = {}
        self._deliveries: dict[tuple[str, str], _Delivery] = {}
        self._delivery_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    def _record(self, event: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event({"event": event, "monotonic_sec": time.monotonic(), **copy.deepcopy(fields)})

    @staticmethod
    def _episode_started(runtime: Any) -> bool:
        return (bool(getattr(runtime, "_run_active", False))
                or getattr(runtime, "_episode_started_at", None) is not None
                or getattr(runtime, "end_reason", None) is not None
                or bool(getattr(runtime, "_normal_run_completed", False))
                or bool(getattr(runtime, "_closed", False)))

    def register(self, runtime: Any) -> None:
        if self._closed:
            raise RuntimeError("Message bus is closed")
        for name, existing in self._members.items():
            if existing is runtime:
                if getattr(runtime.agent, "agent_name", None) != name:
                    raise ValueError("Registered agent name changed")
                return
        if (self._started or self._episode_started(runtime)
                or any(self._episode_started(member) for member in self._members.values())):
            raise RuntimeError("Register all agents before the episode starts")
        cooperation = str(getattr(runtime, "_cooperation_type", "none")).strip().upper()
        if cooperation not in {"D", "C", "H"}:
            raise ValueError("Message bus requires D, C or H coordination")
        body = getattr(runtime, "agent", None)
        if not getattr(body, "is_spawned", False):
            raise ValueError("Deploy the agent before registering its runtime")
        name = getattr(body, "agent_name", None)
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("A deployed agent requires a non-empty name")
        if name in self._members:
            raise ValueError(f"Agent name is already registered: {name}")
        owner = getattr(runtime, "_emas_message_bus", None)
        if owner is not None and owner is not self:
            raise ValueError("Runtime already belongs to another episode message bus")
        if getattr(runtime, "_message_sender", None) is not None:
            raise ValueError("Runtime already has another message sender")
        if not callable(getattr(runtime, "on_other_agent_info", None)):
            raise ValueError("Runtime must expose on_other_agent_info")

        async def send(message: dict[str, Any]) -> None:
            await self._broadcast(name, message)

        self._members[name] = runtime
        self._callbacks[name] = send
        runtime._emas_message_bus = self
        runtime._message_sender = send

    async def _broadcast(self, sender: str, message: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Message bus is closed")
        if not isinstance(message, dict):
            raise ValueError("Outgoing messages must be objects")
        identifier = message.get("message_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Outgoing message requires a stable non-empty message_id")
        normalized = normalize_message(message, sender=sender)
        self._started = True
        async with self._delivery_lock:
            if self._closed:
                raise RuntimeError("Message bus is closed")
            runtime = self._members[sender]
            if runtime.agent.agent_name != sender or getattr(runtime, "_closed", False):
                raise RuntimeError("Message sender is unavailable or was renamed")
            key = (sender, identifier)
            delivery = self._deliveries.get(key)
            if delivery is None:
                delivery = _Delivery(copy.deepcopy(normalized),
                                     tuple(name for name in self._members if name != sender))
                self._deliveries[key] = delivery
                self._record("submitted", sender=sender, message_id=identifier, message=normalized)
            elif delivery.message != normalized:
                raise ValueError("A message_id cannot be reused with different content")
            failures = {}
            for recipient in delivery.recipients:
                if recipient in delivery.acknowledged:
                    continue
                receiver = self._members[recipient]
                try:
                    if getattr(receiver, "_closed", False):
                        raise RuntimeError("Recipient is closed")
                    await receiver.on_other_agent_info(copy.deepcopy(delivery.message), sender=sender)
                except Exception as exc:
                    failures[recipient] = type(exc).__name__
                    self._record("delivery_failed", sender=sender, recipient=recipient,
                                 message_id=identifier, error_type=type(exc).__name__)
                else:
                    delivery.acknowledged.add(recipient)
                    self._record("delivered", sender=sender, recipient=recipient, message_id=identifier)
            if failures:
                raise MessageDeliveryError(failures)

    async def close(self) -> None:
        """Detach owned callbacks after in-flight deliveries finish."""
        async with self._delivery_lock:
            self._closed = True
            for name, runtime in self._members.items():
                if getattr(runtime, "_message_sender", None) is self._callbacks[name]:
                    runtime._message_sender = None
                if getattr(runtime, "_emas_message_bus", None) is self:
                    del runtime._emas_message_bus
            self._members.clear()
            self._callbacks.clear()
            self._deliveries.clear()
