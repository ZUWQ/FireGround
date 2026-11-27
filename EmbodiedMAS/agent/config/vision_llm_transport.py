"""Bounded async HTTP attempts for explicitly enabled vision pause episodes.

No provider is contacted until called. This module never selects a model,
changes Oracle clients, or retries protocol/parse failures.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import inspect
import math
import time
from typing import Any, Callable

ATTEMPT_TIMEOUT_SEC = 180.0
MAX_ATTEMPTS = 2
MAX_RETRY_DELAY_SEC = 180.0
CLIENT_CLOSE_TIMEOUT_SEC = 5.0
_COST_HEADER = "x-litellm-response-cost"


def _cost(headers: Any, *, raw_available: bool) -> dict[str, Any]:
    result = {
        "gateway_response_cost_usd": None,
        "gateway_response_cost_source": _COST_HEADER if raw_available else None,
        "gateway_response_cost_status": "missing_header" if raw_available else "raw_response_unavailable",
    }
    value = headers.get(_COST_HEADER) if headers is not None else None
    if value is None:
        return result
    try:
        amount = float(value)
    except (ValueError, TypeError, OverflowError):
        amount = math.nan
    if math.isfinite(amount) and amount >= 0:
        result.update(gateway_response_cost_usd=amount, gateway_response_cost_status="reported")
    else:
        result["gateway_response_cost_status"] = "invalid_header"
    return result


async def _completion(client: Any, request: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    completions = client.chat.completions
    raw_create = getattr(getattr(completions, "with_raw_response", None), "create", None)
    if not callable(raw_create):
        return await completions.create(**request), _cost(None, raw_available=False)
    raw = await raw_create(**request)
    completion = raw.parse()
    if inspect.isawaitable(completion):
        completion = await completion
    return completion, _cost(getattr(raw, "headers", None), raw_available=True)


def _retryable(exc: BaseException) -> bool:
    from openai import APIConnectionError, APIStatusError
    if isinstance(exc, (TimeoutError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code in {408, 409, 429} or exc.status_code >= 500
    )


def _retry_delay(exc: BaseException) -> float | None:
    """Never retry earlier than Retry-After; reject an unbounded delay."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if value is None:
        return 0.5
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = (when - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.5
    if not math.isfinite(delay) or delay > MAX_RETRY_DELAY_SEC:
        return None
    return max(0.0, delay)


async def _close_client(client: Any) -> None:
    """Finish owned transport cleanup even if its caller is cancelled twice."""
    async def close() -> None:
        async with asyncio.timeout(CLIENT_CLOSE_TIMEOUT_SEC):
            result = client.close()
            if inspect.isawaitable(result):
                await result
    task = asyncio.create_task(close(), name="vision-model-client-close")
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            cancellation = exc
            if task.done():
                # Propagate a cleanup exception rather than lose it.
                task.result()
                break
    if cancellation is not None:
        raise cancellation


async def create_vision_completion(
    client_kwargs: dict[str, Any], request_kwargs: dict[str, Any], *,
    record_attempt: Callable[[dict[str, Any]], None] | None = None,
    timeout_sec: float = ATTEMPT_TIMEOUT_SEC,
    max_attempts: int = MAX_ATTEMPTS,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Make at most two audited attempts with actual async cancellation.

    ``timeout_sec`` is an absolute per-attempt wall budget covering the entire
    response, not merely the SDK's inactivity timeout. Explicit overrides are
    intended for offline tests; production uses the authorized 180 seconds and
    two total attempts. Successful protocol validation belongs to the caller.
    """
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("vision attempt timeout must be positive and finite")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts not in (1, 2):
        raise ValueError("vision model max_attempts must be one or two")
    if client_factory is None:
        from openai import AsyncOpenAI
        client_factory = AsyncOpenAI
    kwargs = dict(client_kwargs)
    kwargs.update(max_retries=0, timeout=float(timeout_sec))
    request = copy.deepcopy(request_kwargs)
    client = client_factory(**kwargs)
    emit = record_attempt or (lambda _: None)
    try:
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            event = {"attempt_index": attempt, "attempt_limit": max_attempts,
                     "attempt_timeout_sec": float(timeout_sec),
                     "attempt_started_at_monotonic": started}
            emit({"event": "http_attempt_started", **event})
            try:
                async with asyncio.timeout(float(timeout_sec)):
                    completion, cost = await _completion(client, copy.deepcopy(request))
            except asyncio.CancelledError:
                emit({"event": "http_attempt_cancelled", **event,
                      "attempt_finished_at_monotonic": time.monotonic(),
                      "usage_known": False})
                raise
            except BaseException as exc:
                # Exception bodies and arbitrary headers may contain private
                # routing details; only these diagnostic fields are allowed.
                retryable = _retryable(exc)
                delay = _retry_delay(exc) if retryable and attempt < max_attempts else None
                skipped = (None if delay is not None else
                           "attempt_limit_reached" if attempt >= max_attempts else
                           "non_retryable_error" if not retryable else
                           "retry_after_exceeds_limit")
                emit({"event": "http_attempt_error", **event,
                      "attempt_finished_at_monotonic": time.monotonic(),
                      "error_type": type(exc).__name__,
                      "http_status": getattr(exc, "status_code", None),
                      "request_id": getattr(exc, "request_id", None),
                      "will_retry": delay is not None, "retry_delay_sec": delay,
                      "retry_skipped_reason": skipped,
                      "usage_known": False})
                if delay is None:
                    raise
                # The failed await has unwound before another attempt begins;
                # no sync executor thread or prior coroutine is left running.
                emit({"event": "http_retry_wait", "attempt_index": attempt,
                      "retry_delay_sec": delay})
                await asyncio.sleep(delay)
            else:
                usage = getattr(completion, "usage", None)
                usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
                usage = ({key: value for key, value in usage.items() if key in {
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_tokens_details", "completion_tokens_details"}}
                    if isinstance(usage, dict) else None)
                emit({"event": "http_attempt_completed", **event, **cost,
                      "attempt_finished_at_monotonic": time.monotonic(),
                      "http_status": 200,
                      "request_id": getattr(completion, "_request_id", None),
                      "usage_known": usage is not None, "usage": usage})
                cost.update(http_attempt_count=attempt,
                            http_attempt_timeout_sec=float(timeout_sec),
                            http_attempt_limit=max_attempts)
                return completion, cost
        raise AssertionError("vision attempt loop exited without a result")
    finally:
        await _close_client(client)
