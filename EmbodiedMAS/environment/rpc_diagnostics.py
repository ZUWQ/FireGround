"""Evaluation-only RPC diagnostics; never alter sensor payloads or retry calls."""
from __future__ import annotations
import time


def error_details(error):
    current = error
    seen = set()
    result = dict(error_type=type(error).__name__, error_message=str(error))
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, 'code', None)
        if callable(code):
            try:
                status = code()
                result['rpc_status'] = getattr(status, 'name', str(status))
                result['rpc_error_type'] = type(current).__name__
                break
            except Exception:
                pass
        current = current.__cause__ or current.__context__
    result['error_category'] = ('rpc_timeout' if result.get('rpc_status') == 'DEADLINE_EXCEEDED'
                                else 'rpc_not_found' if result.get('rpc_status') == 'NOT_FOUND'
                                else 'capture_error')
    return result


def clock_details(context, *, at=None):
    monitor = getattr(context, '_emas_monitor', None)
    now = time.monotonic() if at is None else at
    start, deadline = getattr(monitor, 'started_at', None), getattr(monitor, 'deadline', None)
    return dict(episode_elapsed_sec=None if start is None else now-start,
                episode_remaining_sec=None if deadline is None else max(0.0, deadline-now))
