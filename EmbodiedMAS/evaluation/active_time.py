"""Audit confirmed pause intervals without rewriting raw wall-clock evidence."""
from __future__ import annotations
import math


def pause_mapper(summary: dict, metadata: dict):
    """Return (clock, mapper, valid); missing/invalid opt-in evidence fails closed."""
    clock = summary.get("clock", metadata.get("clock", "monotonic_wall_seconds"))
    if clock != "active_wall_seconds":
        # Legacy metrics always use and report wall time, even when an older
        # artifact has a missing, null or unrecognized clock label.
        return "monotonic_wall_seconds", lambda value: value, True
    ledger = summary.get("pause_intervals", metadata.get("pause_intervals"))
    complete = summary.get("pause_ledger_complete", metadata.get("pause_ledger_complete"))
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if complete is not True or not isinstance(ledger, list):
        return clock, lambda value: value, False
    intervals = []
    previous = -float("inf")
    for item in ledger:
        if not isinstance(item, dict):
            return clock, lambda value: value, False
        start, end = item.get("start_wall"), item.get("end_wall")
        if not finite(start) or not finite(end) or end < start or start < previous:
            return clock, lambda value: value, False
        duration = item.get("duration_sec")
        if not finite(duration) or not math.isclose(duration, end - start, abs_tol=1e-6):
            return clock, lambda value: value, False
        intervals.append((float(start), float(end))); previous = end
    total = summary.get("total_paused_sec", metadata.get("total_paused_sec"))
    if not finite(total) or not math.isclose(total, sum(b-a for a,b in intervals), abs_tol=1e-6):
        return clock, lambda value: value, False
    def map_wall(value):
        if not finite(value):
            return value
        return value - sum(max(0.0, min(value, end) - start) for start, end in intervals)
    return clock, map_wall, True
