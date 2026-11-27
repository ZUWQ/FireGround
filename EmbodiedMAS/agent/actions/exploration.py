"""Bounded room-perimeter exploration using shared navigation, never sensing."""
from __future__ import annotations

import asyncio
import copy
import math
import time
from typing import Any

import tongsim as ts

from EmbodiedMAS.environment.room_geometry import finite_number, exploration_region_from_room
from EmbodiedMAS.agent.config.environment_config import exploration_options
from .geometry import _actor_key, NAV_SPEED_UU_PER_SEC
from .pause_clock import active_now, elapsed_now, timeout_active, wait_running

class _ExploreFailure(Exception):
    pass


def _xyz(value: Any) -> tuple[float, float, float]:
    try:
        result = tuple(finite_number(getattr(value, axis), f"position.{axis}")
                       for axis in ("x", "y", "z"))
    except (AttributeError, ValueError) as exc:
        raise _ExploreFailure("invalid_actor_or_path_position") from exc
    return result


def _xy_distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ExplorationActions:
    async def explore_room(self, actor_id: Any, room: dict[str, Any], *,
                           coordinate_unit: str = "uu", units_per_meter: float = 100.0) -> dict[str, Any]:
        """Visit in-room projected waypoints, recover locally, and report skips.

        Preflight and actor readback reduce room-escape errors. The controller
        still replans server-side, so this does not claim strict path confinement
        or visibility coverage. The runtime owns all periodic/boundary sensing.
        """
        progress: dict[str, Any] = {
            "room": room.get("room_name"), "polygon_source": room.get("polygon_source", "unknown"),
            "route_kind": "clockwise_inset_perimeter", "waypoints": [],
            "completed_waypoints": [], "completed_segments": 0, "closed_loop": False,
            "actual_end_position": None, "last_observed_position": None,
            "partial_exploration": False, "visited_waypoint_count": 0,
            "skipped_waypoint_count": 0, "alternative_waypoints_used": 0, "projected_waypoints_used": 0,
            "skipped_waypoints": [], "skipped_waypoint_reasons": [],
            "recovery_attempts": [], "accepted_waypoints": [], "travelled_distance_m": 0.0,
        }

        def publish() -> None:
            self.exploration_progress = copy.deepcopy(progress)

        def result(status: str, reason: str) -> dict[str, Any]:
            progress["stop_reason"] = reason
            publish()
            return {"status": status, "reason": reason, "data": copy.deepcopy(progress)}

        publish()
        try:
            options = exploration_options(getattr(self, "exploration_config", None))
            scale = finite_number(units_per_meter, "units_per_meter", positive=True)
            region = exploration_region_from_room(
                room, inset_m=options["inset_m"], coordinate_unit=coordinate_unit, units_per_meter=scale)
        except (ValueError, TypeError, KeyError) as exc:
            return result("failed", f"invalid_room_or_config: {exc}")

        actor_key = _actor_key(actor_id)
        tolerance = options["endpoint_tolerance_m"] * scale
        acceptance = options["acceptance_radius_m"] * scale
        segment_timeout = options["segment_timeout_sec"]
        progress.update(inset_m=options["inset_m"], coordinate_unit="uu", units_per_meter=scale,
                        acceptance_radius_m=options["acceptance_radius_m"],
                        endpoint_tolerance_m=options["endpoint_tolerance_m"])
        # Distinct corners must have disjoint arrival regions. Otherwise the
        # acceptance radius can count an entire tiny loop without any movement.
        corners = region.inset_corners_clockwise()
        if min(math.dist(a, b) for a, b in zip(corners, corners[1:] + corners[:1])) <= 2 * acceptance:
            return result("failed", "room_too_small_for_navigation_tolerance")

        async def position() -> tuple[float, float, float]:
            await wait_running(self)
            transform = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
            if transform is None:
                raise _ExploreFailure("actor_state_unavailable")
            current = _xyz(transform.location)
            progress["actual_end_position"] = list(current)
            progress["last_observed_position"] = list(current)
            publish()
            return current

        async def checked_path(start, target, *, inside: bool) -> tuple[float, tuple[float, float, float]]:
            check = dict(requested_start=list(start), requested_target=list(target),
                         room_path_required=inside, reason=None)
            progress["last_path_check"] = check
            publish()

            def reject(reason: str) -> None:
                check["reason"] = reason
                publish()
                raise _ExploreFailure(reason)

            await wait_running(self)
            path = await ts.UnaryAPI.query_navigation_path(
                self._conn, ts.Vector3(*start), ts.Vector3(*target), allow_partial=False,
                require_navigable_end_location=True, timeout=segment_timeout,
            )
            if not isinstance(path, dict) or path.get("is_partial") is not False:
                reject("path_unavailable_or_partial")
            raw_points = path.get("points")
            if not isinstance(raw_points, (list, tuple)) or not raw_points:
                reject("path_has_no_points")
            points = tuple(_xyz(value) for value in raw_points)
            check.update(points=[list(point) for point in points],
                         projected_start=list(points[0]), projected_end=list(points[-1]))
            # Room membership is independent of the coarse arrival tolerance.
            # A nearby endpoint across a wall can never be an accepted goal.
            if not region.contains(points[-1][:2]):
                reject("path_endpoint_outside_room")
            check.update(projection_start_distance_m=_xy_distance(points[0], start) / scale,
                         projection_end_distance_m=_xy_distance(points[-1], target) / scale)
            if inside and not region.contains_path([start[:2], *(point[:2] for point in points)]):
                reject("path_leaves_room")
            length = path.get("path_length")
            if (isinstance(length, bool) or not isinstance(length, (int, float))
                    or not math.isfinite(length) or length < 0):
                reject("invalid_path_length")
            measured = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            check["reason"] = "accepted"
            publish()
            return (float(length) if length > 0 else measured), points[-1]

        visited_targets: list[tuple[float, float, float]] = []
        visited_positions: list[tuple[float, float, float]] = []

        async def move_to(target, *, inside: bool, approach: bool = False,
                          closing: bool = False) -> tuple[float, float, float]:
            current = await position()
            if inside and not region.contains(current[:2]):
                raise _ExploreFailure("actor_left_room")
            # Fresh preflight is required even when candidate selection already
            # found a path; obstacles/controller state can change between legs.
            path_length, move_target = await checked_path(current, target, inside=inside)
            if not closing and any(_xy_distance(move_target, point) <= acceptance
                                   for point in visited_targets):
                raise _ExploreFailure("projected_target_already_visited")
            navigation_budget_elapsed = False
            navigation_started = _xy_distance(current, move_target) > acceptance
            if navigation_started:
                navigation_timeout = segment_timeout
                if approach:
                    # Crossing wings can exceed a circuit leg's 30-second cap
                    # even without obstacles. Budget the already checked route
                    # at the same commanded speed as NavigationActions, with
                    # 25% travel slack and 5 seconds for acceleration/settling.
                    speed = finite_number(NAV_SPEED_UU_PER_SEC, "navigation speed", positive=True)
                    navigation_timeout = min(
                        max(segment_timeout, path_length / speed * 1.25 + 5.0),
                        exploration_deadline - active_now(self))
                    if navigation_timeout <= 0:
                        raise TimeoutError
                    progress["approach_navigation"] = dict(path_length_uu=path_length,
                        speed_uu_per_sec=speed, timeout_sec=navigation_timeout,
                        elapsed_sec=None, response_received=False, budget_elapsed=False)
                    publish()
                started = elapsed_now(self)
                navigation = await self.navigate_to_location(
                    actor_key, ts.Vector3(*move_target), accept_radius=acceptance,
                    allow_partial=False, timeout=navigation_timeout,
                )
                elapsed = elapsed_now(self) - started
                if approach:
                    # The SDK folds every RPC error into None. Report timing and
                    # response presence; do not invent a specific gRPC status.
                    navigation_budget_elapsed = navigation is None and elapsed >= navigation_timeout
                    progress["approach_navigation"].update(elapsed_sec=elapsed,
                        response_received=isinstance(navigation, dict),
                        budget_elapsed=navigation_budget_elapsed)
                    publish()
                actual = await position()
                progress["last_navigation"] = dict(
                    requested_target=list(target), target=list(move_target),
                    started_position=list(current), actual_end_position=list(actual),
                    moved_distance_m=_xy_distance(current, actual) / scale,
                    response_received=isinstance(navigation, dict),
                    success=navigation.get("success") if isinstance(navigation, dict) else None,
                    is_partial=navigation.get("is_partial") if isinstance(navigation, dict) else None,
                    message=navigation.get("message") if isinstance(navigation, dict) else None)
                progress["travelled_distance_m"] += _xy_distance(current, actual) / scale
                publish()
                # Arrival is verified below from the actual pose, including when
                # the navigation service reports a partial path.
            else:
                actual = current
            if not region.contains(actual[:2]):
                raise _ExploreFailure("navigation_ended_outside_room")
            # Two metres is an arrival verdict, not permission to count several
            # nearby waypoints while the robot is stuck at the previous one.
            if (navigation_started and _xy_distance(actual, move_target) > acceptance
                    and _xy_distance(current, actual) < 5.0):
                raise _ExploreFailure("navigation_budget_elapsed" if navigation_budget_elapsed
                                      else "navigation_no_progress")
            # A valid in-room projection is the accepted destination. Its
            # distance from the ideal corner does not invalidate arrival.
            arrival_target = progress["loop_start_position"] if closing else move_target
            if _xy_distance(actual, arrival_target) > tolerance:
                raise _ExploreFailure("navigation_budget_elapsed" if navigation_budget_elapsed
                                      else "navigation_ended_short")
            return actual

        recoverable = {
            "path_unavailable_or_partial", "path_has_no_points", "path_endpoint_outside_room",
            "path_leaves_room", "navigation_budget_elapsed", "navigation_no_progress",
            "navigation_ended_short", "projected_target_already_visited", "waypoint_no_new_position",
        }

        def nearby_targets(ideal):
            # Bounded local recovery: original point, then eight directions on
            # 0.5m and 1m rings. Prefer inward alternatives at the same radius.
            yield tuple(ideal), False
            centre = (sum(p[0] for p in corners) / len(corners),
                      sum(p[1] for p in corners) / len(corners))
            for radius_m in (0.5, 1.0):
                radius = radius_m * scale
                ring = [(ideal[0] + radius * math.cos(i * math.pi / 4),
                         ideal[1] + radius * math.sin(i * math.pi / 4), ideal[2])
                        for i in range(8)]
                for point in sorted(ring, key=lambda p: _xy_distance(p, centre)):
                    if region.contains(point[:2]):
                        yield point, True

        async def attempt_waypoint(ideal, index, *, approach=False, closing=False):
            progress["active_waypoint"] = index
            for candidate, alternative in nearby_targets(ideal):
                current = await position()
                progress["candidate_kind"] = "nearby" if alternative else "ideal"
                attempt = dict(waypoint_index=index, closing=closing, ideal_target=list(ideal),
                               requested_target=list(candidate), alternative=alternative,
                               started_position=list(current), status="checking")
                progress["recovery_attempts"].append(attempt)
                publish()
                try:
                    actual = await move_to(candidate, inside=(region.contains(current[:2])
                                           if approach else True), approach=approach, closing=closing)
                except _ExploreFailure as exc:
                    attempt.update(status="rejected", reason=str(exc))
                    check = progress.get("last_path_check", {})
                    if check.get("requested_target") == list(candidate):
                        attempt["path_check"] = copy.deepcopy(check)
                    publish()
                    if str(exc) not in recoverable:
                        raise
                    continue
                if not closing and any(_xy_distance(actual, p) <= 5.0 for p in visited_positions):
                    attempt.update(status="rejected", reason="waypoint_no_new_position",
                                   actual_position=list(actual), path_check=copy.deepcopy(progress["last_path_check"]))
                    publish()
                    continue
                check = copy.deepcopy(progress["last_path_check"])
                attempt.update(status="visited", reason="accepted", path_check=check,
                               projected_target=check["projected_end"], actual_position=list(actual))
                if not closing:
                    visited_targets.append(tuple(check["projected_end"]))
                    visited_positions.append(actual)
                # One centimetre only filters floating-point jitter for result
                # classification; projection distance never gates acceptance.
                if check["projection_end_distance_m"] > 0.01:
                    progress["projected_waypoints_used"] += 1
                if alternative:
                    progress["alternative_waypoints_used"] += 1
                progress["accepted_waypoints"].append(copy.deepcopy(attempt))
                publish()
                return actual
            return None

        def skip(index):
            progress["skipped_waypoints"].append(index)
            progress["skipped_waypoint_reasons"].append(
                dict(waypoint_index=index, reason="no_reachable_local_alternative"))
            progress["skipped_waypoint_count"] = len(progress["skipped_waypoints"])
            progress["partial_exploration"] = True
            publish()

        async def run_loop() -> dict[str, Any]:
            initial = await position()
            candidates, rejections = [], []
            inside = region.contains(initial[:2])
            for index, point in enumerate(corners):
                target = (point[0], point[1], initial[2])
                try:
                    length, _ = await checked_path(initial, target, inside=inside)
                except _ExploreFailure as exc:
                    if str(exc) not in recoverable:
                        raise
                    rejections.append(str(exc))
                else:
                    candidates.append((length, index))
            progress["candidate_rejections"] = rejections
            # Prefer an originally reachable corner; if none exists, local
            # alternatives around every corner can still provide an entry.
            starts = [index for _, index in sorted(candidates)]
            starts += [i for i in sorted(range(len(corners)),
                       key=lambda i: _xy_distance(initial, corners[i])) if i not in starts]
            loop_start, first = None, None
            progress["phase"] = "approach"
            for first in starts:
                route = list(corners[first:] + corners[:first])
                progress["waypoints"] = [[x, y, initial[2]] for x, y in route]
                progress["ideal_waypoints"] = copy.deepcopy(progress["waypoints"])
                publish()
                loop_start = await attempt_waypoint(tuple(progress["waypoints"][0]), 0, approach=True)
                if loop_start is not None:
                    break
            if loop_start is None:
                raise _ExploreFailure("no_reachable_start")
            progress["loop_start_position"] = list(loop_start)
            progress["waypoints"][0] = list(loop_start)
            progress["completed_waypoints"].append(0)
            progress["visited_waypoint_count"] = 1
            progress["phase"] = "circuit"
            publish()
            for index, point in enumerate(route[1:], start=1):
                progress["next_segment"] = index - 1
                publish()
                actual = await attempt_waypoint((point[0], point[1], initial[2]), index)
                if actual is None:
                    skip(index)
                    continue
                progress["completed_segments"] += 1
                progress["completed_waypoints"].append(index)
                progress["visited_waypoint_count"] += 1
                publish()
            # A partial survey ends where useful progress ended. It does not
            # spend the remaining budget forcing a return to its first point.
            if not progress["partial_exploration"]:
                progress["next_segment"] = len(route) - 1
                actual = await attempt_waypoint(loop_start, 0, closing=True)
                if actual is None:
                    skip(0)
                else:
                    progress["completed_segments"] += 1
                    progress["completed_waypoints"].append(0)
                    progress["closed_loop"] = True
            progress["phase"] = "finished"
            progress["next_segment"] = None
            if progress["travelled_distance_m"] < 0.05:
                return result("failed", "exploration_no_progress")
            if progress["partial_exploration"]:
                return result("completed", "reachable_route_completed")
            return result("completed", "adapted_route_completed" if (progress["alternative_waypoints_used"] or progress["projected_waypoints_used"])
                          else "perimeter_route_completed")

        exploration_deadline = active_now(self) + options["total_timeout_sec"]
        try:
            async with timeout_active(self, options["total_timeout_sec"]):
                return await run_loop()
        except asyncio.CancelledError:
            # A cancelled navigation may have moved since the last readback.
            # Preserve that historical sample without calling it the endpoint;
            # the runtime will obtain a fresh pose in its normal boundary scan.
            progress["actual_end_position"] = None
            progress["stop_reason"] = "interrupted"
            publish()
            raise
        except TimeoutError:
            progress["actual_end_position"] = None
            return result("failed", "exploration_timeout")
        except _ExploreFailure as exc:
            return result("failed", str(exc))
        except Exception as exc:
            return result("failed", f"exploration_error: {type(exc).__name__}: {exc}")
