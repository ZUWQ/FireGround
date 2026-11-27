"""Common navigation and scan control for all robot observation modes."""
from __future__ import annotations
import asyncio
import math
from typing import Any, Optional, Callable, Awaitable, Dict
import tongsim as ts
from .geometry import *
from .geometry import _actor_key, _get_yaw_from_quaternion
from .pause_clock import controller_for, timeout_active, wait_running, wait_for_cleanup

async def _navigate_to_location(
    conn: Any,
    actor_id: bytes | str,
    target_location: ts.Vector3,
    *,
    accept_radius: float | None = None,
    allow_partial: bool | None = None,
    timeout: float | None = None,
) -> Optional[dict]:
    """调用 UnaryAPI.navigate_to_location，其余参数使用模块级全局默认。"""
    return await ts.UnaryAPI.navigate_to_location(
        conn,
        actor_id,
        target_location,
        accept_radius=NAV_ACCEPT_RADIUS_UU if accept_radius is None else accept_radius,
        allow_partial=NAV_ALLOW_PARTIAL if allow_partial is None else allow_partial,
        speed_uu_per_sec=NAV_SPEED_UU_PER_SEC,
        timeout=NAV_TIMEOUT_SEC if timeout is None else timeout,
    )

class NavigationActions:
    async def navigate_to_location(
        self, actor_id: bytes | str | dict, target_location: ts.Vector3, *,
        accept_radius: float | None = None, allow_partial: bool | None = None,
        timeout: float | None = None,
    ) -> Optional[dict]:
        """Shared navigation primitive, with stricter per-call exploration options."""
        key = _actor_key(actor_id)
        await wait_running(self)
        await ts.UnaryAPI.set_extinguisher_rotation(self._conn, key, 0.0, 0.0, timeout=5.0)
        controller = controller_for(self)
        if controller is None:
            return await _navigate_to_location(
                self._conn, key, target_location, accept_radius=accept_radius,
                allow_partial=allow_partial, timeout=timeout)
        # UE movement may be frozen repeatedly by other members. Keep the
        # navigation business limit in environment time; a separate finite real
        # transport deadline only insures a non-responsive RPC. Do not resend
        # navigation when the world resumes.
        business_timeout = NAV_TIMEOUT_SEC if timeout is None else timeout
        transport_timeout = max(float(business_timeout), controller.rpc_wall_timeout_sec)
        await wait_running(self)
        try:
            async with timeout_active(self, business_timeout):
                result = await _navigate_to_location(
                    self._conn, key, target_location, accept_radius=accept_radius,
                    allow_partial=allow_partial, timeout=transport_timeout)
                await wait_running(self)
                return result
        except TimeoutError:
            # The legacy SDK returns None for a deadline error. Preserve that
            # action contract, including exploration's actual-position readback.
            return None

    async def world_locations_from_sos(
        self, listener_actor_id: bytes | str | dict, sos_info: Dict[str, Any],
    ) -> list[Optional[ts.Vector3]]:
        """Project every bearing/range pair with one listener-pose query, in UU.

        Yaw is the legacy world-space angle in degrees. Invalid samples retain
        their indices as None; Z is the listener height, not a measured NPC Z.
        """
        orientations = sos_info.get("orientations")
        distances = sos_info.get("distances")
        orientations = orientations if isinstance(orientations, (list, tuple)) else []
        distances = distances if isinstance(distances, (list, tuple)) else []
        points: list[Optional[ts.Vector3]] = [None] * max(len(orientations), len(distances))
        samples = []
        for index, (orientation, distance) in enumerate(zip(orientations, distances)):
            if not isinstance(orientation, dict):
                continue
            yaw = orientation.get("yaw")
            if isinstance(yaw, bool) or isinstance(distance, bool):
                continue
            try:
                yaw, distance = float(yaw), float(distance)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(yaw) and math.isfinite(distance) and distance > 0:
                samples.append((index, yaw, distance))
        if not samples:
            return points
        tf = await ts.UnaryAPI.get_actor_transform(self._conn, _actor_key(listener_actor_id))
        if not tf or not all(math.isfinite(float(getattr(tf.location, axis))) for axis in ("x", "y", "z")):
            return points
        for index, yaw, distance in samples:
            point = world_position_from_yaw_distance_2d(
                tf.location.x, tf.location.y, tf.location.z, yaw, distance)
            if point is not None and all(math.isfinite(value) for value in point):
                points[index] = ts.Vector3(*point)
        return points

    async def world_location_from_sos(
        self, listener_actor_id: bytes | str | dict, sos_info: Dict[str, Any],
    ) -> Optional[ts.Vector3]:
        """Legacy first-sample API; runtime uses all samples for spatial filtering."""
        points = await self.world_locations_from_sos(listener_actor_id, sos_info)
        return points[0] if points else None

    async def navigate_entity(
        self,
        actor_id: bytes | str | dict,
        target_location: ts.Vector3,
    ) -> Optional[dict]:
        """navigate / entity：导航至世界坐标。"""
        actor_key = _actor_key(actor_id)

        return await self.navigate_to_location(actor_key, target_location)
    async def navigate_relative(
        self,
        actor_id: bytes | str | dict,
        distance: float,
        direction: str,
    ) -> Optional[dict]:
        """navigate / relative：按米与离散方向相对当前朝向移动。"""
        actor_key = _actor_key(actor_id)
        await wait_running(self)
        distance_uu = meters_to_uu(validate_relative_distance(distance))
        angle_deg = direction_to_angle(direction)

        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not tf:
            print(f"[ActionAPI] navigate_relative: failed to get transform for {actor_key!r}")
            return None

        phi = _get_yaw_from_quaternion(tf.rotation)
        move_yaw = phi + math.radians(angle_deg)
        target_location = ts.Vector3(
            tf.location.x + distance_uu * math.cos(move_yaw),
            tf.location.y + distance_uu * math.sin(move_yaw),
            NAV_TARGET_Z,
        )

        return await self.navigate_to_location(actor_key, target_location)
    async def observe_surroundings(
        self,
        actor_id: bytes | str | dict,
        *,
        capture_view: Callable[[str], Awaitable[None]],
    ) -> None:
        """Capture front/left/back/right, then recover the original orientation.

        Transform readback confirms the RPC-visible pose only. It does not prove
        that UE navigation has stopped or that a rendered frame has advanced.
        """
        actor_key = _actor_key(actor_id)
        await wait_running(self)
        initial = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if initial is None:
            raise RuntimeError("Cannot observe surroundings without the current transform")

        def rotation_matches(actual: Any, expected: Any) -> bool:
            a = tuple(float(getattr(actual, axis)) for axis in ("x", "y", "z", "w"))
            b = tuple(float(getattr(expected, axis)) for axis in ("x", "y", "z", "w"))
            norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
            if not math.isfinite(norm) or norm <= 0:
                return False
            cosine = abs(sum(x * y for x, y in zip(a, b))) / norm
            # Quaternions q and -q denote the same orientation; allow 0.5 degree.
            return min(1.0, cosine) >= math.cos(math.radians(0.5) / 2)

        if not rotation_matches(initial.rotation, initial.rotation):
            raise RuntimeError("Cannot observe surroundings with an invalid rotation")
        original_rotation = initial.rotation
        euler = ts.math.quaternion_to_euler(original_rotation, is_degree=True)
        await wait_running(self)
        ok = await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=5.0,
        )
        if not ok:
            raise RuntimeError("Cannot reset the observation heading")

        async def restore_orientation() -> None:
            await wait_for_cleanup(self)
            current = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
            if current is None:
                raise RuntimeError("Cannot recover orientation: current transform unavailable")
            await wait_for_cleanup(self)
            restored = await ts.UnaryAPI.set_actor_transform(
                self._conn, actor_key,
                ts.Transform(location=current.location, rotation=original_rotation, scale=current.scale),
            )
            if not restored:
                raise RuntimeError("Cannot recover the original orientation")
            checked = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
            if checked is None or not rotation_matches(checked.rotation, original_rotation):
                raise RuntimeError("Original orientation recovery could not be confirmed")

        try:
            await wait_running(self)
            await capture_view("front")
            # UE's right vector is +Y at zero yaw, so left is negative yaw.
            for view, offset in (("left", -90.0), ("back", 180.0), ("right", 90.0)):
                await wait_running(self)
                current = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
                if current is None:
                    raise RuntimeError(f"Cannot turn to {view}: current transform unavailable")
                rotation = ts.math.euler_to_quaternion(
                    ts.Vector3(float(euler.x), float(euler.y), float(euler.z) + offset),
                    is_degree=True,
                )
                await wait_running(self)
                turned = await ts.UnaryAPI.set_actor_transform(
                    self._conn, actor_key,
                    ts.Transform(location=current.location, rotation=rotation, scale=current.scale),
                )
                if not turned:
                    raise RuntimeError(f"Failed to turn to {view}")
                checked = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
                if checked is None or not rotation_matches(checked.rotation, rotation):
                    raise RuntimeError(f"Turn to {view} could not be confirmed")
                await wait_running(self)
                await capture_view(view)
        finally:
            # Recovery belongs to this operation even if its caller is cancelled.
            # Shield it and await it through repeated cancellation: no detached RPC.
            recovery = asyncio.create_task(restore_orientation())
            cancelled_during_recovery = False
            while not recovery.done():
                try:
                    await asyncio.shield(recovery)
                except asyncio.CancelledError:
                    cancelled_during_recovery = True
            recovery.result()
            if cancelled_during_recovery:
                raise asyncio.CancelledError
