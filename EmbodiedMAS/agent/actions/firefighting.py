"""Shared fire suppression control, preserving existing RPC/timing behavior."""
from __future__ import annotations
import asyncio
import tongsim as ts
from .geometry import _actor_key, _extinguisher_ab_from_target
from .pause_clock import sleep_active, wait_running

class FirefightingActions:
    async def suppress_fire(
        self,
        actor_id: bytes | str | dict,
        target_location: ts.Vector3,
        *,
        timeout: float = 5.0,
    ) -> str:
        """suppress_fire：对准目标世界坐标并执行灭火。"""
        actor_key = _actor_key(actor_id)
        await wait_running(self)

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=timeout
        )

        await wait_running(self)
        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not tf:
            print("[ActionAPI] 错误：无法获取当前坐标")
            return "Fail_CanotExtinguish"

        a, b = _extinguisher_ab_from_target(tf, target_location)

        await wait_running(self)
        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn,
            actor_key,
            a,
            b,
            timeout=timeout,
        )

        await wait_running(self)
        result = await ts.UnaryAPI.extinguish_fire(self._conn, actor_key, timeout=timeout)
        await sleep_active(self, self.suppression_duration_sec)

        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result
