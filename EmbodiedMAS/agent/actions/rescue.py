"""Civilian instruction controls with an optional result callback."""
from __future__ import annotations
import tongsim as ts
from .geometry import _actor_key
from .pause_clock import wait_running

class RescueActions:
    async def instruct_civilian(
        self,
        actor_id: bytes | str | dict,
        subtype: str,
    ) -> bool:
        """Send follow/stop; the RPC returns bool, not NPC recipient IDs."""
        key = str(subtype).strip().lower()
        if key not in ("follow", "stop_following"):
            raise ValueError(
                f"Unknown instruct_civilian subtype {subtype!r}. Expected 'follow' or 'stop_following'."
            )

        actor_key = _actor_key(actor_id)
        await wait_running(self)
        if self.supports_fire:
            await ts.UnaryAPI.set_extinguisher_rotation(
                self._conn, actor_key, 0.0, 0.0, timeout=5.0
            )

        await wait_running(self)
        if key == "follow":
            return await ts.UnaryAPI.sendfollow(self._conn, actor_key)

        result = await ts.UnaryAPI.sendstopfollow(self._conn, actor_key)
        if self.on_stop_follow is not None:
            await self.on_stop_follow(actor_key)
        return result
