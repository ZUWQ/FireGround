"""Shared low-level robot actions without metric recording or legacy adapters."""
from __future__ import annotations
import asyncio
from typing import Any
import tongsim as ts
from tongsim.core.world_context import WorldContext
from .geometry import *
from .navigation import NavigationActions
from .firefighting import FirefightingActions
from .rescue import RescueActions
from .exploration import ExplorationActions
from .results import action_result
from .pause_clock import sleep_active, wait_running
from EmbodiedMAS.agent.config.environment_config import load_environment_config, exploration_options

class ActionAPI(NavigationActions, FirefightingActions, RescueActions, ExplorationActions):
    def __init__(self, context: WorldContext, observation_camera: Any = None,
                 *, modality: str = "text", supports_fire: bool = True,
                 on_stop_follow: Any = None, exploration_config: dict[str, Any] | None = None):
        self._context = context
        self._modality = modality
        self._conn = context.conn
        self._observation_camera = observation_camera
        self.supports_fire = supports_fire
        self.suppression_duration_sec = 15.0 if modality == "vision" else 10.0
        self.on_stop_follow = on_stop_follow
        self.exploration_config = exploration_options(
            load_environment_config().get("exploration") if exploration_config is None else exploration_config)
        self.exploration_progress: dict[str, Any] = {}

    async def wait(self, actor_id: bytes | str | dict) -> dict[str, Any]:
        """Wait in place; the runtime owns periodic observation and cancellation."""
        await wait_running(self)
        await sleep_active(self, 10.0)
        return action_result("completed", "wait_elapsed", {"duration_sec": 10.0})


    async def get_camera_info_and_convert(
        self,
        agent_ref: bytes | str | dict,
    ) -> dict[str, Any]:
        """获取 agent 相机位姿等信息（调试用 / 与 VisionRobot 接口一致）。"""
        from EmbodiedMAS.agent.perception.vision_pipeline.camera_config import camera_world_position_from_agent, orientation_to_yaw_deg
        agent_id_str = agent_ref["id"] if isinstance(agent_ref, dict) else agent_ref
        agent_transform = await ts.UnaryAPI.get_actor_transform(self._conn, agent_id_str)
        if not agent_transform:
            raise RuntimeError(f"get_actor_transform failed for {agent_id_str!r}")
        cam_x, cam_y, cam_z = camera_world_position_from_agent(
            agent_transform.location.x,
            agent_transform.location.y,
            agent_transform.location.z,
            agent_transform.rotation,
        )
        camera_pos = ts.Vector3(cam_x, cam_y, cam_z)
        yaw_deg = orientation_to_yaw_deg(agent_transform.rotation)
        orientation = (0.0, 0.0, yaw_deg)
        return {
            "camera_position": camera_pos,
            "camera_rotation": agent_transform.rotation,
            "agent_transform": agent_transform,
            "orientation": orientation,
            "world_coord": None,
        }




