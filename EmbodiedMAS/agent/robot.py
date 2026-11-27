"""Robot body shared by text and vision cognitive agents.

The body owns identity, cameras and action access. Decision policy, cognitive
memory and event scheduling belong to the runtime above it.
"""
from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext
from EmbodiedMAS.environment.camera import ObservationCamera
from EmbodiedMAS.environment.entities import (
    SpawnedActor, validate_agent_name, spawn_entity, get_entity_state,
    agent_id_to_fguid_bytes,
)
from EmbodiedMAS.agent.actions.api import ActionAPI
from EmbodiedMAS.environment.regions import resolve_room_spec
from EmbodiedMAS.agent.actions.executor import ActionExecutor
from EmbodiedMAS.agent.actions.targets import DestroyedTargetError, TargetResolver
from EmbodiedMAS.agent.perception.records import ObservationFrame
from EmbodiedMAS.agent.visual_backends import supports_image_targets
from EmbodiedMAS.evaluation.monitor import record_event

FireDog_BP = "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
SaveDog_BP = "/Game/Blueprint/BP_Savedog.BP_Savedog_C"
Firefighter_BP = "/Game/Blueprint/BP_Firefighter.BP_Firefighter_C"
ExploreDog_BP = "/Game/Blueprint/BP_ExploreDog.BP_ExploreDog_C"
Z_HEIGHT = 1000
_BLUEPRINTS = {"FR": FireDog_BP, "RR": SaveDog_BP, "HF": Firefighter_BP, "ER": ExploreDog_BP}


class RobotAgent:
    def __init__(self, context: WorldContext, *, modality: str = "text", agent_type: str = "FR",
                 log_dir: Optional[Path] = None, perception: Any = None, actions: Any = None,
                 on_spawn: Any = None):
        self.modality = str(modality).lower()
        self.agent_type = str(agent_type).upper()
        if self.modality not in {"text", "vision"}:
            raise ValueError(f"Unsupported modality: {modality!r}")
        if self.agent_type not in _BLUEPRINTS:
            raise ValueError(f"Unsupported agent_type: {agent_type!r}")
        self._context = context
        self._conn = context.conn
        self._blueprint = _BLUEPRINTS[self.agent_type]
        self.supports_fire = self.agent_type in {"FR", "HF"}
        self._obs_camera = ObservationCamera(context, log_dir=log_dir) if self.modality == "vision" else None
        self._actions = actions if actions is not None else ActionAPI(
            context, observation_camera=self._obs_camera, modality=self.modality,
            supports_fire=self.supports_fire)
        self._perception_module = perception
        self._spawned: Optional[SpawnedActor] = None
        self._owns_spawned_actor = False
        self._spawn_camera_id = None
        self._agent_cameras = {}
        self._owned_camera_ids = set()
        self._semantic_maps = []
        self._targets = TargetResolver()
        self._image_observation = None
        self._image_locations: dict[tuple, Any] = {}
        self.image_grounding_history: list[dict[str, Any]] = []
        self._resolved_navigation_target: Any = None
        self._executor = ActionExecutor(self)
        self._execution_tasks: set[asyncio.Task] = set()
        self._on_spawn = on_spawn
        self._deploy_listeners: list[Callable[[RobotAgent], Awaitable[None]]] = []

    @property
    def is_spawned(self) -> bool:
        return self._spawned is not None

    @property
    def agent_guid(self) -> str:
        if self._spawned is None:
            raise RuntimeError("Agent not spawned")
        return self._spawned.guid

    @property
    def agent_name(self) -> str:
        if self._spawned is None or not self._spawned.name:
            raise RuntimeError("Agent not spawned")
        return self._spawned.name

    @property
    def _agent(self) -> Optional[dict]:
        if self._spawned is None:
            return None
        raw = self._spawned.to_spawn_dict()
        if self._spawn_camera_id is not None:
            raw["camera_id"] = self._spawn_camera_id
        return raw

    def bind_spawned_actor(self, raw: dict, *, camera_id: Any = None) -> SpawnedActor:
        self._spawned = SpawnedActor.from_spawn_dict(raw)
        self._spawn_camera_id = camera_id
        return self._spawned

    def add_deploy_listener(self, callback: Callable[[RobotAgent], Awaitable[None]]) -> None:
        """Register an async hook invoked after this body's deployment succeeds."""
        if callback not in self._deploy_listeners:
            self._deploy_listeners.append(callback)

    def remove_deploy_listener(self, callback: Callable[[RobotAgent], Awaitable[None]]) -> None:
        if callback in self._deploy_listeners:
            self._deploy_listeners.remove(callback)

    def bind_perception_module(self, module: Any) -> None:
        self._perception_module = module

    def _ensure_perception_module(self) -> Any:
        if self._perception_module is None:
            if self.modality == "vision":
                from EmbodiedMAS.agent.perception.vision import VisionPerceptionModule
                self._perception_module = VisionPerceptionModule(self._context)
            else:
                from EmbodiedMAS.agent.perception.text import PerceptionModule
                self._perception_module = PerceptionModule(self._context)
        return self._perception_module

    def set_semantic_maps(self, maps: list[dict[str, Any]] | None) -> None:
        self._semantic_maps = maps or []

    def bind_targets(self, targets: dict[str, tuple[float, float, float]], *,
                     entities: Any = ()) -> None:
        """Replace this decision's grounding; text history remains in perception."""
        self._targets.bind_targets(targets, entities=entities)

    @property
    def supports_image_targets(self) -> bool:
        backend = getattr(self._perception_module, "backend", None)
        return self.modality == "vision" and supports_image_targets(getattr(backend, "name", None))

    def bind_image_observation(self, observation: Any) -> None:
        """Pin the immutable decision snapshot; periodic captures cannot replace it."""
        self._image_observation = observation if self.supports_image_targets else None
        self._image_locations.clear()

    async def resolve_image_location(self, target: dict, *, purpose: str) -> Any:
        from EmbodiedMAS.agent.actions.image_targets import ImageTargetError, validate_image_target
        if not self.supports_image_targets:
            raise ImageTargetError("image_target_unavailable", "Image targets require the raw_vlm or grounded_sam_pixel backend.")
        from EmbodiedMAS.agent.perception.vision.image_grounding import ground_image_target
        backend = getattr(self._perception_module.backend, "name", None)
        audit = {"purpose": purpose, "target_kind": "pixel", "target": copy.deepcopy(target),
                 "agent_id": self.agent_guid, "vision_backend": backend,
                 "batch_id": getattr(self._image_observation, "batch_id", None)}
        try:
            target = validate_image_target(target)
            if self._image_observation is None:
                raise ImageTargetError("frame_not_actionable", "No current actionable image is bound.")
            key = purpose, target["frame_ref"], tuple(target["pixel_xy"])
            if key not in self._image_locations:
                result = ground_image_target(self._image_observation, target, purpose=purpose)
                self._image_locations[key] = tuple(result.execution_point)
                audit.update(result.audit())
                self.image_grounding_history.append(copy.deepcopy(audit))
                record_event(self._context, "actions", {"event": "image_target_grounded", **audit})
        except ImageTargetError as exc:
            record_event(self._context, "actions", {"event": "image_target_rejected", **audit,
                                                   "reason": exc.code})
            raise
        return ts.Vector3(*self._image_locations[key])

    async def navigate_image(self, target: dict) -> Any:
        return await self.navigate_resolved(await self.resolve_image_location(target, purpose="navigate"))

    async def deploy_agent(self, *, name: str, location: Any = None, rotation: Any = None,
                           tags: Any = None, timeout: float = 5.0) -> SpawnedActor:
        await self.spawn_agent(name=name, location=location, rotation=rotation, tags=tags, timeout=timeout)
        if self._spawned is None:
            raise RuntimeError(f"Failed to deploy agent (blueprint={self._blueprint!r}, name={name!r})")
        return self._spawned

    deploy = deploy_agent

    async def spawn_agent(self, *, name: str, location: Any = None, rotation: Any = None,
                          tags: Any = None, timeout: float = 5.0) -> Any:
        raw = await spawn_entity(self._context, blueprint=self._blueprint, name=name, location=location,
                                 rotation=rotation, tags=tags, timeout=timeout)
        if not raw:
            if self.modality == "vision":
                raise RuntimeError(f"spawn_actor failed for blueprint {self._blueprint!r}")
            return None
        identity = self.bind_spawned_actor(raw)
        self._owns_spawned_actor = True
        # Retain ownership even if camera setup fails, so a runner can clean up.
        if self.modality == "vision":
            from EmbodiedMAS.agent.perception.vision_pipeline.camera_config import (
                AGENT_CAMERA_CAPTURE_PARAMS, camera_world_position_from_agent,
            )
            transform = await ts.UnaryAPI.get_actor_transform(self._conn, identity.guid)
            if not transform:
                raise RuntimeError(f"get_actor_transform failed for {identity.guid!r}")
            position = camera_world_position_from_agent(
                transform.location.x, transform.location.y, transform.location.z, transform.rotation)
            camera_id = await ts.CaptureAPI.create_camera(
                self._conn, transform=ts.Transform(location=ts.Vector3(*position), rotation=transform.rotation),
                params=dict(AGENT_CAMERA_CAPTURE_PARAMS), capture_name=f"AgentCamera_{validate_agent_name(name)}",
                attach_parent=agent_id_to_fguid_bytes(identity.guid), attach_socket="", keep_world=True)
            if not camera_id:
                raise RuntimeError(f"create_camera failed for agent {identity.guid!r}")
            self._owned_camera_ids.add(camera_id)
            self._agent_cameras[identity.guid] = camera_id
            self._obs_camera.register_agent_camera(identity.guid, camera_id)
            self.bind_spawned_actor(raw, camera_id=camera_id)
        else:
            self.bind_spawned_actor(raw)
        if self._on_spawn is not None:
            if self.modality == "vision":
                await self._on_spawn(self._context, self._agent)
            else:
                try:
                    await self._on_spawn(self._context, self._agent)
                except Exception:
                    # Existing text spawn treats diagnostic recording as best effort.
                    pass
        for callback in tuple(self._deploy_listeners):
            await callback(self)
        return self._spawned if self.modality == "vision" else self._spawned.to_spawn_dict()

    async def observe(self, *, tick: int = 0, name_suffix: Optional[str] = None, **kwargs: Any) -> ObservationFrame:
        perception = self._ensure_perception_module()
        capture_kwargs = dict(kwargs, get_agent_position=self.get_agent_position_and_forward, tick=tick)
        if self.modality == "vision":
            capture_kwargs["save_images"] = self.save_images
            capture_kwargs["name_suffix"] = name_suffix
        started = time.monotonic()
        frame = None
        event, error_type = "perception_capture_failed", None
        failure_details = {}
        try:
            frame = await perception.capture(self.agent_guid, **capture_kwargs)
            if self.modality == "text":
                self._targets.remember_entities(frame.entities)
            event = "perception_capture_completed"
            return frame
        except asyncio.CancelledError:
            event, error_type = "perception_capture_cancelled", "CancelledError"
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            from EmbodiedMAS.environment.rpc_diagnostics import error_details
            failure_details = error_details(exc)
            raise
        finally:
            completed = time.monotonic()
            # Diagnostic timing must not change capture results, errors or cancellation.
            try:
                from EmbodiedMAS.environment.rpc_diagnostics import clock_details
                record_event(self._context, "observations", {
                    **clock_details(self._context, at=completed), **failure_details,
                    "engine_agent_id": self._spawned.guid if self._spawned is not None else None,
                    "event": event, "valid": event == "perception_capture_completed",
                    "agent_id": self._spawned.name if self._spawned is not None else None,
                    "tick": tick, "action_id": kwargs.get("action_id"),
                    "scope": getattr(perception, "observation_scope", "local"),
                    "modality": self.modality,
                    "vision_backend": (getattr(frame, "vision_backend", None)
                        or getattr(getattr(perception, "backend", None), "name", None))
                        if self.modality == "vision" else None,
                    "view": kwargs.get("view", "front"),
                    "capture_kind": kwargs.get("capture_kind", "front"),
                    "requested_at_monotonic": started,
                    "completed_at_monotonic": completed,
                    "capture_elapsed_sec": max(0.0, completed - started),
                    "error_type": error_type,
                })
            except Exception:
                pass

    async def get_state(self) -> Optional[dict[str, Any]]:
        return await get_entity_state(self._context, self.agent_guid)

    async def get_self_state(self) -> Optional[dict[str, Any]]:
        """Read the simulator-reported following NPC IDs for follow confirmation."""
        return await ts.UnaryAPI.get_selfstate(self._conn, self.agent_guid)

    get_agent_position_and_forward = get_state

    async def _check_target_destroyed(self, target: str) -> None:
        if self.modality != "text":
            return
        raw_names = self._targets.raw_names_for(target)
        identities = self._targets.entity_ids_for(target)
        if not raw_names or not identities:
            return
        destroyed = await ts.UnaryAPI.get_destroyed_objects(self._conn)
        if not isinstance(destroyed, list) or not raw_names.issubset({
            name.strip() for name in destroyed if isinstance(name, str)
        }):
            return
        # Different rooms can reuse the same UE name. Confirm only the observed
        # GUIDs; another instance's destruction must not invalidate this target.
        registry = await ts.UnaryAPI.query_info(self._conn)
        states = {str(row["id"]).replace("-", "").casefold(): row.get("destroyed")
                  for row in registry if isinstance(row, dict) and row.get("id")} if isinstance(registry, list) else {}
        if identities.issubset(states) and all(states[key] is True for key in identities):
            raise DestroyedTargetError(target)

    async def resolve_entity_location(self, entity_name: str) -> Any:
        await self._check_target_destroyed(entity_name)
        if self.modality == "text":
            perception = self._ensure_perception_module()
            # Global decisions bind a complete current snapshot, potentially
            # from another CMAS worker. Older worker state must not restore it.
            object_list = ({} if getattr(perception, "observation_scope", "local") == "global"
                           else perception.merged_object_list())
        else:
            object_list = {}
        return self._targets.resolve(entity_name, object_list=object_list, semantic_maps=self._semantic_maps)

    async def world_locations_from_sos(self, sos: dict[str, Any]) -> Any:
        return await self._actions.world_locations_from_sos(self.agent_guid, sos)

    async def world_location_from_sos(self, sos: dict[str, Any]) -> Any:
        return await self._actions.world_location_from_sos(self.agent_guid, sos)

    async def navigate_entity(self, target: str) -> Any:
        try:
            location = await self.resolve_entity_location(target)
        except DestroyedTargetError as exc:
            return exc.result()
        return await self._actions.navigate_entity(self.agent_guid, location)

    def bind_navigation_target(self, location: Any = None) -> None:
        self._resolved_navigation_target = location

    async def navigate_resolved(self, location: Any) -> Any:
        """Execute an already resolved target without re-reading the pose."""
        if isinstance(location, (list, tuple)):
            location = ts.Vector3(*location)
        return await self._actions.navigate_entity(self.agent_guid, location)

    async def navigate_relative(self, distance: float, direction: str) -> Any:
        if self._resolved_navigation_target is not None:
            return await self.navigate_resolved(self._resolved_navigation_target)
        return await self._actions.navigate_relative(self.agent_guid, distance, direction)

    async def wait(self) -> Any:
        return await self._actions.wait(self.agent_guid)

    def validate_exploration_room(self, room_name: str) -> None:
        resolve_room_spec(self._semantic_maps, room_name)

    async def explore_room(self, room_name: str) -> Any:
        room, metadata = resolve_room_spec(self._semantic_maps, room_name)
        return await self._actions.explore_room(self.agent_guid, room, **metadata)

    def reset_action_progress(self, action_name: str) -> None:
        if action_name == "explore":
            self._actions.exploration_progress = {}

    def get_action_progress(self, action_name: str) -> dict[str, Any]:
        if action_name == "explore":
            return copy.deepcopy(getattr(self._actions, "exploration_progress", {}))
        return {}

    async def observe_surroundings(self, *, capture_view: Callable[[str], Awaitable[None]]) -> None:
        """Internal four-view sensing, independent of the public action space."""
        task = asyncio.current_task()
        if task is not None:
            self._execution_tasks.add(task)
        try:
            await self._actions.observe_surroundings(self.agent_guid, capture_view=capture_view)
        finally:
            self._execution_tasks.discard(task)

    async def instruct_civilian(self, subtype: str) -> Any:
        return await self._actions.instruct_civilian(self.agent_guid, subtype)

    async def suppress_fire(self, target: str | dict, actor: Any = None, *, timeout: float = 5.0) -> Any:
        if not self.supports_fire:
            raise NotImplementedError(f"Action 'suppress_fire' is not supported by {type(self).__name__}")
        try:
            target_location = (await self.resolve_image_location(target, purpose="suppress_fire")
                               if isinstance(target, dict) else await self.resolve_entity_location(target))
        except DestroyedTargetError as exc:
            return exc.result()
        target_actor, actor_id = self._get_actor_id(actor)
        if target_actor is None:
            return actor_id
        result = await self._actions.suppress_fire(actor_id=actor_id, target_location=target_location, timeout=timeout)
        return ("Success" if result else "Fail_CanotExtinguish") if isinstance(result, bool) else result

    def _get_actor_id(self, actor: Any) -> tuple[Any, Any]:
        target = actor if actor is not None else self._agent
        if target is None:
            return None, "Fail_NoAgent"
        actor_id = target.get("id") or target.get("guid") if isinstance(target, dict) else target
        if actor_id is None:
            return None, "Fail_InvalidActor"
        return target, actor_id

    def validate_action(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        """Validate before accepting a model decision; no body action runs here."""
        return self._executor.validate(action_payload)

    async def execute_action(self, action_payload: dict[str, Any]) -> Any:
        task = asyncio.current_task()
        if task is not None:
            self._execution_tasks.add(task)
        try:
            return await self._executor.execute(action_payload)
        finally:
            self._execution_tasks.discard(task)

    execute = execute_action

    async def stop(self) -> None:
        """Cancel this body's active Python calls; do not pause the shared scene."""
        current = asyncio.current_task()
        tasks = [task for task in self._execution_tasks if task is not current and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        await self.stop()
        for camera_id in tuple(self._owned_camera_ids):
            await ts.CaptureAPI.destroy_camera(self._conn, camera_id)
            self._owned_camera_ids.discard(camera_id)
        if self._obs_camera is not None and self._spawned is not None:
            self._obs_camera.unregister_agent_camera(self.agent_guid)
        self._agent_cameras.clear()
        self._spawn_camera_id = None

    async def destroy_spawned_actor(self) -> None:
        """Delete only the UE actor created by this body; bound actors are external.

        Runners call this after close() has stopped listeners/actions/cameras.
        Retain ownership on RPC failure so cleanup can be diagnosed or retried.
        """
        if self._owns_spawned_actor and self._spawned is not None:
            if await ts.UnaryAPI.destroy_actor(self._conn, self.agent_guid) is not True:
                raise RuntimeError("UE did not confirm owned actor cleanup")
            self._owns_spawned_actor = False
            self._spawned = None

    async def get_camera_info_and_convert(self, agent_id: Any = None) -> dict[str, Any]:
        target = agent_id if agent_id is not None else self._agent
        if target is None:
            raise ValueError("No agent_id provided and self._agent is None")
        return await self._actions.get_camera_info_and_convert(target)

    async def save_images(
        self,
        agent_id: bytes | str | dict | None = None,
        *,
        image_prefix: Optional[str] = None,
        image_format: str = "png",
        save_rgb: bool = True,
        save_depth: bool = False,
    ) -> dict[str, Any]:
        """
        保存 agent 的第一视角图像（RGB 和/或深度）到文件。
        通过调用 observation.py 中的 ObservationCamera 方法实现。
        
        Args:
            agent_id: Agent 的 ID（可以是 dict、str 或 bytes），如果为 None 则使用 self._agent
            image_prefix: 图像文件名前缀，如果为 None 则使用默认前缀
            image_format: RGB 图像格式，'png' 或 'jpg'（仅对 RGB 图像有效）
            save_rgb: 是否保存 RGB 图像
            save_depth: 是否保存深度图像
        
        Returns:
            包含 rgb 和 depth 路径；同时采集时还包含同帧 camera_metadata。未保存的路径为 None
            例如: {"rgb": "/path/to/image.png", "depth": "/path/to/depth.exr"}
        """
        if self._obs_camera is None:
            return {"rgb": None, "depth": None}
        agent = agent_id if agent_id is not None else self._agent
        if agent is None:
            return {"rgb": None, "depth": None}

        results: dict[str, Any] = {"rgb": None, "depth": None}

        if save_rgb and save_depth:
            results.update(
                await self._obs_camera.capture_rgb_and_depth(
                    agent,
                    image_prefix=image_prefix,
                    image_format=image_format,
                )
            )
        else:
            if save_rgb:
                results["rgb"] = await self._obs_camera.capture_rgb_image(
                    agent,
                    image_prefix=image_prefix,
                    image_format=image_format,
                )
            if save_depth:
                results["depth"] = await self._obs_camera.capture_depth_image(
                    agent,
                    image_prefix=image_prefix,
                )

        return results





class TextRobot(RobotAgent):
    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, modality="text", **kwargs)


class VisionRobot(RobotAgent):
    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None, **kwargs: Any):
        super().__init__(context, modality="vision", log_dir=log_dir, **kwargs)


class TextFireAgent(TextRobot):
    _blueprint = FireDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="FR", **kwargs)


class TextSaveAgent(TextRobot):
    _blueprint = SaveDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="RR", **kwargs)


class TextFirefighterAgent(TextRobot):
    _blueprint = Firefighter_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="HF", **kwargs)


class TextExploreAgent(TextRobot):
    _blueprint = ExploreDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="ER", **kwargs)


class VisionFireAgent(VisionRobot):
    _blueprint = FireDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="FR", **kwargs)


class VisionSaveAgent(VisionRobot):
    _blueprint = SaveDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="RR", **kwargs)


class VisionFirefighterAgent(VisionRobot):
    _blueprint = Firefighter_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="HF", **kwargs)


class VisionExploreAgent(VisionRobot):
    _blueprint = ExploreDog_BP

    def __init__(self, context: WorldContext, **kwargs: Any):
        super().__init__(context, agent_type="ER", **kwargs)


class LimitedWaterMixin:
    def __init__(self, context: WorldContext, *, water_capacity: int = 100, recover_time: int = 10, **kwargs: Any):
        super().__init__(context, **kwargs)
        self._water_capacity = water_capacity
        self._recover_time = recover_time

    async def spawn_agent(self, **kwargs: Any) -> Any:
        actor = await super().spawn_agent(**kwargs)
        if actor is None:
            return None
        ok = await self.reapply_extinguisher_config(timeout=kwargs.get("timeout", 5.0))
        if not ok:
            if self.modality == "vision":
                raise RuntimeError(f"set_extinguisher failed (water_capacity={self._water_capacity}, recover_time={self._recover_time})")
            print(f"[LimitedWaterFireAgent] Warning: set_extinguisher failed (water_capacity={self._water_capacity}, recover_time={self._recover_time})")
        return actor

    async def reapply_extinguisher_config(self, timeout: float = 5.0) -> bool:
        return bool(await ts.UnaryAPI.set_extinguisher(self._conn, self.agent_guid, self._water_capacity,
                                                     self._recover_time, timeout=timeout))


class TextLimitedWaterFireAgent(LimitedWaterMixin, TextFireAgent):
    pass


class VisionLimitedWaterFireAgent(LimitedWaterMixin, VisionFireAgent):
    pass


# Text aliases preserve the most common direct imports; vision shims select Vision*.
FireAgent = TextFireAgent
SaveAgent = TextSaveAgent
FirefighterAgent = TextFirefighterAgent
ExploreAgent = TextExploreAgent
LimitedWaterFireAgent = TextLimitedWaterFireAgent
