"""A CMAS robot's sensing/execution lifetime, without an individual LLM.

The central scheduler owns decisions. This worker owns exactly one robot action,
its periodic sensor, final observation and feedback. SA/DMAS keep their runtime.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from EmbodiedMAS.agent.actions.effects import fire_effect, follow_effect, navigation_effect
from EmbodiedMAS.agent.actions.geometry import direction_to_angle, NAV_TARGET_Z
from EmbodiedMAS.agent.actions.results import action_result, navigation_feedback, normalize_action_result
from EmbodiedMAS.agent.actions.image_targets import ImageTargetError
from EmbodiedMAS.agent.actions.targets import DestroyedTargetError, StaleVisualTargetError, validate_current_visual_target
from EmbodiedMAS.agent.config.environment_config import load_environment_config, perception_interval
from EmbodiedMAS.agent.factory import create_components
from EmbodiedMAS.agent.perception.adapters import _fire_state, _is_npc, _text_rooms
from EmbodiedMAS.agent.perception.oracle import names_for_context, room_for
from EmbodiedMAS.agent.perception.public_feedback import public_model_feedback
from EmbodiedMAS.agent.perception.records import DecisionObservation, ObservationFrame
from EmbodiedMAS.agent.perception.visual_distance import project_visual_entity
from EmbodiedMAS.agent.visual_backends import image_target_metadata
from EmbodiedMAS.agent.perception.seen_objects import stable_entity_id
from EmbodiedMAS.agent.prompt.src.config_loader import ConfigLoader
from EmbodiedMAS.agent.runtime.events import SOSListener, SOSLocationMemory, build_sos_trigger_context, public_sos_context
from EmbodiedMAS.agent.spatial_units import public_pose_m, world_xy_m
from EmbodiedMAS.environment.safety import npc_is_safe
from EmbodiedMAS.serialization import json_safe
from EmbodiedMAS.evaluation.monitor import record_event


class RobotWorker:
    supports_dispatch_guard = True

    def __init__(self, context: Any, *, robot_id: str, role: str,
                 modality: str = "text", observation_scope: str = "local",
                 record_mode: str | None = None, vision_backend: str | None = None,
                 log_dir: Path | None = None, on_event: Callable[[dict], None] | None = None,
                 body: Any = None, perception: Any = None) -> None:
        self.context, self.robot_id, self.role = context, robot_id, str(role).upper()
        self.modality, self.observation_scope = modality, observation_scope
        cfg = load_environment_config()
        self.record_mode = "full_snapshot"  # Complete latest boundaries, never action history.
        self.body, self.perception = create_components(context, agent_type=self.role, modality=modality,
            log_dir=log_dir, robot=body, perception=perception,
            interval_sec=perception_interval(modality, cfg), vision_backend=vision_backend or cfg["vision_backend"],
            observation_scope=observation_scope, record_mode=self.record_mode)
        self.agent = self.body  # Shared runner cleanup owns this body.
        self._maps = ConfigLoader().load_scenario_config("scenario_example.json").get("semantic_maps", [])
        self._names = names_for_context(context)
        self._safe_areas = copy.deepcopy(cfg["SafeArea"])
        self.body.set_semantic_maps(self._maps)
        self.body.bind_perception_module(self.perception)
        if hasattr(self.perception, "set_semantic_maps"):
            self.perception.set_semantic_maps(self._maps)
        self._timeout = float(cfg["perception_capture_timeout_sec"])
        self._interval = float(self.perception.interval_sec)
        if self._interval <= 0 or self._timeout <= 0:
            raise ValueError("Worker sensing interval and timeout must be positive")
        self._navigation_tolerance = cfg["navigation_tolerance_m"]
        self._on_event = on_event
        self.status = "idle"
        self.action_id = 0
        self.current_action: dict | None = None
        self.last_feedback: dict | None = None
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self._action_started: float | None = None
        self._action_started_active: float | None = None
        self._action_entered = False
        self._action_decision_step: int | None = None
        self._monitored_action_end_id = 0
        self._frames: list[ObservationFrame] = []
        self._public_frames: list[dict] = []
        self._last_boundary: list[dict] = []
        self._last_boundary_images: list[str] = []
        self._last_boundary_targets: dict[str, tuple] = {}
        self._last_boundary_entities: tuple[dict, ...] = ()
        self._last_boundary_known: list[dict] = []
        self._action_target_bindings: dict[str, tuple] = {}
        self._last_image_observation = None
        self._images: list[str] = []
        self._last_states: dict[str, tuple] = {}
        self._known: dict[tuple[str, str], dict] = {}
        self._targets: dict[str, tuple] = {}
        self._target_entities: dict[str, dict] = {}
        self._tick = self._batch = 0
        self._capture_lock = asyncio.Lock()
        self._transition_lock = asyncio.Lock()
        self._sampler: asyncio.Task | None = None
        self._sos: SOSListener | None = None
        self._sos_locations = SOSLocationMemory()
        self._closed = False
        self._stopping = False
        self._initialized = False
        self._position: dict | None = None
        self._follow_state: dict | None = None

    def _vision_pause_controller(self):
        if self.modality != "vision":
            return None
        controller = getattr(self.context, "_emas_vision_pause_controller", None)
        # Initial deployment/camera preparation precedes the episode clock.
        if controller is not None and getattr(controller, "started", True) is False:
            return None
        return controller

    async def _wait_vision_running(self) -> None:
        controller = self._vision_pause_controller()
        if controller is not None:
            await controller.wait_running()

    def _action_elapsed_sec(self, *, at_wall: float | None = None) -> float | None:
        if self._action_started is None:
            return None
        controller = self._vision_pause_controller()
        if controller is not None and self._action_started_active is not None:
            return max(0.0, controller.active_now() - self._action_started_active)
        now = time.monotonic() if at_wall is None else at_wall
        return max(0.0, now - self._action_started)

    async def _sensor_call(self, call, *args, timeout=None, **kwargs):
        """A sensor's business deadline excludes confirmed shared pauses."""
        limit = self._timeout if timeout is None else timeout
        controller = self._vision_pause_controller()
        if controller is None:
            return await asyncio.wait_for(call(*args, **kwargs), timeout=limit)
        await controller.wait_running()
        async with controller.timeout_active(limit):
            value = await call(*args, **kwargs)
            # A reply arriving while another model owns a pause is not ready
            # for publication or a new command until physical scene resume.
            await controller.wait_running()
            return value

    @property
    def on_event(self):
        return self._on_event

    @on_event.setter
    def on_event(self, callback):
        self._on_event = callback

    def _emit(self, kind: str, **details: Any) -> None:
        if self._on_event is not None:
            self._on_event({"type": kind, "robot_id": self.robot_id,
                            "captured_at": time.monotonic(), **copy.deepcopy(details)})

    async def deploy(self, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("Worker is closed")
        result = await self.body.deploy(**kwargs)
        if self._sos is None:
            self._sos = SOSListener(self.context, self.body.agent_guid, self._handle_sos,
                on_event=lambda event, details: self._emit("sos_transport", event=event, details=details))
            self._sos.start()
        return result

    async def _handle_sos(self, sos: dict) -> None:
        if self._closed or self._stopping:
            return
        accepted = copy.deepcopy(sos)
        orientations, distances = sos.get("orientations"), sos.get("distances")
        if isinstance(orientations, (list, tuple)) and orientations and isinstance(distances, (list, tuple)) and distances:
            try:
                estimates = await self._sensor_call(self.body.world_locations_from_sos, sos, timeout=1.0)
            except Exception:
                estimates = []
            keep = []
            for index in range(max(len(orientations), len(distances))):
                xy = world_xy_m(estimates[index]) if isinstance(estimates, list) and index < len(estimates) else None
                if xy is None or self._sos_locations.remember(xy):
                    keep.append(index)
            if not keep:
                return
            accepted.update({key: [sos[key][i] if i < len(sos[key]) else None for i in keep]
                             for key in ("orientations", "distances")})
        if not self._closed and not self._stopping:
            public = public_sos_context(build_sos_trigger_context(accepted), self._names)
            identity = {key: accepted.get(key) for key in ("orientations", "distances", "room", "name", "message", "direction")}
            signal_key = hashlib.sha256(json.dumps(json_safe(identity), sort_keys=True).encode()).hexdigest()
            self._emit("sos", action_id=self.action_id, context=public, _signal_key=signal_key)

    async def initialize(self) -> None:
        if self._closed or self._initialized:
            raise RuntimeError("Initialize a fresh worker once")
        await self._capture_boundary(reason="initial")
        await self._refresh_state()
        self._initialized = True
        self._start_sampler()

    def _start_sampler(self) -> None:
        if not self._stopping and not self._closed and (self._sampler is None or self._sampler.done()):
            self._sampler = asyncio.create_task(self._sample_loop(), name=f"cmas-sense-{self.robot_id}")

    async def _sample_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval)
                if self.status != "finalizing":
                    await self._wait_vision_running()
                    async with self._capture_lock:
                        await self._wait_vision_running()
                        if self.status != "finalizing":
                            await self._capture_frame()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit("worker_error", error_type=type(exc).__name__, phase="sensing")

    async def _capture_frame(self, *, view: str | None = None, batch_id: str | None = None,
                             boundary: bool = False) -> ObservationFrame:
        await self._wait_vision_running()
        self._tick += 1
        global_view = self.observation_scope == "global"
        kind = "global" if global_view else ("surround" if boundary else "front")
        frame = await self._sensor_call(self.body.observe, tick=self._tick,
            view="global" if global_view else view or "front", batch_id=batch_id,
            action_id=self.action_id, capture_kind=kind,
            name_suffix=f"_{self.robot_id}_{view or 'front'}")
        if self.modality == "vision":
            # Public target refs are qualified independently of annotation labels.
            def scoped_ref(ref: str) -> str:
                return ref if "::" in ref else f"{self.robot_id}::{ref}"

            targets = {scoped_ref(ref): xyz for ref, xyz in frame.targets.items()}
            entities = []
            for source in frame.entities:
                entity = copy.deepcopy(source)
                entity["target_ref"] = scoped_ref(entity["target_ref"])
                if entity.get("attached_to") is not None:
                    entity["attached_to"] = scoped_ref(entity["attached_to"])
                entities.append(entity)
            frame = replace(frame, entities=tuple(entities), targets=targets)
        # Periodic reads remain telemetry only, never controller input or bindings.
        if not boundary:
            return frame
        self._frames.append(frame)
        self._publish_frame(frame)
        return frame

    def _publish_frame(self, frame: ObservationFrame) -> None:
        if self.observation_scope == "global":
            self._targets.clear()
            self._target_entities.clear()
            self._known.clear()
        selected = []
        for entity in frame.entities:
            ref = str(entity.get("target_ref") or entity.get("name") or "")
            if ref:
                self._target_entities[ref] = {**copy.deepcopy(entity), "source_robot": self.robot_id,
                                              "captured_at_monotonic": frame.captured_at_monotonic}
            identity = stable_entity_id(entity)
            state = ((_fire_state(entity), npc_is_safe(entity.get("location"), self._safe_areas))
                     if _is_npc(entity) else (_fire_state(entity),))
            if self.modality == "text" and self.record_mode != "full_snapshot" and identity is not None:
                if self._last_states.get(identity) == state:
                    continue
                self._last_states[identity] = state
            selected.append(entity)
        self._targets.update(copy.deepcopy(frame.targets))
        public = {"robot_id": self.robot_id, "action_id": self.action_id, "tick": frame.tick,
                  "view": frame.view, "capture_kind": frame.capture_kind, "batch_id": frame.batch_id,
                  "captured_at": frame.captured_at_monotonic, "agent_position": json_safe(public_pose_m(frame.agent_position))}
        if self.modality == "text":
            public["rooms"] = _text_rooms(selected, self._safe_areas)
            if self.observation_scope == "global" and frame.global_metadata is not None:
                public["destroyed_entities"] = copy.deepcopy(frame.global_metadata.get("destroyed_entities", []))
            for room, members in public["rooms"].items():
                for group in ("objects", "npcs"):
                    for name, facts in members[group].items():
                        self._known[(group, name)] = {"name": name, "group": group, "room": room,
                            **copy.deepcopy(facts), "source_robot": self.robot_id,
                            "last_seen_action": self.action_id, "captured_at": frame.captured_at_monotonic}
        else:
            if frame.rgb_path is None:
                raise RuntimeError("Visual frame has no RGB image")
            public["entities"] = [project_visual_entity(
                entity, observer_uu=frame.agent_position,
                include_distance=frame.vision_backend == "grounded_sam_pixel")
                for entity in frame.entities]
            public["visual_images"] = [{"path": path, "image_role": role,
                                        "annotation_available": len(frame.image_paths) > 1,
                                        **image_target_metadata(frame, actionable=
                                            role == "raw" and frame.capture_kind == "surround")}
                                       for role, path in zip(("raw", "annotated"), frame.image_paths)]
            self._images.extend(frame.image_paths)
            if getattr(frame, "vision_backend", None) == "raw_vlm":
                public.pop("entities", None)
            public.update(image_target_metadata(frame, actionable=frame.capture_kind == "surround"))
        self._public_frames.append(public)

    async def _capture_boundary(self, *, reason: str) -> DecisionObservation:
        await self._wait_vision_running()
        self._batch += 1
        batch_id = f"{self.robot_id}:boundary:{self._batch}"
        frames = []
        async with self._capture_lock:
            await self._wait_vision_running()
            self._frames, self._public_frames, self._images = [], [], []
            self._targets, self._target_entities, self._known = {}, {}, {}
            self._last_states.clear()
            async def capture(view: str) -> None:
                frames.append(await self._capture_frame(view=view, batch_id=batch_id, boundary=True))
            if self.observation_scope == "global":
                await capture("global")
            else:
                await self.body.observe_surroundings(capture_view=capture)
            self._position = await self._sensor_call(self.body.get_agent_position_and_forward)
            if not isinstance(self._position, dict):
                raise RuntimeError("Robot state unavailable after boundary observation")
            self._last_boundary = copy.deepcopy(self._public_frames[-len(frames):])
            self._last_boundary_images = [path for frame in frames for path in frame.image_paths]
            self._last_boundary_targets = copy.deepcopy(self._targets)
            self._last_boundary_entities = tuple(copy.deepcopy(list(self._target_entities.values())))
            self._last_boundary_known = copy.deepcopy(list(self._known.values()))
        observation = DecisionObservation(batch_id=batch_id, reason=reason, action_id=self.action_id,
                                   surroundings=tuple(frames), agent_position=copy.deepcopy(self._position),
                                   scope=self.observation_scope)
        if getattr(self.body, "supports_image_targets", False):
            self._last_image_observation = observation
        return observation

    async def _read_follow_state(self) -> dict | None:
        try:
            value = await self._sensor_call(self.body.get_self_state)
        except Exception:
            return None
        return copy.deepcopy(value) if isinstance(value, dict) else None

    async def _refresh_state(self) -> None:
        position = await self._sensor_call(self.body.get_agent_position_and_forward)
        self._position = copy.deepcopy(position) if isinstance(position, dict) else None
        self._follow_state = await self._read_follow_state()

    async def snapshot(self) -> dict:
        await self._refresh_state()
        position = self._position or {}
        forward = json_safe(position.get("forward"))
        if isinstance(forward, dict):
            forward = [forward.get(axis) for axis in ("x", "y", "z")]
        state = {"robot_id": self.robot_id, "role": self.role,
                 "location": world_xy_m(position), "orientation": forward,
                 "room": room_for(position, self._maps),
                 "following_npcs": self._names.following_names(self._follow_state),
                 "resource_state": {}, "execution_status": self.status,
                 "is_executing": self.status == "executing", "action_id": self.action_id,
                 "current_action": copy.deepcopy(self.current_action) if self.status != "idle" else None,
                 "action_elapsed_sec": (None if self.status == "idle" else self._action_elapsed_sec()),
                 "last_execution_feedback": copy.deepcopy(self.last_feedback),
                 "captured_at": time.monotonic()}
        snapshot = {"state": state, "observation": {"observation_scope": self.observation_scope,
                "entity_record_mode": self.record_mode, "frames": copy.deepcopy(self._last_boundary)},
                **({"known_objects": copy.deepcopy(self._last_boundary_known)}
                   if self.observation_scope == "local" else {}),
                "image_paths": list(self._last_boundary_images),
                "target_bindings": self.target_snapshot()}
        if self.observation_scope == "global":
            # The latest complete snapshot also defines the action bindings.
            snapshot["observation"]["frames"] = snapshot["observation"]["frames"][-1:]
        if getattr(self.body, "supports_image_targets", False):
            # Geometry is execution-private; the central payload only receives observation/state.
            snapshot["image_observation"] = self._last_image_observation
        return snapshot

    def target_snapshot(self) -> tuple[dict, tuple]:
        return copy.deepcopy(self._last_boundary_targets), copy.deepcopy(self._last_boundary_entities)

    async def _navigation_target(self, action: dict) -> Any:
        if action["subtype"] == "entity":
            return await self.body.resolve_entity_location(action["parameters"]["target"])
        if action["subtype"] == "image":
            return await self.body.resolve_image_location(action["parameters"]["target"], purpose="navigate")
        pose = await self.body.get_agent_position_and_forward()
        forward = pose["forward"]
        fx, fy = ((forward["x"], forward["y"]) if isinstance(forward, dict) else (forward.x, forward.y))
        params = action["parameters"]
        angle = math.atan2(fy, fx) + math.radians(direction_to_angle(params["direction"]))
        distance = float(params["distance"]) * 100.0
        return pose["x"] + distance * math.cos(angle), pose["y"] + distance * math.sin(angle), NAV_TARGET_Z

    async def start_action(self, action: dict, *, decision_step: int,
                           targets: dict | None = None, entities: tuple = (),
                           image_observation: DecisionObservation | None = None,
                           current_target_refs: Callable[[], set[str]] | None = None,
                           dispatch_allowed: Callable[[], bool] | None = None) -> bool | None:
        normalized = self.body.validate_action(action)
        await self._wait_vision_running()
        async with self._transition_lock:
            await self._wait_vision_running()
            if self._closed or self._stopping:
                raise RuntimeError("Worker is stopped")
            if self.task is not None and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self._record_cancelled_before_execution()
            # Finish an in-flight periodic capture before changing action ownership.
            await self._wait_vision_running()
            async with self._capture_lock:
                await self._wait_vision_running()
                if dispatch_allowed is not None and not dispatch_allowed():
                    return False
                self._action_target_bindings = copy.deepcopy(targets if targets is not None else self.target_snapshot()[0])
                if self.modality == "vision":
                    # Cancelling the previous action can itself capture a newer
                    # boundary. Check freshness after that await, not before it.
                    valid_refs = (current_target_refs() if current_target_refs is not None
                                  else set(self.target_snapshot()[0]))
                    self._action_target_bindings = {ref: xyz for ref, xyz in self._action_target_bindings.items()
                                                    if ref in valid_refs}
                    entities = tuple(e for e in entities if e.get("target_ref") in valid_refs)
                self.body.bind_targets(self._action_target_bindings, entities=entities)
                if getattr(self.body, "supports_image_targets", False):
                    # Only the latest complete boundary is actionable. Old
                    # frame_ref values receive ImageTargetError, never new geometry.
                    self.body.bind_image_observation(self._last_image_observation)
                self.action_id += 1
                self.current_action = copy.deepcopy(normalized)
                self._action_started = time.monotonic()
                controller = self._vision_pause_controller()
                self._action_started_active = None if controller is None else controller.active_now()
                self._action_entered = False
                self._action_decision_step = decision_step
                record_event(self.context, "actions", {"event": "action_started",
                    "agent_id": self.robot_id, "action_id": self.action_id,
                    "decision_step": decision_step, "action": normalized,
                    "started_at_monotonic": self._action_started})
                self.status = "executing"
                self._frames, self._public_frames, self._images = [], [], []
                self._last_states.clear()
                self.task = asyncio.create_task(self._run_action(normalized, decision_step=decision_step),
                                                 name=f"cmas-action-{self.robot_id}-{self.action_id}")

    def _save_feedback(self, action: dict, result: dict, decision_step: int, *,
                       phase: str = "finalized") -> None:
        params = copy.deepcopy(action["parameters"])
        if "subtype" in action:
            params["subtype"] = action["subtype"]
        self.last_feedback = public_model_feedback({"action_id": self.action_id, "action": action["name"],
                                                    "action_params": params, **result}, self._names)
        self.history.append({"robot_id": self.robot_id, "action_id": self.action_id,
                             "decision_step": decision_step, "action": copy.deepcopy(action),
                             "result": copy.deepcopy(self.last_feedback), "finished_at": time.monotonic()})
        self._emit("action_result", action_id=self.action_id, decision_step=decision_step,
                   action=action, feedback=self.last_feedback)
        finished = time.monotonic()
        timing = {"elapsed_sec": self._action_elapsed_sec(at_wall=finished)}
        if self._vision_pause_controller() is not None and self._action_started_active is not None:
            timing["wall_action_elapsed_sec"] = max(0.0, finished - self._action_started)
        record_event(self.context, "actions", {"event": "action_finished", "phase": phase,
            "agent_id": self.robot_id, "action_id": self.action_id, "decision_step": decision_step,
            "action": action, "result": copy.deepcopy(result),
            "started_at_monotonic": self._action_started, "finished_at_monotonic": finished,
            **timing})
        self._monitored_action_end_id = self.action_id

    def _record_cancelled_before_execution(self) -> None:
        """A task cancelled before its coroutine runs cannot publish its own end."""
        if (self.task is None or not self.task.cancelled() or self._action_entered
                or self.current_action is None or self._monitored_action_end_id == self.action_id):
            return
        finished = time.monotonic()
        fields = {"agent_id": self.robot_id, "action_id": self.action_id,
            "decision_step": self._action_decision_step, "action": copy.deepcopy(self.current_action),
            "result": action_result("interrupted", "cancelled_before_execution", {"execution_started": False})}
        record_event(self.context, "actions", {**fields, "event": "action_result", "phase": "execution",
            "captured_at_monotonic": finished})
        # Publish the same public feedback as an entered coroutine would, so
        # central memory can finish this accepted decision without inventing a
        # physical action. _save_feedback records the sole completion event.
        self._save_feedback(self.current_action, fields["result"], self._action_decision_step,
                            phase="cancelled_before_execution")

    async def _run_action(self, action: dict, *, decision_step: int) -> None:
        self._action_entered = True
        target, previous_follow, result = None, None, None
        pinned = False
        cancelled = False
        execution_returned = False
        try:
            await self._wait_vision_running()
            if self.modality == "vision":
                validate_current_visual_target(action, self._action_target_bindings, semantic_maps=self._maps)
            reset = getattr(self.body, "reset_action_progress", None)
            if callable(reset):
                reset(action["name"])
            if action["name"] == "navigate":
                target = await self._navigation_target(action)
                if action["subtype"] == "relative":
                    self.body.bind_navigation_target(target)
                    pinned = True
            if action["name"] == "instruct_civilian" and action["subtype"] == "follow":
                previous_follow = await self._read_follow_state()
            await self._wait_vision_running()
            raw = await self.body.execute_action(action)
            result = normalize_action_result(action["name"], raw)
            execution_returned = True
        except asyncio.CancelledError:
            cancelled = True
            result = action_result("interrupted", "cancelled", {})
        except DestroyedTargetError as exc:
            result = exc.result()
        except StaleVisualTargetError as exc:
            result = action_result("failed", exc.code, {"message": exc.public_message,
                                                        "execution_started": False})
        except ImageTargetError as exc:
            result = action_result("failed", exc.code, {"message": exc.public_message,
                                                        "execution_started": False})
        except (ValueError, TimeoutError) as exc:
            result = action_result("failed", "action_timeout" if isinstance(exc, TimeoutError) else "invalid_target",
                                   {"error_type": type(exc).__name__, "error": str(exc)})
        except Exception as exc:
            result = action_result("failed", "execution_error",
                                   {"error_type": type(exc).__name__, "error": str(exc)})
            self._emit("worker_error", error_type=type(exc).__name__, phase="execution")
        finally:
            if pinned:
                self.body.bind_navigation_target()
        record_event(self.context, "actions", {"event": "action_result", "phase": "execution",
            "agent_id": self.robot_id, "action_id": self.action_id, "decision_step": decision_step,
            "action": action, "result": copy.deepcopy(result),
            "captured_at_monotonic": time.monotonic()})
        try:
            if not self._stopping and not self._closed:
                self.status = "finalizing"
                observation = await self._capture_boundary(reason="interrupted" if cancelled else "action_end")
                await self._refresh_state()
                if not cancelled and execution_returned and action["name"] == "navigate":
                    result = navigation_feedback(result,
                        navigation_effect(observation.agent_position, target, self._navigation_tolerance))
                elif not cancelled and execution_returned and result.get("status") not in {"failed", "interrupted"}:
                    if action["name"] == "suppress_fire":
                        result = fire_effect(observation, action["parameters"]["target"], modality=self.modality)
                    elif action["name"] == "instruct_civilian" and action["subtype"] == "follow":
                        result = follow_effect(self._follow_state, previous_state=previous_follow)
                    elif action["name"] == "instruct_civilian":
                        receipt = result["data"].get("execution_result")
                        if type(receipt) is bool:
                            result = action_result("completed" if receipt else "failed",
                                "stop_following_succeeded" if receipt else "stop_following_failed",
                                {"execution_result": receipt, "effect_success": receipt})
                            if receipt:
                                self._follow_state = {"following_NPC": {}}
            self._save_feedback(action, result, decision_step)
            self.status = "idle"
            if not self._stopping and not self._closed:
                self._emit("action_finished", action_id=self.action_id, feedback=self.last_feedback)
        except asyncio.CancelledError:
            # Reassignment can cancel sensing after a physical command returned.
            # Preserve its execution verdict; only effect verification is missing.
            final = copy.deepcopy(result) if execution_returned else action_result("interrupted", "cancelled", {})
            final["data"].update(verification="finalization_interrupted")
            self._save_feedback(action, final, decision_step)
            self.status = "idle"
            raise
        except Exception as exc:
            self._save_feedback(action, result or action_result("failed", "execution_error", {}), decision_step)
            self.status = "idle"
            self._emit("worker_error", error_type=type(exc).__name__, phase="finalizing")

    async def stop(self) -> None:
        self._stopping = True
        tasks = [task for task in (self.task, self._sampler) if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._record_cancelled_before_execution()
        self.status = "idle"
        await self.body.stop()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.stop()
        finally:
            try:
                if self._sos is not None:
                    await self._sos.stop()
            finally:
                await self.body.close()
