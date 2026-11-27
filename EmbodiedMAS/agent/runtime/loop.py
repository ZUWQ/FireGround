"""Shared single-agent lifecycle, decision, execution and event orchestration.

Model calls and cognitive memory intentionally remain in BaseLLMAgent. Bodies,
perception and per-modality observation adapters are composed at construction.
"""
from __future__ import annotations

import asyncio
import copy
import math
import inspect
import time
from pathlib import Path
from typing import Any, Optional

from EmbodiedMAS.agent.base_llm_agent import BaseLLMAgent
from EmbodiedMAS.agent.decision_output import InvalidDecisionOutput, PROTOCOL_VERSION
from EmbodiedMAS.agent.config.environment_config import load_environment_config, perception_interval
from EmbodiedMAS.agent.config.llm_config import configured_model_name
from EmbodiedMAS.agent.factory import create_components
from EmbodiedMAS.agent.perception.adapters import create_observation_adapter
from EmbodiedMAS.agent.perception.seen_objects import SeenObjects
from EmbodiedMAS.agent.perception.oracle import names_for_context, room_for
from EmbodiedMAS.agent.perception.public_feedback import public_model_feedback
from EmbodiedMAS.agent.perception.records import DecisionObservation, ObservationFrame
from EmbodiedMAS.agent.spatial_units import world_xy_m
from EmbodiedMAS.agent.prompt.emas_prompt import build_runtime_input
from EmbodiedMAS.agent.prompt.src.config_loader import ConfigLoader
from EmbodiedMAS.serialization import json_safe
from EmbodiedMAS.agent.runtime.control import ControlModule
from EmbodiedMAS.agent.runtime.context_logging import DecisionContextLogger
from EmbodiedMAS.agent.actions.effects import fire_effect, follow_effect, navigation_effect
from EmbodiedMAS.agent.actions.geometry import direction_to_angle, NAV_TARGET_Z
from EmbodiedMAS.agent.runtime.messages import normalize_message
from EmbodiedMAS.agent.actions.results import action_result, navigation_feedback, normalize_action_result
from EmbodiedMAS.agent.runtime.decision import DecisionModule
from EmbodiedMAS.agent.runtime.events import SOSListener, SOSLocationMemory, build_sos_trigger_context, public_sos_context
from EmbodiedMAS.agent.runtime.timing import StepTimingCache, track
from EmbodiedMAS.agent.runtime.t1_pause import T1PauseBudgetExhausted
from EmbodiedMAS.agent.runtime.vision_pause import VisionPauseBudgetExhausted
from EmbodiedMAS.agent.runtime import logging as session_logging
from EmbodiedMAS.evaluation.monitor import record_event
from EmbodiedMAS.agent.actions.image_targets import ImageTargetError
from EmbodiedMAS.agent.actions.targets import DestroyedTargetError, StaleVisualTargetError, validate_current_visual_target


SOS_LOCATION_ESTIMATE_TIMEOUT_SEC = 1.0


class AgentRuntime(BaseLLMAgent):
    """One shared loop for text and vision agents, retaining legacy public hooks."""

    def __init__(
        self, context: Any, *, agent_type: str, modality: str,
        log_dir: Optional[Path] = None, robot: Any = None, perception: Any = None,
        vision_backend: Optional[str] = None,
        observation_scope: str | None = None,
        record_mode: str | None = None,
        cooperation_type: str | None = None,
        time_limit_sec: float | None = None,
        message_sender: Any = None,
        episode_briefing: dict[str, Any] | None = None,
        vision_grace_pause: bool | None = None,
    ) -> None:
        runtime_cfg = load_environment_config()
        scope = observation_scope or runtime_cfg.get("observation_scope", "local")
        mode = record_mode or runtime_cfg.get("record_mode") or (
            "full_snapshot" if scope == "global" or modality == "vision"
            else "action_first_seen_or_fire_state_change")
        cooperation = str(cooperation_type or runtime_cfg.get("coordination_mode", "none")).strip()
        cooperation = "none" if cooperation.lower() == "none" else cooperation.upper()
        limit = time_limit_sec if time_limit_sec is not None else runtime_cfg.get("time_limit_sec")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
                raise ValueError("time_limit_sec must be a positive finite number")
            limit = float(limit)
        if runtime_cfg.get("pause_during_llm", False) and not (
                limit is not None and getattr(context, "_emas_t1_pause_authorized", False)
                and modality == "text" and scope == "global" and agent_type == "FR"
                and cooperation == "none"):
            raise ValueError("Timed model pause requires an explicitly validated T1 task")
        from .vision_protocol import bind_controller
        vision_enabled = (runtime_cfg.get("vision_grace_pause", False)
                          if vision_grace_pause is None else vision_grace_pause)
        self._vision_pause_controller = bind_controller(
            context, enabled=vision_enabled, mode=modality, scope=scope, duration=limit,
            pause_during_llm=runtime_cfg.get("pause_during_llm", False))
        body, perception = create_components(
            context, agent_type=agent_type, modality=modality,
            log_dir=log_dir, robot=robot, perception=perception,
            interval_sec=perception_interval(modality, runtime_cfg),
            vision_backend=vision_backend or runtime_cfg.get("vision_backend", "grounded_sam"),
            observation_scope=scope, record_mode=mode,
        )
        brief = dict(runtime_cfg["prompt_environment"])
        brief.update(episode_briefing or {})
        brief["time_limit_sec"] = limit
        super().__init__(context=context, agent=body, prompt_type=agent_type,
                         input_modality=modality, cooperation_type=cooperation,
                         observation_scope=scope, record_mode=mode, episode_briefing=brief,
                         vision_backend=vision_backend or runtime_cfg.get("vision_backend", "grounded_sam"))
        self._communication_enabled = brief.get("communication_enabled", True)
        if type(self._communication_enabled) is not bool:
            raise ValueError("communication_enabled must be boolean")
        if not self._communication_enabled:
            self._system_prompt += "\nExplicit teammate communication is disabled for this episode. Return communication: null. Do not send or forward teammate messages. Your observations, environmental SOS, and action feedback remain available.\n"
        self._time_limit_sec = limit
        self._episode_started_at: float | None = None
        self._episode_deadline: float | None = None
        self.end_reason: str | None = None
        self._active_decision_timeout = None
        self._t1_pause_controller = None
        if runtime_cfg.get("pause_during_llm", False) and limit is not None:
            from .t1_pause import T1PauseController
            self._t1_pause_controller = T1PauseController(self)
            context._emas_t1_pause_controller = self._t1_pause_controller
        self._message_sender = message_sender
        self._outgoing_messages: list[dict[str, Any]] = []
        self._transport_lock = asyncio.Lock()
        self.last_transport_error: str | None = None
        self._outgoing_sequence = 0
        self._super_agent_assignment: dict[str, Any] | None = None
        maps = ConfigLoader().load_scenario_config("scenario_example.json").get("semantic_maps", [])
        body.set_semantic_maps(maps)
        self._semantic_maps = maps
        self._oracle_names = names_for_context(context)
        if hasattr(perception, "set_semantic_maps"):
            perception.set_semantic_maps(maps)
        body.bind_perception_module(perception)
        self._perception = perception
        self._observation_adapter = create_observation_adapter(
            modality, observation_scope=scope, record_mode=mode)
        self._seen_objects = SeenObjects(modality)
        self._navigation_tolerance_m = runtime_cfg["navigation_tolerance_m"]
        self._pending_effect: dict[str, Any] | None = None
        from .vision_feedback import PROTOCOL, VisionActionFeedback
        self._vision_action_feedback = (
            VisionActionFeedback() if self._vision_feedback_protocol == PROTOCOL else None)
        self._monitored_action: dict[str, Any] | None = None
        self._follow_state_batch: str | None = None
        self._follow_state: dict[str, Any] | None = None
        self._message_inbox: list[dict[str, Any]] = []
        self._decision_messages: list[dict[str, Any]] = []
        self._decision = DecisionModule()
        self._context_logger: DecisionContextLogger | None = None
        self._control = ControlModule()
        self.timing = StepTimingCache()
        self._decision_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._perception_tick = 0
        self._current_agent_idx: Optional[int] = None
        self._sos_lifecycle_lock = asyncio.Lock()
        self._sos_listener: Optional[SOSListener] = None
        self._sos_guid: Optional[str] = None
        self._sos_locations = SOSLocationMemory()
        self._sos_decision_in_flight = False
        self._queued_interrupts: list[dict[str, Any]] = []
        self._event_revision = 0
        self._batch_counter = 0
        self._ready_observation: DecisionObservation | None = None
        self._prepared_observation: DecisionObservation | None = None
        self._prepared_payload: dict[str, Any] | None = None
        self._pending_boundary: dict[str, Any] | None = None
        self._normal_run_completed = False
        self._capture_lock = asyncio.Lock()
        self._capture_timeout = float(runtime_cfg["perception_capture_timeout_sec"])
        if self._perception.interval_sec <= 0 or self._capture_timeout <= 0:
            raise ValueError("Perception interval and capture timeout must be positive")
        self._run_active = False
        self._accept_interrupts = False
        self._run_task: Optional[asyncio.Task[Any]] = None
        self._closed = False
        if hasattr(body, "add_deploy_listener"):
            body.add_deploy_listener(self._on_body_deployed)

    def _build_runtime_input_package(
        self,
        *,
        agent_position: Optional[dict[str, Any]],
        current_observation: dict[str, Any],
        other_agent_info: Optional[str],
        agent_idx: Optional[int],
    ) -> dict[str, Any]:
        package = build_runtime_input(
            step=self._log_round,
            agent_id=self.agent.agent_name,
            agent_state=self._build_emas_agent_state(agent_position),
            current_observation=current_observation,
            teammate_messages=(self._teammate_messages(other_agent_info) + copy.deepcopy(self._decision_messages)
                               if self._cooperation_type != "none" and getattr(self, "_communication_enabled", True) else []),
            super_agent_assignment=copy.deepcopy(self._super_agent_assignment),
            memory_state={
                "note": self._episode_memory.note,
                "note_max_chars": self._episode_memory.note_max_chars,
                "short_term": public_model_feedback(json_safe(self._memory), self._oracle_names),
                **({"known_objects": self._seen_objects.snapshot()}
                   if self._input_modality == "text" and self._observation_scope == "local" else {}),
            },
            previous_execution_feedback=public_model_feedback(self._previous_execution_feedback, self._oracle_names),
            previous_decision_feedback=copy.deepcopy(self._previous_decision_feedback),
        )
        package["task_time"] = self.task_time
        return package

    def record_decision_input(self, runtime_input: dict[str, Any]) -> None:
        """Audit the exact input and retain its received-message snapshot."""
        self._received_decision_messages = copy.deepcopy(
            runtime_input.get("signal_channel", {}).get("teammate_messages", []))
        if self._session_id is None:
            self._session_id = session_logging.ensure_session_id(None)
        if self._context_logger is None:
            self._context_logger = DecisionContextLogger(session_logging.resolve_benchmark_log_dir(),
                                                        self.agent.agent_name, self._session_id)
        config = dict(load_environment_config())
        config.update(agent_type=self._agent_type, modality=self._input_modality,
                      observation_scope=self._observation_scope, record_mode=self._record_mode,
                      cooperation_type=self._cooperation_type, time_limit_sec=self._time_limit_sec,
                      max_tokens=self._max_tokens,
                      perception_interval_sec=self._perception.interval_sec)
        model = self._llm_config.get_model() if self._llm_config is not None else configured_model_name()
        self._context_logger.record_input(runtime_input, system_prompt=self._system_prompt,
                                          model=model, config=config)

    def record_decision_outcome(self, status: str, *, reason: str | None = None,
                                **details: Any) -> None:
        if self._context_logger is not None:
            self._context_logger.record_outcome(self._log_round, status, reason=reason, **details)

    @property
    def task_time(self) -> dict[str, Any]:
        controller = (getattr(self, "_vision_pause_controller", None)
                      or getattr(self, "_t1_pause_controller", None))
        if controller is not None:
            state = controller.snapshot()
            return {"elapsed_sec": state["elapsed_sec"], "limit_sec": self._time_limit_sec,
                    "remaining_sec": state["remaining_sec"], "end_reason": self.end_reason}
        elapsed = 0.0 if self._episode_started_at is None else max(0.0, time.monotonic() - self._episode_started_at)
        return {"elapsed_sec": elapsed, "limit_sec": self._time_limit_sec,
                "remaining_sec": (None if self._episode_deadline is None
                                  else max(0.0, self._episode_deadline - time.monotonic())),
                "end_reason": self.end_reason}

    def _budget_remaining(self):
        if self._vision_pause_controller is not None:
            return self._vision_pause_controller.snapshot()['remaining_sec']
        return None if self._episode_deadline is None else self._episode_deadline - time.monotonic()

    def _budget_expired(self) -> bool:
        remaining = self._budget_remaining()
        return remaining is not None and remaining <= 0

    async def _wait_vision_running(self):
        if self._vision_pause_controller is not None:
            await self._vision_pause_controller.wait_running()

    @property
    def outgoing_messages(self) -> list[dict[str, Any]]:
        """Messages awaiting transport; presence does not assert delivery."""
        return copy.deepcopy(self._outgoing_messages)

    async def flush_outgoing_messages(self) -> None:
        if self._message_sender is None:
            return
        await self._wait_vision_running()
        async with self._transport_lock:
            while self._outgoing_messages:
                try:
                    result = self._message_sender(copy.deepcopy(self._outgoing_messages[0]))
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    # Transport failure must not suppress an accepted rescue action.
                    # Leave the message queued for an explicit later retry.
                    self.last_transport_error = type(exc).__name__
                    return
                self._outgoing_messages.pop(0)
                self.last_transport_error = None



    def start_episode(self, *, started_at: float | None = None) -> None:
        """Bind a fresh runtime to a runner's monotonic episode start.

        Team runners call this after deployment/transport registration. Reusing
        a runtime cannot reset its original clock or extend its deadline.
        """
        if self._closed or self._run_active:
            raise RuntimeError("Episode clock requires a fresh, idle runtime")
        start = time.monotonic() if started_at is None else started_at
        if (isinstance(start, bool) or not isinstance(start, (int, float))
                or not math.isfinite(start) or start > time.monotonic()):
            raise ValueError("started_at must be a finite monotonic time at or before now")
        if self._episode_started_at is not None:
            if start != self._episode_started_at:
                raise RuntimeError("Episode clock is already bound")
            return
        self._episode_started_at = float(start)
        if self._time_limit_sec is not None:
            self._episode_deadline = self._episode_started_at + self._time_limit_sec
        if self._vision_pause_controller is not None:
            self._vision_pause_controller.register_runtime(self)
            self._vision_pause_controller.start(self._episode_started_at, self._time_limit_sec)

    async def run(self, *, max_decisions: int | None = None, agent_idx: Optional[int] = 0) -> None:
        """Run one continuous episode; an optional decision cap is for debugging.

        The episode clock starts before initial sensing and keeps running across
        explicit retries and bounded calls. Deadline cancellation never accepts
        a late model answer or starts another action.
        """
        if self._closed:
            raise RuntimeError("Cannot run a closed agent")
        if self._run_active:
            raise RuntimeError("This agent already has an active run")
        if max_decisions is not None and (isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions < 0):
            raise ValueError("max_decisions must be a nonnegative integer or None")
        if max_decisions is None and self._time_limit_sec is None:
            raise ValueError("run requires a task time limit or decision cap")
        if self.end_reason == "time_limit":
            return
        if max_decisions != 0 and self._episode_started_at is None and self._vision_pause_controller is not None:
            self.start_episode(started_at=time.monotonic())
        self._current_agent_idx = agent_idx
        self._run_active = self._accept_interrupts = True
        self._run_task = asyncio.current_task()
        clean_run = False
        try:
            await self.start_sos_listener()
            if max_decisions == 0:
                clean_run = True
                return
            if self._episode_started_at is None:
                self._episode_started_at = time.monotonic()
                if self._time_limit_sec is not None:
                    self._episode_deadline = self._episode_started_at + self._time_limit_sec
            if self._normal_run_completed:
                self._ready_observation = None
            self._normal_run_completed = False
            self.end_reason = None
            attempts = 0
            while max_decisions is None or attempts < max_decisions:
                remaining = self._budget_remaining()
                if remaining is not None and remaining <= 0:
                    self.end_reason = "time_limit"
                    break
                deadline = (self._vision_pause_controller.timeout_active(remaining)
                            if self._vision_pause_controller is not None else asyncio.timeout(remaining))
                try:
                    async with deadline as timer:
                        self._active_decision_timeout = timer
                        await self._call_decision_and_control("decision", agent_idx=agent_idx)
                except (T1PauseBudgetExhausted, VisionPauseBudgetExhausted):
                    self.end_reason = "time_limit"
                    break
                except TimeoutError:
                    if not timer.expired():
                        raise
                    self.end_reason = "time_limit"
                    break
                finally:
                    self._active_decision_timeout = None
                attempts += 1
                if self._budget_expired():
                    self.end_reason = "time_limit"
                    break
                self.flush_session_log(log_round=self._log_round)
            if self.end_reason != "time_limit":
                self.end_reason = "decision_limit"
                self._normal_run_completed = True
            else:
                self._ready_observation = None
                self._pending_boundary = None
                self._pending_effect = None
                self.flush_session_log(log_round=self._log_round)
            clean_run = True
        finally:
            self._accept_interrupts = False
            try:
                # Shared normal deadlines settle independently. Closing here would
                # cancel a teammate before its own timeout can become time_limit.
                # The team driver closes after all normal budget exits.
                if self._vision_pause_controller is not None and not clean_run:
                    await self._vision_pause_controller.close()
            finally:
                await self._finish_run_cleanup()

    async def _finish_run_cleanup(self) -> None:
        try:
            await self._stop_background_work()
        finally:
            self._run_active = False
            self._run_task = None

    async def deploy(self, **kwargs: Any) -> Any:
        """Deploy the body and automatically activate lifetime SOS listening."""
        if self._closed:
            raise RuntimeError("Cannot deploy a closed agent")
        result = await self.agent.deploy_agent(**kwargs)
        # The body hook handles direct body deployments; this fallback also
        # supports an injected body implementing only the public deploy API.
        await self.start_sos_listener()
        return result

    deploy_agent = deploy

    async def _on_body_deployed(self, body: Any) -> None:
        if not self._closed:
            await self.start_sos_listener()

    async def start_sos_listener(self) -> None:
        """Ensure exactly one listener for this deployed body, until close()."""
        if self._closed:
            raise RuntimeError("Cannot start a listener on a closed agent")
        async with self._sos_lifecycle_lock:
            # close() can acquire ownership while a redeployment is waiting.
            if self._closed or not self.agent.is_spawned:
                return
            guid = self.agent.agent_guid
            if self._sos_listener is not None and self._sos_guid != guid:
                await self._sos_listener.stop()
                self._sos_listener = None
                self._sos_locations.clear()
                if self._closed:
                    return
            if self._sos_listener is None:
                self._sos_listener = SOSListener(
                    self.context, guid, on_sos=self._handle_sos, on_event=self._record_sos_event,
                )
                self._sos_guid = guid
            self._sos_listener.start()

    def request_interrupt(
        self, trigger: str, *, context: Optional[dict[str, Any]] = None,
        other_agent_info: Optional[str] = None, agent_idx: Optional[int] = None,
    ) -> None:
        """Queue evidence; only the owning run may start a decision or rotate."""
        if self._closed:
            return
        self._queued_interrupts.append(dict(
            trigger=trigger, context=context, other_agent_info=other_agent_info,
            agent_idx=agent_idx if agent_idx is not None else self._current_agent_idx,
        ))
        self._event_revision += 1
        cancel_action = self._accept_interrupts and self._control.is_running()
        if cancel_action:
            self._control.cancel()
        if trigger == "sos":
            self._record_sos_event("interrupt_requested", {
                "event_revision": self._event_revision, "action_cancel_requested": cancel_action,
                "run_active": self._run_active, "pending_count": len(self._queued_interrupts),
                **(self._sos_listener.current_trace if self._sos_listener else {}),
            })

    def _sos_audit_details(self, details: dict[str, Any]) -> dict[str, Any]:
        action_running = self._control.is_running()
        return dict(phase=("closed" if self._closed else "action" if action_running else
                           "decision" if self._sos_decision_in_flight else "idle_or_observation"),
                    action_id=(self._monitored_action or {}).get("action_id"),
                    event_revision=self._event_revision) | details

    def _record_sos_event(self, event: str, details: dict[str, Any]) -> None:
        """Persist operational evidence separately from model-visible input."""
        try:
            details = self._sos_audit_details(details)
            if self._session_id is None:
                self._session_id = session_logging.ensure_session_id(None)
            path = session_logging.session_sos_event_path(
                session_logging.resolve_benchmark_log_dir(), self.agent.agent_name, self._session_id)
            session_logging.append_sos_event(path, dict(
                event=event, step=self._log_round, monotonic_sec=time.monotonic(),
                agent_id=self.agent.agent_name, **details))
        except Exception as exc:
            print(f"[SOSListener] Event log unavailable: {exc}")

    async def _handle_sos(self, sos: dict[str, Any]) -> None:
        trace = self._sos_listener.current_trace if self._sos_listener else {}
        def audit(event, **details):
            self._record_sos_event(event, dict(trace, **details))
        if self._closed:
            audit("handling_result", result="ignored_closed")
            return
        orientations, distances = sos.get("orientations"), sos.get("distances")
        if (not isinstance(orientations, (list, tuple)) or not orientations
                or not isinstance(distances, (list, tuple)) or not distances):
            audit("handling_result", result="interrupt_without_geometry")
            self.request_interrupt("sos", context=build_sos_trigger_context(sos))
            return
        estimate_started = time.monotonic()
        audit("location_estimation_started", timeout_sec=SOS_LOCATION_ESTIMATE_TIMEOUT_SEC)
        try:
            async with asyncio.timeout(SOS_LOCATION_ESTIMATE_TIMEOUT_SEC):
                estimates = await self.agent.world_locations_from_sos(sos)
        except Exception as exc:
            audit("location_estimate_unavailable", reason=type(exc).__name__,
                  elapsed_sec=time.monotonic() - estimate_started)
            audit("handling_result", result="interrupt_location_unavailable")
            self.request_interrupt("sos", context=build_sos_trigger_context(sos))
            return
        audit("location_estimation_finished", elapsed_sec=time.monotonic() - estimate_started,
              estimate_count=len(estimates) if isinstance(estimates, list) else 0)
        if self._closed:
            audit("handling_result", result="ignored_closed")
            return
        if not isinstance(estimates, list) or not estimates:
            audit("location_estimate_unavailable", reason="no_estimate")
            audit("handling_result", result="interrupt_location_unavailable")
            self.request_interrupt("sos", context=build_sos_trigger_context(sos))
            return
        keep, locations = [], []
        sample_count = max(len(orientations), len(distances))
        for index in range(sample_count):
            point = estimates[index] if index < len(estimates) else None
            xy_m = world_xy_m(point)
            if xy_m is None:
                # Unlocated signals cannot safely be classified as repeats.
                keep.append(index)
                audit("location_estimate_unavailable", sample_index=index, reason="invalid_sample")
                continue
            location = build_sos_trigger_context(sos, point)["estimated_location"]
            audit("location_estimated", sample_index=index, estimated_location=location,
                  coordinate_unit="uu", estimated_xy_m=xy_m)
            if not self._sos_locations.remember(xy_m):
                audit("spatial_duplicate", sample_index=index, estimated_xy_m=xy_m,
                      radius_m=self._sos_locations.radius_m)
                continue
            keep.append(index)
            locations.append({"sample_index": index, "location": location})
        if not keep:
            audit("handling_result", result="ignored_duplicate", accepted_samples=0,
                  sample_count=sample_count)
            return
        # A packet may mix old and new sources. Pass only accepted samples on.
        accepted = copy.deepcopy(sos)
        for key in ("orientations", "distances"):
            accepted[key] = [sos[key][index] if index < len(sos[key]) else None for index in keep]
        context = build_sos_trigger_context(accepted, locations[0]["location"] if locations else None)
        context["estimated_locations"] = locations
        audit("handling_result", result="interrupt_new_signal", accepted_samples=len(keep),
              sample_count=sample_count)
        self.request_interrupt("sos", context=context)

    def set_assignment(self, assignment: dict[str, Any] | None) -> None:
        """Publish a controller's semantic task; active runs replan on change."""
        if self._cooperation_type not in {"C", "H"}:
            raise ValueError("assignments require centralized or hybrid coordination")
        if assignment is not None and not isinstance(assignment, dict):
            raise ValueError("assignment must be an object or None")
        allowed = {"task_id", "goal", "target", "room", "message", "priority"}
        if assignment is not None and (set(assignment) - allowed or
                any(not isinstance(value, str) for value in assignment.values())):
            raise ValueError("assignment uses only semantic string fields: " + ", ".join(sorted(allowed)))
        current = copy.deepcopy(assignment)
        if current == self._super_agent_assignment:
            return
        self._super_agent_assignment = current
        self.request_interrupt("assignment")

    async def on_other_agent_info(
        self, info: str | dict[str, Any], *, priority: str | None = None,
        sender: str | None = None, agent_idx: Optional[int] = None,
    ) -> None:
        if self._closed:
            return
        if self._cooperation_type == "none" or not getattr(self, "_communication_enabled", True):
            raise ValueError("teammate messages are unavailable in single-agent mode")
        message = normalize_message(info, priority=priority, sender=sender)
        if message["priority"] == "urgent":
            self.request_interrupt("other_agent", context={"teammate_message": message},
                                   agent_idx=agent_idx)
        else:
            # Ordinary arrivals must neither cancel an action nor invalidate an
            # in-flight model answer. Snapshot and consume them at acceptance.
            self._message_inbox.append(message)

    @property
    def seen_objects(self) -> dict[str, Any] | list[dict[str, Any]]:
        """All observed objects, independent of the model's semantic memory."""
        return self._seen_objects.snapshot()

    async def _stop_background_work(self) -> None:
        async with self._shutdown_lock:
            self._control.cancel()
            await self._control.wait_until_done()

    async def close(self) -> None:
        """Stop the run, all owned event work and the body; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        from .vision_protocol import close_and_cleanup
        await close_and_cleanup(self.context, self._close_resources, release=False)

    async def _close_resources(self) -> None:
        if hasattr(self.agent, "remove_deploy_listener"):
            self.agent.remove_deploy_listener(self._on_body_deployed)
        async with self._sos_lifecycle_lock:
            if self._sos_listener is not None:
                await self._sos_listener.stop()
        self._queued_interrupts.clear()
        self._sos_locations.clear()
        self._message_inbox.clear()
        self._decision_messages.clear()
        run_task = self._run_task
        if run_task is not None and run_task is not asyncio.current_task():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        await self._stop_background_work()
        await self.agent.close()

    async def _call_decision_and_control(
        self, trigger: str, *, agent_idx: Optional[int] = None,
        other_agent_info: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._wait_vision_running()
        async with self._decision_lock:
            if self._pending_boundary is not None:
                self._ready_observation = await self._capture_surroundings(**self._pending_boundary)
                self._pending_boundary = None
            # A cancelled follow-state query can leave a complete ready package
            # with unfinished effect confirmation. Retry it before any new model call.
            if self._ready_observation is not None and self._pending_effect is not None:
                await self._finalize_action_effect(self._ready_observation)
            if self._ready_observation is None:
                self._ready_observation = await self._capture_surroundings(reason="initial", action_id=self._log_round)

            await self._refresh_following_state(self._ready_observation)
            ordinary_messages = tuple(self._message_inbox)
            requests = tuple(self._queued_interrupts)
            self._decision_messages = sorted(copy.deepcopy(list(ordinary_messages)),
                                             key=lambda m: m["priority"] != "important")
            self._decision_messages.extend(
                copy.deepcopy(r["context"]["teammate_message"])
                for r in requests if (r.get("context") or {}).get("teammate_message")
            )
            revision = self._event_revision
            if requests:
                trigger = requests[-1]["trigger"]
                context = dict(requests[-1].get("context") or {})
                context["events"] = list(requests)
                messages = [r["other_agent_info"] for r in requests if r.get("other_agent_info")]
                if messages:
                    other_agent_info = "\n".join(messages)
            self._log_round += 1
            if any(request["trigger"] == "sos" for request in requests):
                self._record_sos_event("decision_triggered", {
                    "event_revision": revision, "sos_count": sum(r["trigger"] == "sos" for r in requests),
                })
            self.timing.clear_step()
            ready = self._ready_observation
            self._last_decision_output = None
            self._sos_decision_in_flight = True
            try:
                decision = await self._decision.call(
                    agent=self, observation=ready, trigger=trigger, agent_idx=agent_idx,
                    other_agent_info=other_agent_info, context=context,
                )
            except InvalidDecisionOutput as exc:
                self._reject_decision(exc.errors, audit=exc.audit, phase="parse")
                return
            finally:
                self._sos_decision_in_flight = False
            await self._wait_vision_running()
            if revision != self._event_revision:
                self.record_decision_outcome("discarded", reason="event_revision_changed",
                                             event_revision=self._event_revision)
                # New evidence invalidates the answer, including its memory delta.
                if any(r["trigger"] == "sos" for r in self._queued_interrupts[len(requests):]):
                    self._record_sos_event("decision_discarded", {
                        "reason": "sos_during_decision", "event_revision": self._event_revision,
                    })
                return
            if self._budget_expired():
                self.record_decision_outcome("discarded", reason="time_limit", end_reason="time_limit")
                return
            if not isinstance(decision, dict) or not isinstance(decision.get("action_payload"), dict):
                self._reject_decision([{"code": "invalid_decision", "path": "$.action",
                                       "message": "A decision requires one action object."}])
                return
            if (self._cooperation_type == "none" or not getattr(self, "_communication_enabled", True)) and decision.get("communication") is not None:
                self._reject_decision([{"code": "communication_unavailable", "path": "$.communication",
                                       "message": ("communication must be null in single-agent mode."
                                                   if self._cooperation_type == "none" else
                                                   "communication must be null when teammate communication is unavailable.")}])
                return
            try:
                normalized = self.agent.validate_action(decision["action_payload"])
            except (ValueError, NotImplementedError):
                # Invalid model choices do not abort an episode. Do not expose
                # engine exception text, private geometry or substitute actions.
                self._reject_decision([{"code": "action_validation_failed", "path": "$.action",
                                       "message": "The action does not satisfy the available action contract."}])
                return
            if normalized["name"] != decision.get("action"):
                self._reject_decision([{"code": "action_name_mismatch", "path": "$.action.name",
                                       "message": "Decision action and payload name disagree."}])
                return
            decision = dict(decision, action_payload=normalized)
            # A global boundary replaces current facts: route frames must not
            # restore destroyed entities or superseded positions as action targets.
            binding_frames = (ready.surroundings if self._observation_scope == "global"
                              or self._input_modality == "vision" else ready.frames)
            targets = {ref: xyz for frame in binding_frames for ref, xyz in frame.targets.items()}
            entities = tuple(entity for frame in binding_frames for entity in frame.entities)
            if self._input_modality == "vision":
                try:
                    validate_current_visual_target(normalized, targets, semantic_maps=self._semantic_maps)
                except StaleVisualTargetError as exc:
                    self._reject_decision([{"code": exc.code, "path": "$.action.parameters.target",
                                           "message": exc.public_message}])
                    return
            self.agent.bind_targets(targets, entities=entities)
            bind_images = getattr(self.agent, "bind_image_observation", None)
            if callable(bind_images):
                bind_images(ready)
            self.accept_decision(decision)
            self._remember_action(decision, self._log_round, _build_action_params_for_memory)
            # Acceptance consumes exactly the input snapshot, including when the
            # accepted action is subsequently cancelled before it can start.
            del self._queued_interrupts[:len(requests)]
            del self._message_inbox[:len(ordinary_messages)]
            self.record_decision_outcome("accepted", action=normalized)
            if self._current_communication is not None:
                self._outgoing_sequence += 1
                message = dict(self._current_communication, sender=self.agent.agent_name,
                               captured_at=time.monotonic(),
                               message_id=f"{id(self):x}:{self._outgoing_sequence}")
                self._outgoing_messages.append(message)
                try:
                    await self.flush_outgoing_messages()
                except asyncio.CancelledError:
                    self._cancel_before_execution(decision, "pre_execution_transport_cancelled")
                    self.record_decision_outcome("cancelled", reason="pre_execution_transport_cancelled")
                    raise
                except Exception as exc:
                    self._cancel_before_execution(decision, "pre_execution_transport_error")
                    self.record_decision_outcome("error", reason="pre_execution_transport_error",
                                                 error_type=type(exc).__name__)
                    raise
            if (revision != self._event_revision or
                    self._budget_expired()):
                reason = "event_before_execution" if revision != self._event_revision else "time_limit"
                self._cancel_before_execution(decision, reason)
                self.record_decision_outcome("cancelled", reason=reason)
                return
            self._ready_observation = None
            finalization_status = "completed"
            try:
                await self._wait_vision_running()
                status, during = await self._execute_with_execution_perception(decision)
                if status == "interrupted" and any(r["trigger"] == "sos" for r in self._queued_interrupts):
                    self._record_sos_event("action_interruption_observed", {
                        "action_id": self._log_round, "result_status": status,
                        "physical_stop_verified": False})
                self._pending_boundary = dict(
                    reason="interrupted" if status == "interrupted" else "action_end",
                    action_id=self._log_round, during_action=during,
                )
                # Keep this boundary pending if capture fails; a retry must not replay
                # an already executed action or present its old starting observation.
                self._ready_observation = await self._capture_surroundings(**self._pending_boundary)
                self._pending_boundary = None
                await self._finalize_action_effect(self._ready_observation)
            except asyncio.CancelledError:
                finalization_status = "cancelled"
                if self._previous_execution_feedback.get("action_id") != self._log_round:
                    self._cancel_before_execution(decision, "cancelled_before_execution")
                self.record_decision_outcome("cancelled", reason="execution_or_observation_cancelled")
                raise
            except Exception as exc:
                finalization_status = "error"
                self.record_decision_outcome("error", reason="execution_or_observation_error",
                                             error_type=type(exc).__name__)
                raise
            else:
                self.record_decision_outcome("executed", reason=self._previous_execution_feedback.get("reason"),
                    action=normalized, execution_status=self._previous_execution_feedback.get("status"))
            finally:
                self._finish_monitored_action(self._log_round, finalization_status)

    def _cancel_before_execution(self, decision: dict[str, Any], reason: str) -> None:
        result = action_result("interrupted", reason, {"execution_started": False})
        self._record_action_result(self._log_round, str(decision["action"]), result)
        self._set_execution_feedback(decision, result=result)

    def _reject_decision(
        self, errors: list[dict[str, str]], *, audit: dict[str, Any] | None = None,
        phase: str = "runtime_validation",
    ) -> None:
        """Record one failed attempt; the next loop obtains a fresh observation.

        No action, memory update, or outgoing message is committed. Keep queued
        evidence available for the next decision and retain the last real action
        feedback. Infrastructure/API failures are deliberately not caught here.
        """
        self.record_decision_outcome("rejected", reason=errors[0]["code"] if errors else "invalid_decision")
        record = copy.deepcopy(audit if audit is not None else self._last_decision_output) or {
            "protocol_version": PROTOCOL_VERSION, "raw_response": None, "original_errors": [],
            "repairs": [], "normalized_response": None,
        }
        record.update(step=self._log_round, phase=phase, status="invalid", errors=copy.deepcopy(errors))
        last = self._last_output_record or {}
        if not (phase == "parse" and last.get("step") == self._log_round
                and last.get("phase") == "parse" and last.get("status") == "invalid"):
            self.record_decision_output(record)
        self._previous_decision_feedback = {
            "step": self._log_round, "status": "invalid", "errors": copy.deepcopy(errors),
            "action_executed": False,
        }
        self._ready_observation = None
        self._pending_boundary = {"reason": "invalid_decision", "action_id": self._log_round}

    def _prepare_observation(self, observation: DecisionObservation) -> None:
        """Publish a complete batch's projection and emitted facts exactly once.

        Capture owns this transition. Offline replay can explicitly prepare its
        reconstructed batch before reading an input, without changing capture or
        model retry behavior.
        """
        if self._prepared_observation is observation:
            return
        payload = self._observation_adapter.prepare_observation(observation)
        if self._input_modality == "text":
            self._seen_objects.record_observation(payload)
        self._prepared_payload = payload
        self._prepared_observation = observation

    def _build_runtime_input_from_observation(
        self, observation: DecisionObservation, *, trigger: str,
        other_agent_info: Optional[str], agent_idx: Optional[int],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self._prepared_observation is not observation or self._prepared_payload is None:
            raise RuntimeError("Observation must be prepared before building model input")
        # The entity projection is stable; each model attempt receives its own
        # copy plus current event metadata and the batch's latest self-state.
        payload = copy.deepcopy(self._prepared_payload)
        payload["decision_trigger"] = trigger
        if context:
            payload["trigger_context"] = public_sos_context(context, self._oracle_names)
        return self._build_runtime_input_package(
            agent_position=observation.agent_position, current_observation=payload,
            other_agent_info=other_agent_info, agent_idx=agent_idx,
        )

    async def _capture_frame(self, **metadata: Any) -> ObservationFrame:
        await self._wait_vision_running()
        async with self._capture_lock:
            await self._wait_vision_running()
            self._perception_tick += 1
            frame = await self.agent.observe(tick=self._perception_tick, **metadata)
            if self._input_modality == "vision":
                self._seen_objects.record(frame)
            return frame

    async def _capture_frame_bounded(self, **metadata: Any) -> ObservationFrame:
        if self._vision_pause_controller is None:
            return await asyncio.wait_for(self._capture_frame(**metadata), timeout=self._capture_timeout)
        async with self._vision_pause_controller.timeout_active(self._capture_timeout):
            return await self._capture_frame(**metadata)

    async def _capture_surroundings(
        self, *, reason: str, action_id: int | None = None,
        during_action: tuple[ObservationFrame, ...] = (),
    ) -> DecisionObservation:
        self._batch_counter += 1
        batch_id = f"surround-{self._batch_counter}"
        frames: list[ObservationFrame] = []

        async def capture_view(view: str) -> None:
            frame = await self._capture_frame_bounded(
                view=view, batch_id=batch_id, action_id=action_id,
                capture_kind="surround", name_suffix=f"_{view}",
            )
            frames.append(frame)

        if self._observation_scope == "global":
            frames.append(await self._capture_frame_bounded(
                view="global", batch_id=batch_id, action_id=action_id, capture_kind="global",
            ))
        else:
            await self.agent.observe_surroundings(capture_view=capture_view)
        position = await self.agent.get_agent_position_and_forward()
        if not position:
            raise RuntimeError("Agent state unavailable after surrounding observation")
        position = copy.deepcopy(position)
        position["room"] = room_for(position, self._semantic_maps)
        position["following_npcs"] = None
        observation = DecisionObservation(
            batch_id=batch_id, reason=reason, action_id=action_id,
            surroundings=tuple(frames), agent_position=copy.deepcopy(position),
            during_action=during_action, scope=self._observation_scope,
        )
        # Final actions publish too, even when no further model call is requested.
        self._prepare_observation(observation)
        return observation

    async def _execute_with_execution_perception(
        self, decision: dict[str, Any],
    ) -> tuple[str, tuple[ObservationFrame, ...]]:
        frames: list[ObservationFrame] = []
        interval = self._perception.interval_sec
        action_finished = False

        async def execute() -> None:
            stop = asyncio.Event()
            async def perform_action() -> None:
                nonlocal action_finished
                await self._execute_action(decision)
                action_finished = True

            action_task = asyncio.create_task(perform_action())

            async def capture() -> None:
                clock = asyncio.get_running_loop()
                controller = self._vision_pause_controller
                now = controller.active_now if controller is not None else clock.time
                deadline = now() + interval
                while not stop.is_set():
                    try:
                        remaining = max(0.0, deadline - now())
                        if controller is None:
                            await asyncio.wait_for(stop.wait(), timeout=remaining)
                        else:
                            async with controller.timeout_active(remaining):
                                await stop.wait()
                        return
                    except asyncio.TimeoutError:
                        pass
                    if stop.is_set() or action_task.done():
                        return
                    frame = await self._capture_frame_bounded(
                        action_id=self._log_round,
                        capture_kind="global" if self._observation_scope == "global" else "front",
                        view="global" if self._observation_scope == "global" else "front",
                    )
                    # Global decisions use only the fresh post-action snapshot.
                    if self._observation_scope != "global":
                        frames.append(frame)
                    deadline += interval
                    # Skip missed deadlines instead of building an inference backlog.
                    while deadline <= now():
                        deadline += interval

            capture_task = asyncio.create_task(capture())
            try:
                done, _ = await asyncio.wait({action_task, capture_task}, return_when=asyncio.FIRST_COMPLETED)
                if capture_task in done:
                    await capture_task  # Surface sensor failure and cancel the action.
                await action_task
                stop.set()
                # Finish an in-flight frame before rotating; timeout is per frame.
                await capture_task
            finally:
                stop.set()
                for task in (action_task, capture_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(action_task, capture_task, return_exceptions=True)

        try:
            status = await self._control.execute(
                decision, execute_fn=lambda _: track(
                    "action_execution", execute, self.timing,
                    step=self._log_round, action=decision["action"],
                ),
            )
        except Exception as exc:
            if action_finished:
                status = self._previous_execution_feedback.get("status", "completed")
                self._previous_execution_feedback["observation_error"] = f"{type(exc).__name__}: {exc}"
            else:
                status = "failed"
                data = self._action_progress(decision)
                data["error"] = f"{type(exc).__name__}: {exc}"
                result = action_result("failed", "execution_error", data)
                if self._pending_effect is not None:
                    self._pending_effect["execution_result"] = copy.deepcopy(result)
                self._record_action_result(self._log_round, str(decision["action"]), result)
                self._set_execution_feedback(decision, result=result)
                self._previous_execution_feedback["error"] = data["error"]
        if status == "interrupted" and not action_finished:
            self._set_execution_feedback(decision, status=status)
        elif action_finished and status in {"completed", "interrupted"}:
            # A late SOS can cancel observation cleanup, but cannot undo an
            # already completed action or discard its returned result.
            status = self._previous_execution_feedback.get("status", "completed")
        return status, tuple(frames)

    async def _execute_action(self, decision: dict[str, Any]) -> None:
        step = self._log_round
        self.record_decision_outcome("executing", action=decision["action_payload"])
        self._monitored_action = {"agent_id": self.agent.agent_name, "action_id": step,
            "decision_step": step, "action": copy.deepcopy(decision["action_payload"]),
            "started_at_monotonic": time.monotonic()}
        record_event(self.context, "actions", {"event": "action_started", **self._monitored_action})
        reset = getattr(self.agent, "reset_action_progress", None)
        if callable(reset):
            reset(str(decision["action"]))
        self._pending_effect = {"decision": decision, "step": step, "target_location": None}
        bind_navigation = getattr(self.agent, "bind_navigation_target", None)
        pinned = False
        try:
            if decision["action"] == "navigate":
                self._pending_effect["target_location"] = await self._navigation_target(decision)
                if decision["action_payload"]["subtype"] == "relative" and callable(bind_navigation):
                    bind_navigation(self._pending_effect["target_location"])
                    pinned = True
            if (decision["action"] == "instruct_civilian"
                    and decision["action_payload"]["subtype"] == "follow"):
                self._pending_effect["previous_follow_state"] = await self._read_follow_state()
            try:
                raw = await self.agent.execute_action(decision["action_payload"])
            finally:
                if pinned:
                    bind_navigation()
        except asyncio.CancelledError:
            result = action_result("interrupted", "cancelled", self._action_progress(decision))
            self._pending_effect["execution_result"] = copy.deepcopy(result)
            self._record_action_result(step, str(decision["action"]), result)
            self._set_execution_feedback(decision, result=result)
            raise
        except DestroyedTargetError as exc:
            result = exc.result()
            self._pending_effect = None
            self._record_action_result(step, str(decision["action"]), result)
            self._set_execution_feedback(decision, result=result)
            return
        except ImageTargetError as exc:
            result = action_result("failed", exc.code, {"message": exc.public_message,
                "execution_started": False})
            self._pending_effect = None
            self._record_action_result(step, str(decision["action"]), result)
            self._set_execution_feedback(decision, result=result)
            return
        except Exception as exc:
            data = self._action_progress(decision)
            data["error"] = f"{type(exc).__name__}: {exc}"
            result = action_result("failed", "execution_error", data)
            self._pending_effect["execution_result"] = copy.deepcopy(result)
            self._record_action_result(step, str(decision["action"]), result)
            self._set_execution_feedback(decision, result=result)
            raise
        result = normalize_action_result(str(decision["action"]), raw)
        if (decision["action"] == "instruct_civilian"
                and decision["action_payload"]["subtype"] == "stop_following"):
            receipt = result["data"].get("execution_result")
            if type(receipt) is bool:
                result = action_result("completed" if receipt else "failed",
                    "stop_following_succeeded" if receipt else "stop_following_failed",
                    {"execution_result": receipt, "effect_success": receipt})
                if receipt:
                    self._pending_effect["stop_following_succeeded"] = True
                    self._clear_following_state()
        self._pending_effect["execution_result"] = result
        self._record_action_result(step, str(decision["action"]), result)
        self._set_execution_feedback(decision, result=result)

    async def _navigation_target(self, decision: dict[str, Any]) -> Any:
        payload = decision["action_payload"]
        parameters = payload["parameters"]
        if payload["subtype"] == "entity":
            return await self.agent.resolve_entity_location(parameters["target"])
        if payload["subtype"] == "image":
            return await self.agent.resolve_image_location(parameters["target"], purpose="navigate")
        pose = await self.agent.get_agent_position_and_forward()
        forward = pose["forward"]
        fx, fy = ((forward["x"], forward["y"]) if isinstance(forward, dict)
                  else (forward.x, forward.y))
        angle = math.atan2(fy, fx) + math.radians(direction_to_angle(parameters["direction"]))
        distance = float(parameters["distance"]) * 100.0
        return (pose["x"] + distance * math.cos(angle),
                pose["y"] + distance * math.sin(angle), NAV_TARGET_Z)

    async def _read_follow_state(self) -> dict[str, Any] | None:
        try:
            state = await asyncio.wait_for(self.agent.get_self_state(), timeout=self._capture_timeout)
        except Exception:
            # Failed state queries are unknown, never an empty follower list.
            return None
        return copy.deepcopy(state) if isinstance(state, dict) else None

    def _clear_following_state(self, observation: DecisionObservation | None = None) -> None:
        """Apply a successful stop receipt to this batch's cached follower state."""
        self._follow_state = copy.deepcopy(self._follow_state) if isinstance(self._follow_state, dict) else {}
        self._follow_state["following_NPC"] = {}
        observation = observation if observation is not None else self._ready_observation
        if observation is not None:
            observation.agent_position["following_npcs"] = []
            self._follow_state_batch = observation.batch_id

    async def _refresh_following_state(self, observation: DecisionObservation) -> None:
        if self._input_modality != "text" or self._follow_state_batch == observation.batch_id:
            return
        state = await self._read_follow_state()
        observation.agent_position["following_npcs"] = self._oracle_names.following_names(state)
        self._follow_state = state
        self._follow_state_batch = observation.batch_id

    async def _finalize_action_effect(self, observation: DecisionObservation) -> None:
        pending = self._pending_effect
        if pending is None:
            return
        # Ready frames already belong to the runtime before this cancellable
        # query, so a retry reuses them and never re-executes the action.
        if pending.get("stop_following_succeeded"):
            # The stop contract is its bool receipt, not a follower-state check.
            self._clear_following_state(observation)
        else:
            await self._refresh_following_state(observation)
        decision, step = pending["decision"], pending["step"]
        action = decision["action"]
        payload = decision["action_payload"]
        if action == "suppress_fire":
            result = fire_effect(observation, payload["parameters"]["target"],
                                 modality=self._input_modality)
        elif action == "instruct_civilian" and payload["subtype"] == "follow":
            state = (self._follow_state if self._input_modality == "text"
                     else await self._read_follow_state())
            result = follow_effect(state, previous_state=pending.get("previous_follow_state"))
        elif action == "instruct_civilian":
            if "execution_result" not in pending:
                self._pending_effect = None
                return
            execution = pending["execution_result"]
            result = execution
        elif action == "navigate" and self._previous_execution_feedback.get("status") != "interrupted":
            result = navigation_feedback(pending.get("execution_result"),
                navigation_effect(observation.agent_position, pending["target_location"], self._navigation_tolerance_m))
        else:
            # wait retains its elapsed/interrupted result; explore retains its
            # per-leg pose checks and full-loop progress.
            self._pending_effect = None
            return
        execution = pending.get("execution_result")
        if action != "navigate" and execution is not None and execution.get("status") in {"failed", "interrupted"}:
            # Observed world changes cannot prove that a failed command ran.
            # Keep command errors/cancellation primary; attach independent effect
            # evidence instead of overwriting failure with unknown or success.
            observed_effect = result
            result = copy.deepcopy(execution)
            result["data"]["observed_effect"] = observed_effect
        diagnostics = {key: self._previous_execution_feedback[key]
                       for key in ("error", "observation_error")
                       if key in self._previous_execution_feedback}
        self._record_action_result(step, action, result, phase="verification")
        self._set_execution_feedback(decision, result=result)
        self._previous_execution_feedback.update(diagnostics)
        self._pending_effect = None

    def _action_progress(self, decision: dict[str, Any]) -> dict[str, Any]:
        getter = getattr(self.agent, "get_action_progress", None)
        return getter(str(decision["action"])) if callable(getter) else {}

    def _record_action_result(self, step: int, action: str, result: Any, *, phase: str = "execution") -> None:
        """Store the result and its stage without counting a second physical action."""
        result = normalize_action_result(action, result)
        if self._vision_action_feedback is not None:
            self._vision_action_feedback.record(step, result, phase=phase)
            result = self._vision_action_feedback.enrich(step, result)
        self._update_memory_entry(step, action_result=json_safe(result))
        current = self._monitored_action
        if current is not None and current["action_id"] == step:
            current["result"] = copy.deepcopy(json_safe(result))
            record_event(self.context, "actions", {**current,
                "event": "action_feedback" if phase == "verification" else "action_result",
                "phase": phase, "captured_at_monotonic": time.monotonic()})

    def _finish_monitored_action(self, step: int, finalization_status: str) -> None:
        current = self._monitored_action
        if current is None or current["action_id"] != step or current.get("finished"):
            return
        finished = time.monotonic()
        record_event(self.context, "actions", {**current, "event": "action_finished",
            "finished_at_monotonic": finished,
            "elapsed_sec": finished - current["started_at_monotonic"],
            "finalization_status": finalization_status})
        current["finished"] = True

    def _set_execution_feedback(
        self, decision: dict[str, Any], *, status: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        if result is None:
            previous = self._previous_execution_feedback
            data = self._action_progress(decision)
            if previous.get("action_id") == self._log_round:
                data = {**previous.get("data", {}), **data}
            result = action_result(status or "failed",
                                   "cancelled" if status == "interrupted" else "execution_error", data)
            self._record_action_result(self._log_round, str(decision["action"]), result)
        result = normalize_action_result(str(decision["action"]), result)
        if self._vision_action_feedback is not None:
            result = self._vision_action_feedback.enrich(self._log_round, result)
        self._previous_execution_feedback = {
            "action": str(decision["action"]), "action_id": self._log_round,
            "action_params": _build_action_params_for_memory(decision),
            **json_safe(result),
        }


def _build_action_params_for_memory(decision: dict[str, Any]) -> dict[str, Any]:
    payload = decision.get("action_payload")
    if not isinstance(payload, dict):
        return {}
    parameters = payload.get("parameters")
    params = copy.deepcopy(parameters) if isinstance(parameters, dict) else {}
    if payload.get("subtype"):
        params["subtype"] = payload["subtype"]
    return params
