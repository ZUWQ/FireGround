from __future__ import annotations

import asyncio
import base64
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import EmbodiedMAS  # noqa: F401 — triggers bootstrap

from EmbodiedMAS.agent.decision_output import InvalidDecisionOutput, process_response
from EmbodiedMAS.agent.config.llm_config import get_llm_config, _merge_chat_completion_kwargs
from EmbodiedMAS.agent.config.llm_transport import create_chat_completion
from EmbodiedMAS.agent.spatial_units import world_xy_m
from EmbodiedMAS.agent.prompt.emas_prompt import build_initial_prompt
from EmbodiedMAS.agent.config.environment_config import load_environment_config, output_token_budget
from EmbodiedMAS.agent.memory import EpisodeMemory
from EmbodiedMAS.agent.prompt.runtime_text import (
    format_memory_log_line,
    format_runtime_for_llm,
    json_safe,
)
from EmbodiedMAS.agent.runtime import logging as llm_scene_timing
from EmbodiedMAS.evaluation.monitor import record_event


_VALID_AGENT_TYPES = frozenset({"FR", "HF", "RR", "ER"})
_VALID_INPUT_MODALITIES = frozenset({"text", "vision"})


def _image_path_to_content_item(path: str) -> dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/webp"
    b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class BaseLLMAgent:
    """Shared LLM utilities for embodied agents.

    Provides prompt setup, agent-state formatting, LLM calls, EMAS response
    parsing, memory, feedback state, and session logs. Concrete agents
    own the full step orchestration, perception conversion, robot control, and timing.
    """

    def __init__(
        self,
        context: Any,
        agent: Any,
        *,
        prompt_type: str,
        input_modality: str,
        cooperation_type: str = "D",
        max_tokens: int | None = None,
        observation_scope: str = "local",
        record_mode: str | None = None,
        episode_briefing: dict[str, Any] | None = None,
        vision_backend: str | None = None,
    ) -> None:
        runtime_cfg = load_environment_config()
        self.context = context
        self.agent = agent
        self._agent_type = str(prompt_type).upper()
        self._prompt_environment = dict(runtime_cfg["prompt_environment"])
        modality = str(input_modality).strip().lower()
        if modality not in _VALID_INPUT_MODALITIES:
            raise ValueError(
                f"input_modality must be 'text' or 'vision', got {input_modality!r}"
            )
        self._input_modality = modality
        self._vision_feedback_protocol = (
            runtime_cfg.get("vision_feedback_protocol", "legacy") if modality == "vision" else "legacy")
        self._vision_backend = vision_backend or runtime_cfg.get("vision_backend", "grounded_sam")
        self._cooperation_type = cooperation_type
        self._observation_scope = observation_scope
        self._record_mode = record_mode
        self._episode_briefing = episode_briefing or {}
        self._pause_during_llm = bool(runtime_cfg["pause_during_llm"])
        self._max_tokens = output_token_budget(modality, runtime_cfg, explicit=max_tokens)

        # Rule-based interface tests share this runtime without needing keys or
        # importing the OpenAI SDK. Only a real model call initializes a client.
        self._llm_config = None
        self._client = None
        self._system_prompt = self._load_system_prompt()

        self._episode_memory = EpisodeMemory(runtime_cfg["max_memory_size"],
                                             runtime_cfg["memory_note_max_chars"])
        self._memory = self._episode_memory.history
        self._received_decision_messages: list[dict[str, Any]] = []
        self._previous_execution_feedback: dict[str, Any] = {
            "action": None,
            "action_params": None,
        }

        self._session_id: Optional[str] = runtime_cfg.get("session_id")
        self._session_memory_header_written = False
        self._log_round = 0
        self._call_sequence = 0

        self._previous_decision_feedback: dict[str, Any] | None = None
        self._last_decision_output: dict[str, Any] | None = None
        self._last_output_record: dict[str, Any] | None = None
        self._current_decision_basis: Optional[str] = None
        self._current_communication: Optional[dict[str, Any]] = None

    def _build_emas_agent_state(
        self,
        agent_position: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not agent_position:
            return {
                "location": None,
                "orientation": None,
                "room": "unknown",
                "following_npcs": None,
                "resource_state": {},
            }

        forward = json_safe(agent_position["forward"])
        return {
            "location": world_xy_m(agent_position),
            "orientation": [
                float(forward["x"]),
                float(forward["y"]),
                float(forward["z"]),
            ],
            "room": agent_position.get("room", "unknown"),
            "following_npcs": copy.deepcopy(agent_position.get("following_npcs")),
            "resource_state": {},
        }

    def _load_system_prompt(self) -> str:
        agent_type = str(self._agent_type).upper()
        if agent_type not in _VALID_AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")
        prompt = build_initial_prompt(
            self._input_modality,
            agent_type,
            self._cooperation_type,
            observation_scope=self._observation_scope,
            record_mode=self._record_mode,
            episode_briefing=self._episode_briefing,
            vision_backend=self._vision_backend,
        )
        if self._vision_feedback_protocol == "execution_effect_v2":
            prompt += "\n\n# Vision command and effect feedback\n" + (
                Path(__file__).parent / "prompt/templates/vision_feedback.md").read_text(encoding="utf-8")
        return prompt

    def flush_session_log(self, log_round: int) -> Optional[Path]:
        log_dir = llm_scene_timing.resolve_benchmark_log_dir()
        aid = self.agent.agent_name
        if self._session_id is None:
            self._session_id = llm_scene_timing.ensure_session_id(None)
        sid = self._session_id

        mem_path = llm_scene_timing.session_memory_path(log_dir, aid, sid)

        memory_lines = (
            ["(empty)"]
            if not self._memory
            else [format_memory_log_line(mem).strip() for mem in self._memory]
        )

        mem_header = None
        if not self._session_memory_header_written:
            self._session_memory_header_written = True
            mem_header = (
                f"SESSION kind=memory agent_id={aid!r} session_id={sid!r} "
                f"file_started={datetime.now().isoformat()}"
            )

        llm_scene_timing.append_memory_block(
            mem_path,
            log_round=log_round,
            memory_lines=memory_lines,
            session_header=mem_header,
        )
        return mem_path

    def _teammate_messages(self, other_agent_info: Optional[str]) -> list[dict[str, Any]]:
        if not other_agent_info:
            return []
        return [
            {
                "sender": "teammates",
                "type": "teammate_info",
                "content": other_agent_info,
            }
        ]

    @property
    def memory_note(self) -> str:
        return self._episode_memory.note

    def _parse_response(
        self, response_text: str | None, *, finish_reason: str | None = None,
    ) -> dict[str, Any]:
        try:
            payload, audit = process_response(response_text, note_max_chars=self._episode_memory.note_max_chars)
        except InvalidDecisionOutput as exc:
            exc.audit["finish_reason"] = finish_reason
            self._last_decision_output = exc.audit
            raise
        audit["finish_reason"] = finish_reason
        if finish_reason is not None and finish_reason != "stop":
            error = {"code": "incomplete_response", "path": "$",
                     "message": "The model response did not finish normally; no decision was accepted."}
            audit.update(status="invalid", errors=[error])
            audit["original_errors"] = [*audit["original_errors"], error]
            self._last_decision_output = audit
            raise InvalidDecisionOutput(audit)
        self._last_decision_output = audit
        return {
            "action": payload["action"]["name"],
            "action_payload": copy.deepcopy(payload["action"]),
            "decision_basis": payload["decision_basis"],
            "memory_update": copy.deepcopy(payload["memory_update"]),
            "communication": copy.deepcopy(payload["communication"]),
        }

    def record_decision_output(self, record: dict[str, Any]) -> None:
        """Persist raw replies and format changes without adding them to memory."""
        detached = copy.deepcopy(record)
        detached.setdefault("step", self._log_round)
        detached.setdefault("recorded_at", datetime.now().isoformat())
        detached.setdefault("phase", "parse")
        self._last_output_record = detached
        log_dir = llm_scene_timing.resolve_benchmark_log_dir()
        if self._session_id is None:
            self._session_id = llm_scene_timing.ensure_session_id(None)
        path = llm_scene_timing.session_decision_output_path(
            log_dir, self.agent.agent_name, self._session_id)
        llm_scene_timing.append_decision_output(path, detached)

    def accept_decision(self, decision: dict[str, Any]) -> None:
        """Commit cognitive updates only after the runtime accepts the answer."""
        self._previous_decision_feedback = None
        update = decision.get("memory_update") or {}
        self._episode_memory.apply_note(update.get("note"))
        self._current_decision_basis = decision.get("decision_basis") or None
        communication = decision.get("communication")
        self._current_communication = communication if isinstance(communication, dict) else None

    def _build_llm_content(
        self,
        runtime_input: dict[str, Any],
        *,
        perception_image_paths: Optional[list[str]],
    ) -> list[dict[str, Any]]:
        runtime_json = format_runtime_for_llm(runtime_input, input_modality=self._input_modality)
        content_items: list[dict[str, Any]] = [
            {"type": "text", "text": f"RUNTIME INPUT:\n{runtime_json}"}
        ]
        # print(f"[BaseLLMAgent] content_items: {content_items}")
        if perception_image_paths:
            for image_path in perception_image_paths:
                content_items.append(_image_path_to_content_item(image_path))
        return content_items

    def _ensure_llm_client(self) -> None:
        if self._client is None:
            from openai import OpenAI
            config = get_llm_config()
            self._client = OpenAI(**config.get_client_kwargs())
            self._llm_config = config

    async def _call_llm(
        self,
        runtime_input: dict[str, Any],
        *,
        perception_image_paths: Optional[list[str]],
        agent_idx: Optional[int],
    ) -> dict[str, Any]:
        vision_pause = getattr(self.context, "_emas_vision_pause_controller", None)
        if vision_pause is not None:
            if self._input_modality != "vision":
                raise ValueError("Vision pause transport cannot be used by an Oracle agent")
            return await self._call_vision_paused(
                runtime_input, perception_image_paths=perception_image_paths,
                controller=vision_pause,
            )
        self._ensure_llm_client()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._build_llm_content(
                    runtime_input,
                    perception_image_paths=perception_image_paths,
                ),
            },
        ]
        model = self._llm_config.get_model()

        # Resolve the profile on this controller; do not rely on the global
        # SDK patch, which may be absent or belong to another configuration.
        token_parameter = (
            "max_completion_tokens"
            if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
            else "max_tokens"
        )
        request_kwargs = _merge_chat_completion_kwargs(
            {"model": model, "messages": messages, token_parameter: self._max_tokens},
            self._llm_config.get_chat_completion_extra_kwargs(),
        )

        def _sync_call() -> Any:
            return create_chat_completion(self._client, request_kwargs)

        self._call_sequence += 1
        call = {"agent_id": self.agent.agent_name, "call_id": self._call_sequence,
                "decision_step": self._log_round, "model": model,
                "reasoning_effort": request_kwargs.get("reasoning_effort"),
                "started_at_monotonic": time.monotonic()}
        thinking_audit = getattr(self._llm_config, "get_deepseek_thinking_audit", None)
        if model.lower().startswith("deepseek") and callable(thinking_audit):
            call.update(thinking_audit())
        record_event(self.context, "models", {"event": "call_started", **call})
        try:
            pause = getattr(self, "_t1_pause_controller", None)
            if pause is None:
                completion, cost_record = await asyncio.to_thread(_sync_call)
            else:
                async with pause.before_model():
                    completion, cost_record = await asyncio.to_thread(_sync_call)
        except BaseException as exc:
            finished = time.monotonic()
            record_event(self.context, "models", {"event": (
                "call_cancelled" if isinstance(exc, asyncio.CancelledError) else "call_error"),
                **call, "finished_at_monotonic": finished,
                "elapsed_sec": finished - call["started_at_monotonic"],
                "error_type": type(exc).__name__, "usage": None})
            raise
        finished = time.monotonic()
        usage = getattr(completion, "usage", None)
        usage = usage.model_dump() if hasattr(usage, "model_dump") else (
            copy.deepcopy(usage) if isinstance(usage, dict) else None)
        if model.lower().startswith("deepseek") and callable(thinking_audit):
            call.update(thinking_audit(completion))
        record_event(self.context, "models", {"event": "call_completed", **call, **cost_record,
            "finished_at_monotonic": finished,
            "elapsed_sec": finished - call["started_at_monotonic"],
            "model": getattr(completion, "model", model), "usage": usage,
            "finish_reason": completion.choices[0].finish_reason})
        response_text = completion.choices[0].message.content
        started = time.monotonic()
        try:
            decision = self._parse_response(
                response_text, finish_reason=completion.choices[0].finish_reason)
        except InvalidDecisionOutput as exc:
            self.record_decision_output({**exc.audit, "phase": "parse",
                                         "elapsed_sec": time.monotonic() - started})
            raise
        self.record_decision_output({**self._last_decision_output, "phase": "parse",
                                     "elapsed_sec": time.monotonic() - started})
        return decision

    async def _call_vision_paused(
        self, runtime_input: dict[str, Any], *,
        perception_image_paths: Optional[list[str]], controller: Any,
    ) -> dict[str, Any]:
        """The new vision protocol owns async requests; legacy clients stay intact."""
        from EmbodiedMAS.agent.config.vision_llm_transport import create_vision_completion

        await controller.wait_running()
        config = self._llm_config or get_llm_config()
        self._llm_config = config
        model = config.get_model()
        token_parameter = (
            "max_completion_tokens"
            if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
            else "max_tokens"
        )
        request_kwargs = _merge_chat_completion_kwargs(
            {"model": model, "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_llm_content(
                    runtime_input, perception_image_paths=perception_image_paths)},
            ], token_parameter: self._max_tokens},
            config.get_chat_completion_extra_kwargs(),
        )
        self._call_sequence += 1
        call = {"agent_id": self.agent.agent_name, "call_id": self._call_sequence,
                "decision_step": self._log_round, "model": model,
                "reasoning_effort": request_kwargs.get("reasoning_effort"),
                "started_at_monotonic": time.monotonic()}
        thinking_audit = getattr(config, "get_deepseek_thinking_audit", None)
        if model.lower().startswith("deepseek") and callable(thinking_audit):
            call.update(thinking_audit())
        logical_id = f"{self.agent.agent_name}:{self._call_sequence}"
        async with controller.model_wait(logical_id):
            record_event(self.context, "models", {"event": "call_started", **call})
            try:
                completion, cost_record = await create_vision_completion(
                    config.get_client_kwargs(), request_kwargs,
                    record_attempt=lambda event: record_event(
                        self.context, "models", {**call, **event}),
                    client_factory=getattr(self, "_vision_async_client_factory", None),
                )
            except BaseException as exc:
                finished = time.monotonic()
                record_event(self.context, "models", {
                    "event": "call_cancelled" if isinstance(exc, asyncio.CancelledError) else "call_error",
                    **call, "finished_at_monotonic": finished,
                    "elapsed_sec": finished - call["started_at_monotonic"],
                    "error_type": type(exc).__name__, "usage": None,
                })
                raise
            finished = time.monotonic()
            usage = getattr(completion, "usage", None)
            usage = usage.model_dump() if hasattr(usage, "model_dump") else (
                copy.deepcopy(usage) if isinstance(usage, dict) else None)
            if model.lower().startswith("deepseek") and callable(thinking_audit):
                call.update(thinking_audit(completion))
            record_event(self.context, "models", {
                "event": "call_completed", **call, **cost_record,
                "finished_at_monotonic": finished,
                "elapsed_sec": finished - call["started_at_monotonic"],
                "model": getattr(completion, "model", model), "usage": usage,
                "finish_reason": completion.choices[0].finish_reason,
            })
            started = time.monotonic()
            try:
                decision = self._parse_response(
                    completion.choices[0].message.content,
                    finish_reason=completion.choices[0].finish_reason,
                )
            except InvalidDecisionOutput as exc:
                self.record_decision_output({**exc.audit, "phase": "parse",
                                             "elapsed_sec": time.monotonic() - started})
                raise
            self.record_decision_output({**self._last_decision_output, "phase": "parse",
                                         "elapsed_sec": time.monotonic() - started})
            # Exiting model_wait first releases this holder and then waits for
            # the whole world to resume. No action, message or new round escapes.
            return decision

    async def decide(
        self,
        runtime_input: dict[str, Any],
        *,
        perception_image_paths: Optional[list[str]] = None,
        agent_idx: Optional[int] = None,
    ) -> dict[str, Any]:
        return await self._call_llm(
            runtime_input,
            perception_image_paths=perception_image_paths,
            agent_idx=agent_idx,
        )

    def _remember_action(
        self,
        decision: dict[str, Any],
        current_step: int,
        action_params_builder: Any,
    ) -> None:
        self._episode_memory.append(
            {
                "step": current_step,
                "decision_basis": decision.get("decision_basis", ""),
                "action": decision.get("action"),
                "action_params": json_safe(action_params_builder(decision)),
                "teammate_messages": copy.deepcopy(self._received_decision_messages),
            }
        )

    def _update_memory_entry(self, step: int, **kwargs: Any) -> None:
        self._episode_memory.update(step, **json_safe(kwargs))

__all__ = [
    "BaseLLMAgent",
    "get_llm_config",
]
