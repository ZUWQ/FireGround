"""Call the model with a ready package; ownership stays with the runtime."""
from __future__ import annotations

import asyncio
from typing import Any

from EmbodiedMAS.agent.decision_output import InvalidDecisionOutput

from EmbodiedMAS.agent.perception.records import DecisionObservation
from EmbodiedMAS.agent.runtime.timing import track_llm_thinking


class DecisionModule:
    async def call(self, *, agent: Any, observation: DecisionObservation,
                   trigger: str, agent_idx: int | None = None,
                   other_agent_info: str | None = None,
                   context: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime_input = agent._build_runtime_input_from_observation(
            observation, trigger=trigger, other_agent_info=other_agent_info,
            agent_idx=agent_idx, context=context,
        )
        images = [path for frame in observation.frames for path in frame.image_paths]
        record_input = getattr(agent, "record_decision_input", None)
        record_outcome = getattr(agent, "record_decision_outcome", None)
        if callable(record_input):
            record_input(runtime_input)
        try:
            return await track_llm_thinking(
                lambda: agent.decide(runtime_input, perception_image_paths=images or None,
                                     agent_idx=agent_idx),
                agent.timing,
                pause_conn=(agent.agent._conn if agent._pause_during_llm
                            and getattr(agent, "_t1_pause_controller", None) is None else None),
            )
        except InvalidDecisionOutput:
            raise  # Runtime records the parse rejection, preserving its existing handling.
        except asyncio.CancelledError:
            if callable(record_outcome):
                record_outcome("cancelled", reason="model_call_cancelled")
            raise
        except Exception as exc:
            if callable(record_outcome):
                record_outcome("error", reason="model_call_failed", error_type=type(exc).__name__)
            raise
