from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from uuid import uuid4
import math
from pathlib import Path
from typing import Any, Optional

import EmbodiedMAS  # noqa: F401 — triggers bootstrap
import tongsim as ts
from tongsim.core.world_context import WorldContext

from EmbodiedMAS.agent.visual_backends import VISUAL_BACKENDS
from EmbodiedMAS.agent.config.environment_config import get_burn_time, load_environment_config
from EmbodiedMAS.agent.config.llm_config import LLM_PROFILES, select_llm_profile
from EmbodiedMAS.evaluation.monitor import EpisodeMonitor
from EmbodiedMAS.environment.startup import initialize_environment
from EmbodiedMAS.agent.factory import create_agent
from EmbodiedMAS.runners.common import (DEFAULT_POSITIONS, close_runtimes, deploy_runtime,
    preload_vision, run_episode, validate_options, validate_position)

Z_HEIGHT = 1000


def _build_backend(
    *,
    mode: str,
    context: WorldContext,
    agent_type: str,
    log_dir: Optional[Path] = None,
    **runtime_options: Any,
):
    mode = mode.lower()
    agent_type = str(agent_type).upper()
    if agent_type not in {"FR", "RR", "HF", "ER"}:
        raise ValueError(f"Unsupported agent_type={agent_type!r}; only FR/RR/HF/ER are supported")

    return create_agent(context, agent_type=agent_type, modality=mode, log_dir=log_dir, **runtime_options)


class SARunner:
    def __init__(
        self,
        context: WorldContext,
        *,
        mode: str = "text",
        agent_type: str = "FR",
        agent_position: Optional[dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        **runtime_options: Any,
    ) -> None:
        self.context = context
        self.mode = mode.lower()
        self.agent_type = str(agent_type).upper()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        self.output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "outputs" / "sa" / self.run_id
        self.monitor = None
        self.environment_setup = {}
        self._last_end_reason = None
        self.runtime_options = runtime_options
        self.agent_position = validate_position(agent_position if agent_position is not None else DEFAULT_POSITIONS[0])
        self.llm_agent: Optional[Any] = None
        self._closed = False

    async def spawn_agent(self) -> None:
        if self._closed or self.llm_agent is not None:
            raise RuntimeError("Use a fresh runner for deployment")
        self.llm_agent = _build_backend(
            mode=self.mode,
            context=self.context,
            agent_type=self.agent_type,
            log_dir=self.output_dir,
            **self.runtime_options,
        )

        self.llm_agent._session_id = self.run_id
        preload_vision(self.mode, self.runtime_options.get("vision_backend"))
        await deploy_runtime(self.llm_agent, mode=self.mode, role=self.agent_type,
                             index=1, position=self.agent_position)
        self.monitor = EpisodeMonitor(self.context, output_dir=self.output_dir / "evaluation",
            run_id=self.run_id, structure="SA", config={"mode": self.mode, "role": self.agent_type,
            "options": self.runtime_options, "environment": load_environment_config()})
        guid = getattr(self.llm_agent.agent, "agent_guid", None)
        self.monitor.attach_agent(guid or self.llm_agent.agent.agent_name,
                                  self.llm_agent.agent.agent_name, self.agent_type)

    async def run(self, *, max_steps: int | None = None) -> None:
        reason = "error"
        try:
            await run_episode([self.llm_agent], max_steps=max_steps, monitor=self.monitor)
            reason = self.llm_agent.end_reason or "decision_limit"
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        finally:
            self._last_end_reason = reason
            if self.monitor is not None:
                if reason == "decision_limit":
                    # Segmented SA calls retain the same episode and sampler.
                    await self.monitor.capture_snapshot("run_boundary")
                else:
                    await self.monitor.stop(reason=reason)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from EmbodiedMAS.agent.runtime.vision_protocol import close_and_cleanup
        await close_and_cleanup(self.context, self._close_resources)

    async def _close_resources(self):
        try:
            if self.monitor is not None:
                try:
                    await self.monitor.stop(reason=self._last_end_reason or "closed")
                finally:
                    await self.monitor.close()
        finally:
            if self.llm_agent is not None:
                await close_runtimes([self.llm_agent])


async def demo_run(
    context: WorldContext,
    *,
    mode: str = "text",
    agent_type: str = "FR",
    max_steps: int | None = None,
    output_dir: Optional[Path] = None,
    agent_position: Optional[dict[str, Any]] = None,
    burn_time: float | None = None,
    require_initial_fire: bool = False,
    **runtime_options: Any,
) -> None:
    print(f"[SA] mode={mode} agent_type={agent_type} max_steps={max_steps}")
    mode, agent_type = str(mode).lower(), str(agent_type).upper()
    options = validate_options(context, mode=mode, agent_types=[agent_type],
                               max_steps=max_steps, **runtime_options)
    runner = SARunner(context, mode=mode, agent_type=agent_type, output_dir=output_dir,
                      agent_position=agent_position, **options)
    try:
        await runner.spawn_agent()
        if runner.monitor is not None:
            await runner.monitor.capture_snapshot("pre_ignition")
        runner.environment_setup = await initialize_environment(
            context, start_fire=True, burn_time=get_burn_time() if burn_time is None else burn_time,
            **({"require_initial_fire": True} if require_initial_fire else {}))
        if runner.monitor is not None:
            runner.monitor.record("lifecycle", {"event": "environment_setup", "data": runner.environment_setup})
        await runner.run(max_steps=max_steps)
    except BaseException as exc:
        if runner.monitor is not None:
            runner.monitor.record("lifecycle", {"event": "runner_error", "error_type": type(exc).__name__})
            runner._last_end_reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        raise
    finally:
        await runner.close()
    print("[SA] completed")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified minimal SA runner (text / vision).")
    p.add_argument("--endpoint", default="127.0.0.1:5726")
    p.add_argument("--mode", type=str.lower, choices=("text", "vision"), default="text")
    p.add_argument("--vision-backend", choices=VISUAL_BACKENDS, default=None)
    p.add_argument("--agent-type", type=str.upper, choices=("FR", "RR", "HF", "ER"), default="FR")
    p.add_argument("--max-steps", type=int, default=None, help="Optional debugging decision cap; formal episodes use a time limit")
    p.add_argument("--observation-scope", choices=("local", "global"), default=None)
    p.add_argument("--record-mode", choices=("full_snapshot", "action_first_seen_or_fire_state_change"), default=None)
    p.add_argument("--time-limit-sec", type=float, default=None)
    p.add_argument("--coordination-mode", choices=("none", "D", "C", "H"), default=None)
    p.add_argument("--initial-civilians", type=int, default=None)
    p.add_argument("--initial-fires", type=int, default=None)
    p.add_argument("--show-prompt", action="store_true", help="Print the selected English system prompt without contacting UE or an LLM")
    p.add_argument(
        "--llm-profile", choices=tuple(LLM_PROFILES), default=None,
        help="Model preset for this run; experiment models use separate gateway credentials.",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Run directory for raw evaluation data and camera captures.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.llm_profile is not None:
        select_llm_profile(args.llm_profile)
        print(f"[SA] LLM profile={args.llm_profile} model={LLM_PROFILES[args.llm_profile]['model']}")
    cfg = load_environment_config()
    scope = args.observation_scope or cfg["observation_scope"]
    record_mode = args.record_mode or cfg.get("record_mode")
    cooperation = args.coordination_mode or cfg["coordination_mode"]
    limit = args.time_limit_sec if args.time_limit_sec is not None else cfg.get("time_limit_sec")
    if limit is None or not math.isfinite(limit) or limit <= 0:
        raise ValueError("formal SA episodes require a positive finite time limit")
    brief = dict(cfg["prompt_environment"])
    for supplied, key in ((args.initial_civilians, "initial_civilians"), (args.initial_fires, "initial_fires")):
        if supplied is not None:
            if supplied < 0:
                raise ValueError("initial counts must be nonnegative")
            brief[key] = supplied
    brief["time_limit_sec"] = limit
    if args.show_prompt:
        from EmbodiedMAS.agent.prompt.emas_prompt import build_initial_prompt
        print(build_initial_prompt(args.mode, args.agent_type, cooperation,
                                  observation_scope=scope, record_mode=record_mode,
                                  episode_briefing=brief, vision_backend=args.vision_backend or cfg["vision_backend"]))
        return
    with ts.TongSim(grpc_endpoint=args.endpoint) as ue:
        ue.context.sync_run(
            demo_run(
                ue.context,
                mode=args.mode,
                agent_type=args.agent_type,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                observation_scope=scope, record_mode=record_mode,
                cooperation_type=cooperation, time_limit_sec=limit, episode_briefing=brief,
                vision_backend=args.vision_backend or cfg["vision_backend"],
            )
        )


if __name__ == "__main__":
    main()
