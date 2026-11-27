"""Readiness probe and single-episode dispatch for the S1R local Oracle."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import resolve_burn_time, resolve_task_burn_time, validate_scene_config


def load_run_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Resolved task must be a JSON object")
    validate_run_config(config)
    return config


def validate_run_config(config: dict[str, Any]) -> None:
    required = {"runner": "SA", "mode": "text", "agent_types": ["FR"],
                "observation_scope": "local", "record_mode": "full_snapshot",
                "time_limit_sec": 300, "max_steps": None}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Expected the frozen S1R local Oracle SA configuration")
    if len(config.get("agent_positions", [])) != 1:
        raise ValueError("Expected one firefighter spawn position")
    if config.get("pause_during_llm") or config.get("vision_grace_pause"):
        raise ValueError("S1R Oracle uses a continuous episode clock")
    validate_scene_config(config.get("scene_config"))
    if resolve_task_burn_time(config) != 5 or config["scene_config"]["OutfireMultiply"] != 4:
        raise ValueError("Expected a five-second preburn and suppression multiplier 4")


async def run_config(context: Any, config: dict[str, Any], output_dir: Path) -> None:
    """Delegate one episode without duplicating preparation or cleanup logic."""
    validate_run_config(config)
    from EmbodiedMAS.runners.sa import demo_run
    await demo_run(context, mode="text", max_steps=config.get("max_steps"),
                   output_dir=output_dir, burn_time=resolve_task_burn_time(config),
                   require_initial_fire=any(config["scene_config"]["FireRoom"]),
                   observation_scope=config["observation_scope"], record_mode=config["record_mode"],
                   vision_backend=config["vision_backend"], time_limit_sec=config["time_limit_sec"],
                   cooperation_type="none", agent_type=config["agent_types"][0],
                   agent_position=config["agent_positions"][0],
                   episode_briefing={"initial_civilians": "unknown", "initial_fires": "unknown",
                                     "time_limit_sec": config["time_limit_sec"]})


async def check_ready(context: Any, *, timeout_sec: float = 5.0) -> None:
    """Read the registry without refreshing it, spawning agents or igniting fire."""
    import tongsim as ts
    rpc = ts.UnaryAPI.query_info
    result = await asyncio.wait_for(getattr(rpc, "__wrapped__", rpc)(context.conn), timeout=timeout_sec)
    if not isinstance(result, list) or not result or any(not isinstance(row, dict) for row in result):
        raise RuntimeError("UE registry is not ready: expected a nonempty list of actors")


async def check_scene_ready(context: Any, config: dict[str, Any], *, timeout_sec: float = 12.0,
                            stable_for_sec: float = 5.0, poll_interval_sec: float = 0.5) -> dict:
    """Wait for a stable loaded registry and working fire/NPC state endpoints.

    A nonempty registry can appear while streamed rooms are still loading.
    Readiness must complete before deployment and ignition, outside episode time.
    """
    import tongsim as ts
    from EmbodiedMAS.evaluation.monitor import EpisodeMonitor
    previous = None
    stable_since = None
    expected_npcs = sum(config["scene_config"]["NPCRoom"])
    async with asyncio.timeout(timeout_sec):
        while True:
            rpc = ts.UnaryAPI.query_info
            rows = await getattr(rpc, "__wrapped__", rpc)(context.conn)
            EpisodeMonitor._validate("registry", rows)
            if not rows or any(not row.get("id") for row in rows):
                raise RuntimeError("UE scene not ready: registry requires stable actor identities")
            identities = frozenset(str(row["id"]) for row in rows)
            if len(identities) != len(rows):
                raise RuntimeError("UE scene not ready: duplicate actor identities")
            current = time.monotonic()
            if identities != previous:
                previous, stable_since = identities, current
            if current - stable_since >= stable_for_sec:
                values = {}
                for name, method in (("property", "get_obj_residual"), ("fire", "get_burned_area"),
                                     ("npc_health", "get_npc_health"), ("npc_positions", "get_npc_postions")):
                    rpc = getattr(ts.UnaryAPI, method)
                    value = await getattr(rpc, "__wrapped__", rpc)(context.conn)
                    EpisodeMonitor._validate(name, value)
                    values[name] = value
                health, positions = values["npc_health"], values["npc_positions"]
                if len(health) != expected_npcs or set(health) != set(positions):
                    raise RuntimeError(f"UE scene not ready: expected {expected_npcs} NPCs with health and positions")
                if not values["fire"].get("total_num", 0):
                    raise RuntimeError("UE scene not ready: fire object catalog is empty")
                return {"status": "ready", "registry_records": len(rows), "npc_count": len(health),
                        "registry_stable_sec": current - stable_since, "fire": values["fire"]}
            await asyncio.sleep(poll_interval_sec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one resolved EMAS task (normally called by automation)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--environment-config", type=Path)
    parser.add_argument("--check-ready", action="store_true", help="Read-only UE readiness probe; no experiment is run")
    parser.add_argument("--endpoint", default="127.0.0.1:5726")
    args = parser.parse_args(argv)
    if args.check_ready:
        import tongsim as ts
        with ts.TongSim(grpc_endpoint=args.endpoint) as ue:
            if args.config is None:
                ue.context.sync_run(check_ready(ue.context))
            else:
                config = load_run_config(args.config)
                result = ue.context.sync_run(check_scene_ready(ue.context, config))
                print(json.dumps(result), flush=True)
        return
    if any(value is None for value in (args.config, args.output_dir, args.environment_config)):
        parser.error("--config, --output-dir and --environment-config are required for an experiment")
    config = load_run_config(args.config)
    from EmbodiedMAS.environment.resources import validate_oracle_resources
    validate_oracle_resources()
    environment_path = args.environment_config.expanduser().resolve(strict=True)
    os.environ["EMBODIED_ENVIRONMENT_CONFIG"] = str(environment_path)
    from EmbodiedMAS.agent.config.environment_config import load_environment_config
    environment = load_environment_config(reload=True)
    resolve_burn_time(config["scene_config"], environment["burn_time"], label="environment_config.burn_time")
    import tongsim as ts
    with ts.TongSim(grpc_endpoint=args.endpoint) as ue:
        ue.context.sync_run(run_config(ue.context, config, args.output_dir))


if __name__ == "__main__":
    main()
