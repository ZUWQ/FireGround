"""Offline paper metrics from recorded evidence, without UE or model imports.

SVR/HPR preserve the paper's raw ratios, including negative cross-run values.
CT is a sampled monotonic-wall-clock estimate, not an engine simulation clock.
"""
from __future__ import annotations

import copy
import gzip
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any

from .active_time import pause_mapper


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


@lru_cache(maxsize=32)
def _field_pattern(key: str):
    return re.compile(r'"' + re.escape(key) + r'"\s*:')


_DECODER = json.JSONDecoder()


def _field(line: str, key: str):
    """Decode one known monitor field, skipping the very large actor registry.

    Monitor world_states are single-line json.dumps objects. The requested keys
    are unique at their recorded locations; nested actor registry is not needed
    for outcome/clock metrics. Malformed required fields are never made zero.
    """
    match = _field_pattern(key).search(line)
    if match is None:
        return None
    try:
        index = match.end()
        while index < len(line) and line[index].isspace():
            index += 1
        return _DECODER.raw_decode(line, index)[0]
    except ValueError:
        return None


def _identity(value: Any) -> str:
    return str(value).replace("-", "").casefold()


def _phase_record(line: str) -> dict:
    prop, health, npcs = _field(line, "property"), _field(line, "npc_health"), _field(line, "npcs")
    health_map = None
    if isinstance(health, dict) and health.get("valid") is True and isinstance(health.get("data"), dict):
        if all(_finite(value) for value in health["data"].values()):
            health_map = {_identity(key): value for key, value in health["data"].items()}
    return {"property": prop.get("data") if isinstance(prop, dict) and prop.get("valid") is True and _finite(prop.get("data")) else None,
            "health_map": health_map,
            "npc_count": len(npcs) if isinstance(npcs, list) else None,
            "npcs": npcs if isinstance(npcs, list) else [],
            "elapsed_sec": _field(line, "elapsed_sec"),
            "after_deadline_sec": _field(line, "after_deadline_sec")}


def _signature(directory: Path):
    result = []
    for name in ("summary.json", "metadata.json", "world_states.jsonl", "observations.jsonl"):
        path = directory / "evaluation" / name
        if name == "world_states.jsonl" and not path.exists() and path.with_suffix(".jsonl.gz").exists():
            path = path.with_suffix(".jsonl.gz")
        try:
            stat = path.stat()
            result.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            result.append(None)
    try:
        stat = (directory / "scene_config.json").stat()
        result.append((stat.st_mtime_ns, stat.st_size))
    except OSError:
        result.append(None)
    return tuple(result)


@lru_cache(maxsize=128)
def _episode(directory: str, signature: tuple) -> dict:
    del signature  # The cache key invalidates when any evidence file changes.
    root = Path(directory)
    evaluation = root / "evaluation"
    summary = _read_json(evaluation / "summary.json")
    metadata = _read_json(evaluation / "metadata.json")
    started = summary.get("started_at_monotonic", metadata.get("started_at_monotonic"))
    deadline = summary.get("deadline_monotonic", metadata.get("deadline_monotonic"))
    clock, map_wall, clock_valid = pause_mapper(summary, metadata)
    wall_started, wall_deadline = started, deadline
    if clock_valid:
        started, deadline = map_wall(started), map_wall(deadline)
    phases, fire, warnings = {}, [], []
    initial_objects = summary.get("initial_object_ids")
    malformed_world = 0
    try:
        world_path = evaluation / "world_states.jsonl"
        stream_source = (world_path.open(encoding="utf-8") if world_path.exists() else
                         gzip.open(world_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8"))
        with stream_source as stream:
            for line in stream:
                phase = _field(line, "phase")
                if summary.get("run_id") is not None and _field(line, "run_id") != summary["run_id"]:
                    continue
                if not isinstance(phase, str):
                    malformed_world += 1
                    continue
                if phase in {"pre_ignition", "initial", "final"}:
                    phases[phase] = _phase_record(line)
                    if phase == "initial" and not isinstance(initial_objects, list):
                        catalog = _field(line, "object_catalog")
                        if isinstance(catalog, dict) and catalog.get("valid") is True:
                            initial_objects = [item["id"] for item in catalog.get("objects", [])
                                               if isinstance(item, dict) and item.get("id") is not None]
                query = _field(line, "fire")
                if not isinstance(query, dict):
                    continue
                value = query.get("data")
                count = value.get("burning_num") if isinstance(value, dict) else None
                if query.get("valid") is True and _finite(count) and count >= 0:
                    completed = map_wall(query.get("completed_at_monotonic"))
                    requested = map_wall(query.get("requested_at_monotonic"))
                    if _finite(completed):
                        fire.append({"count": count, "completed": completed, "requested": requested, "phase": phase})
    except OSError:
        warnings.append("world_states_missing_or_unreadable")
    if malformed_world:
        warnings.append("malformed_world_records:" + str(malformed_world))
    visibility = []
    try:
        with (evaluation / "observations.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    warnings.append("malformed_visibility_record")
                    continue
                if not isinstance(record, dict):
                    continue
                if summary.get("run_id") is not None and record.get("run_id") != summary["run_id"]:
                    continue
                if record.get("type") != "visibility_sample" and not any(key in record for key in ("visible_object_ids", "visible_entity_ids")):
                    continue
                time = record.get("completed_at_monotonic", record.get("captured_at_monotonic"))
                if not _finite(time):
                    time = record.get("captured_at_monotonic")
                identifiers = record.get("visible_object_ids", record.get("visible_entity_ids", []))
                visibility.append({"time": map_wall(time), "valid": record.get("valid") is True,
                                   "ids": identifiers if isinstance(identifiers, list) else None})
    except OSError:
        warnings.append("visibility_stream_missing_or_unreadable")
    return {"path": directory, "summary": summary, "metadata": metadata,
            "started": started, "deadline": deadline, "phases": phases, "fire": fire,
            "initial_objects": initial_objects, "visibility": visibility, "warnings": warnings,
            "scene": _read_json(root / "scene_config.json"), "clock": clock,
            "clock_evidence_valid": clock_valid, "wall_started": wall_started, "wall_deadline": wall_deadline}


def _health(episode: dict, phase: str, reference_phase: str) -> float | None:
    records = episode["phases"]
    reference_record = records.get(reference_phase, {})
    reference = reference_record.get("health_map")
    current = records.get(phase, {}).get("health_map")
    if not isinstance(reference, dict) or not reference or not isinstance(current, dict):
        return None
    roster = {_identity(npc["id"]) for npc in reference_record.get("npcs", [])
              if isinstance(npc, dict) and npc.get("id") is not None} or set(reference)
    if any(identity not in reference or identity not in current for identity in roster):
        return None
    return sum(current[identity] for identity in roster)


def baseline_scenes_match(agent_scene: dict, baseline_scene: dict, *, policy="exact", ignore_outfire=False) -> bool:
    """Compare raw scenes under an explicit, narrowly scoped baseline policy.

    ``core6_shared_r`` permits only the frozen benchmark's F-to-R NPC-presence
    difference. ``ignore_outfire`` is explicit and only valid for a proven
    agent-free baseline. Raw configurations are never rewritten, and two R
    configurations with different NPC placement remain different scenes.
    """
    if policy not in {"exact", "core6_shared_r"}:
        raise ValueError("baseline_match_policy must be exact or core6_shared_r")
    if not isinstance(agent_scene, dict) or not agent_scene or not isinstance(baseline_scene, dict) or not baseline_scene:
        return False
    if ignore_outfire:
        # Agent-free trajectories contain no extinguishing action. Preserve raw scenes.
        agent_scene = {k:v for k,v in agent_scene.items() if k != "OutfireMultiply"}
        baseline_scene = {k:v for k,v in baseline_scene.items() if k != "OutfireMultiply"}
    if agent_scene == baseline_scene:
        return True
    if policy != "core6_shared_r":
        return False
    required = {"FireMultiply1", "FireMultiply2", "FireRateDelayTime", "OutfireMultiply",
                "FireSpreadScale", "RoomGenerate", "FireRoom", "NPCRoom", "SafeArea"}
    if ignore_outfire:
        required.discard("OutfireMultiply")
    if not required.issubset(agent_scene) or not required.issubset(baseline_scene):
        return False
    agent_npcs, baseline_npcs = agent_scene.get("NPCRoom"), baseline_scene.get("NPCRoom")
    if (not isinstance(agent_npcs, list) or not isinstance(baseline_npcs, list)
            or len(agent_npcs) != 14 or len(baseline_npcs) != 14
            or any(type(value) is not int or value != 0 for value in agent_npcs)
            or any(type(value) is not int or value not in (0, 1) for value in baseline_npcs)
            or sum(baseline_npcs) != 2):
        return False
    return ({key: value for key, value in agent_scene.items() if key != "NPCRoom"} ==
            {key: value for key, value in baseline_scene.items() if key != "NPCRoom"})


def _recorded_preburn(episode: dict) -> float | None:
    """Verify recorded overrides agree with the raw UE phase-transition delay."""
    delay = episode["scene"].get("FireRateDelayTime")
    config = episode["metadata"].get("config", {})
    if not isinstance(config, dict):
        return None
    values = [config["burn_time"]] if "burn_time" in config else []
    environment = config.get("environment", {})
    if isinstance(environment, dict) and "burn_time" in environment:
        values.append(environment["burn_time"])
    if (not _finite(delay) or delay < 0 or not values
            or any(not _finite(value) or not math.isclose(value, delay, rel_tol=0, abs_tol=1e-6)
                   for value in values)):
        return None
    return float(delay)


def _ratios(agent: dict, baseline: dict | None, phase: str, mismatch: bool) -> dict:
    initial_v = agent["phases"].get(phase, {}).get("property")
    final_v = agent["phases"].get("final", {}).get("property")
    initial_h = _health(agent, phase, phase)
    final_h = _health(agent, "final", phase)
    baseline_v = None if baseline is None else baseline["phases"].get("final", {}).get("property")
    baseline_h = None if baseline is None else _health(baseline, "final", phase)
    baseline_initial_v = None if baseline is None else baseline["phases"].get(phase, {}).get("property")
    baseline_initial_h = None if baseline is None else _health(baseline, phase, phase)
    result = {"reference_phase": phase, "v_init": initial_v, "v_agent": final_v, "v_free": baseline_v,
              "h_init": initial_h, "h_agent": final_h, "h_free": baseline_h,
              "baseline_v_init": baseline_initial_v, "baseline_h_init": baseline_initial_h}
    for name, initial, final, free in (("svr", initial_v, final_v, baseline_v), ("hpr", initial_h, final_h, baseline_h)):
        reason = ("baseline_not_supplied" if baseline is None else
                  "baseline_scene_configuration_mismatch" if mismatch else
                  "missing_or_invalid_boundary_data" if not all(_finite(v) for v in (initial, final, free)) else
                  "nonpositive_baseline_loss" if initial - free <= 0 else None)
        result[name] = None if reason else (final - free) / (initial - free)
        result[name + "_reason"] = reason
    return result


def compute_metrics(artifact_dir, baseline_dir=None, *, horizon_sec=300, lambda_ct=100,
                    initial_phase="pre_ignition", baseline_match_policy="exact") -> dict:
    """Calculate auditable offline metrics from a completed episode's evidence.

    Initial V/H use the agent's specified phase. The baseline's matching initial
    values and phase-specific drift are returned independently. EC includes only
    valid budget-time visibility and the initial readable-object catalog.
    """
    if not _finite(horizon_sec) or horizon_sec <= 0:
        raise ValueError("horizon_sec must be positive and finite")
    if not _finite(lambda_ct) or lambda_ct < 0:
        raise ValueError("lambda_ct must be nonnegative and finite")
    if initial_phase not in {"pre_ignition", "initial"}:
        raise ValueError("initial_phase must be pre_ignition or initial")
    if baseline_match_policy not in {"exact", "core6_shared_r"}:
        raise ValueError("baseline_match_policy must be exact or core6_shared_r")
    root = Path(artifact_dir).resolve()
    agent = _episode(str(root), _signature(root))
    reference = None
    if baseline_dir is not None:
        base_path = Path(baseline_dir).resolve()
        reference = _episode(str(base_path), _signature(base_path))
    verified = bool(reference and agent["scene"] and reference["scene"])
    exact_match = None if not verified else agent["scene"] == reference["scene"]
    agent_free_reference = bool(reference
        and reference["metadata"].get("structure") == "BASELINE"
        and reference["metadata"].get("agents") == {}
        and reference["summary"].get("structure") == "BASELINE"
        and reference["summary"].get("model_call_count") == 0
        and reference["summary"].get("physical_action_count") == 0)
    scene_match = None if not verified else baseline_scenes_match(
        agent["scene"], reference["scene"], policy=baseline_match_policy,
        ignore_outfire=agent_free_reference)
    outfire_equivalent = bool(verified and scene_match and agent_free_reference
        and agent["scene"].get("OutfireMultiply") != reference["scene"].get("OutfireMultiply"))
    mismatch = (bool(reference) and scene_match is not True if baseline_match_policy == "core6_shared_r"
                else verified and not scene_match)
    match_kind = (None if not verified else "exact" if exact_match else
                  "no_agent_outfire_equivalent" if outfire_equivalent and baseline_scenes_match(
                      agent["scene"], reference["scene"], ignore_outfire=True) else
                  "shared_r_property_only" if scene_match else "mismatch")
    ratios = {phase: _ratios(agent, reference, phase, mismatch) for phase in ("pre_ignition", "initial")}
    recorded_horizon = (agent["deadline"] - agent["started"]
                        if _finite(agent["deadline"]) and _finite(agent["started"]) else None)
    baseline_horizon = (reference["deadline"] - reference["started"]
                        if reference is not None and _finite(reference["deadline"]) and _finite(reference["started"]) else None)
    horizon_reason = ("agent_terminal_horizon_mismatch" if recorded_horizon is not None
                      and not math.isclose(recorded_horizon, horizon_sec, abs_tol=1e-6) else
                      "baseline_terminal_horizon_mismatch" if baseline_horizon is not None
                      and not math.isclose(baseline_horizon, horizon_sec, abs_tol=1e-6) else None)
    preburn_match = None
    if baseline_match_policy == "core6_shared_r" and reference is not None:
        if not _finite(recorded_horizon):
            horizon_reason = "agent_terminal_horizon_missing"
        elif not _finite(baseline_horizon):
            horizon_reason = "baseline_terminal_horizon_missing"
        agent_preburn, baseline_preburn = _recorded_preburn(agent), _recorded_preburn(reference)
        preburn_match = bool(agent_preburn is not None and baseline_preburn is not None
                             and math.isclose(agent_preburn, baseline_preburn, rel_tol=0, abs_tol=1e-6))
    if not agent["clock_evidence_valid"] or reference is not None and not reference["clock_evidence_valid"]:
        horizon_reason = "invalid_or_incomplete_pause_clock_evidence"
    timing_reason = horizon_reason or ("baseline_preburn_configuration_mismatch" if preburn_match is False else None)
    if timing_reason is not None:
        for values in ratios.values():
            for metric in ("svr", "hpr"):
                values[metric] = None
                values[metric + "_reason"] = timing_reason
    npc_rooms = agent["scene"].get("NPCRoom")
    no_npcs = (isinstance(npc_rooms, list) and bool(npc_rooms)
               and all(type(value) is int and value == 0 for value in npc_rooms))
    if no_npcs:
        for values in ratios.values():
            values.update(hpr=None, hpr_reason="no_npcs_in_task")
    current = ratios[initial_phase]
    result = {key: current[key] for key in ("svr", "svr_reason", "hpr", "hpr_reason")}
    result.update(initial_phase=initial_phase, ratios_by_phase=ratios,
                  svr_pre_ignition=ratios["pre_ignition"]["svr"], hpr_pre_ignition=ratios["pre_ignition"]["hpr"],
                  svr_initial=ratios["initial"]["svr"], hpr_initial=ratios["initial"]["hpr"],
                  baseline_scene_verified=verified, baseline_scene_matches=scene_match,
                  baseline_scene_exact_match=exact_match, baseline_match_policy=baseline_match_policy,
                  baseline_match_kind=match_kind, baseline_outfire_equivalent=outfire_equivalent,
                  baseline_preburn_matches=preburn_match,
                  baseline_timing_matches=(None if reference is None else timing_reason is None),
                  artifact_dir=str(root), baseline_dir=None if reference is None else reference["path"],
                  clock=agent["clock"], ct_exact=False, horizon_sec=float(horizon_sec), lambda_ct=float(lambda_ct),
                  recorded_horizon_sec=recorded_horizon, baseline_recorded_horizon_sec=baseline_horizon,
                  warnings=list(agent["warnings"]))
    if outfire_equivalent:
        result["warnings"].append("OutfireMultiply ignored for verified zero-agent/zero-action baseline only")
    if match_kind == "shared_r_property_only":
        result["warnings"].append("shared_R_property_baseline_used; NPC health is inapplicable to F")
    if reference is not None and not verified:
        result["warnings"].append("baseline_scene_configuration_not_verified")
    started = agent["started"]
    eligible = []
    if _finite(started):
        eligible = [item for item in agent["fire"] if item["phase"] != "pre_ignition"
                    and item["completed"] >= started
                    and (not _finite(item["requested"]) or item["requested"] >= started)]
    before = [item for item in eligible if item["completed"] - started <= horizon_sec]
    after = [item for item in eligible if item["completed"] - started > horizon_sec]
    last = max(before, key=lambda item: item["completed"], default=None)
    following = min(after, key=lambda item: item["completed"], default=None)
    zero = min((item["completed"] - started for item in before if item["count"] == 0), default=None)
    def boundary(item, side):
        if item is None:
            return None
        elapsed = item["completed"] - started
        return {"count": item["count"], "elapsed_sec": elapsed,
                "gap_to_horizon_sec": horizon_sec - elapsed if side == "before" else elapsed - horizon_sec,
                "phase": item["phase"]}
    uncertain = bool(last and following and last["count"] != following["count"])
    result.update(ct_fire_before=boundary(last, "before"), ct_fire_after=boundary(following, "after"),
                  ct_boundary_uncertain=uncertain, ct_first_zero_sec=zero,
                  ct_summary_first_zero_sec=agent["summary"].get("first_zero_fire_elapsed_sec"),
                  ct_sample_interval_sec=agent["metadata"].get("sample_interval_sec"))
    if zero is not None:
        result.update(ct=zero, ct_status="observed_zero", ct_reason=None, ct_penalty_fire_count=None)
    elif last is not None and (following is not None or last["completed"] - started == horizon_sec):
        result.update(ct=float(horizon_sec + lambda_ct * last["count"]),
                      ct_status="boundary_uncertain" if uncertain else "terminal_penalty_estimate",
                      ct_reason="terminal_count_proxied_by_last_valid_in_budget_sample",
                      ct_penalty_fire_count=last["count"])
    else:
        result.update(ct=None, ct_status="unavailable", ct_reason=("missing_clock_or_valid_in_budget_fire_sample" if last is None else
                                                "no_valid_fire_observation_bracketing_horizon"),
                      ct_penalty_fire_count=None)
    catalog = agent["initial_objects"]
    denominator = {_identity(value) for value in catalog} if isinstance(catalog, list) else set()
    observed, total, invalid, missing_clock, outside = set(), 0, 0, 0, 0
    for record in agent["visibility"]:
        if not _finite(started) or not _finite(record["time"]):
            missing_clock += 1
            continue
        elapsed = record["time"] - started
        if not 0 <= elapsed <= horizon_sec:
            outside += 1
            continue
        total += 1
        if not record["valid"] or record["ids"] is None:
            invalid += 1
            continue
        observed.update(_identity(value) for value in record["ids"] if value is not None)
    ec_reason = ("missing_or_empty_initial_object_catalog" if not denominator else
                 "no_valid_in_budget_visibility_samples" if total - invalid <= 0 else None)
    result.update(ec=None if ec_reason else len(observed & denominator) / len(denominator), ec_reason=ec_reason,
                  ec_scope="initial_ObjectList_read1_objects_excluding_NPCs",
                  ec_catalog_count=len(denominator) if isinstance(catalog, list) else None,
                  ec_observed_count=len(observed & denominator), ec_ids_outside_catalog=len(observed - denominator),
                  ec_visibility_samples_in_budget=total, ec_invalid_samples_in_budget=invalid,
                  ec_samples_outside_budget=outside, ec_samples_missing_clock=missing_clock,
                  pv=None, pv_reason="same_version_repeated_trials_not_supplied")
    if agent["clock"] == "active_wall_seconds":
        result.update(clock_evidence_valid=agent["clock_evidence_valid"],
                      wall_recorded_horizon_sec=(agent["wall_deadline"] - agent["wall_started"]
                          if _finite(agent["wall_deadline"]) and _finite(agent["wall_started"]) else None))
        if not agent["clock_evidence_valid"]:
            result.update(ct=None, ct_status="unavailable", ct_reason="invalid_or_incomplete_pause_clock_evidence",
                          ec=None, ec_reason="invalid_or_incomplete_pause_clock_evidence")
    return copy.deepcopy(result)


__all__ = ["compute_metrics", "baseline_scenes_match"]
