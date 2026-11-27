"""Score one S1R Oracle episode against a separately supplied S1R free-burning reference."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from EmbodiedMAS.evaluation.benchmark_metrics import compute_metrics

EXAMPLE = ROOT / "examples/s1r_oracle"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(episode, baseline):
    baseline_run_id = "20261113_234218_892581_772a5a1a"
    scene = read(EXAMPLE / "scene_config.json")
    for path, structure in ((episode, "SA"), (baseline, "BASELINE")):
        summary = read(path / "evaluation/summary.json")
        meta = read(path / "evaluation/metadata.json")
        if read(path / "scene_config.json") != scene:
            raise ValueError(f"Scene differs from S1R in {path}")
        if not summary.get("run_id") or summary["run_id"] != meta.get("run_id"):
            raise ValueError(f"Inconsistent run IDs in {path}")
        config = meta["config"]
        burn_times = [part["burn_time"] for part in (config, config.get("environment", {}))
                      if "burn_time" in part]
        if not burn_times or any(value != scene["FireRateDelayTime"] for value in burn_times):
            raise ValueError(f"Inconsistent preburn in {path}")
        if not math.isclose(summary["deadline_monotonic"] - summary["started_at_monotonic"],
                            300.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Unexpected scoring horizon in {path}")
        if summary["structure"] != structure or meta["structure"] != structure:
            raise ValueError(f"Unexpected structure in {path}")
        if summary.get("termination_reason") != "time_limit":
            raise ValueError(f"Incomplete episode in {path}")
        if structure == "BASELINE":
            if (summary["run_id"] != baseline_run_id or meta["agents"] or
                    summary["model_call_count"] != 0 or summary["physical_action_count"] != 0):
                raise ValueError("Invalid S1R free-burning reference")
        elif (meta["config"]["mode"] != "text" or meta["config"]["options"]["observation_scope"] != "local"
              or min(summary["model_call_count"], summary["physical_action_count"]) < 1):
            raise ValueError("Expected a local Oracle episode with decisions and actions")
    result = compute_metrics(episode, baseline, horizon_sec=300, lambda_ct=100,
                             initial_phase="pre_ignition", baseline_match_policy="exact")
    if not (result["baseline_scene_exact_match"]
            and result["baseline_timing_matches"]):
        raise ValueError("Scene or timing mismatch")
    fields = ("svr", "hpr", "ct", "ec", "ct_status", "svr_reason", "hpr_reason", "ct_reason", "ec_reason", "warnings")
    return {"agent_run_id": read(episode / "evaluation/summary.json")["run_id"],
            "baseline_run_id": baseline_run_id, **{k: result[k] for k in fields}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Selected S1R free-burning run directory")
    args = parser.parse_args()
    result = score(args.episode, args.baseline)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["warnings"] or any(result[k] is None for k in ("svr", "hpr", "ct", "ec")):
        raise SystemExit("Scoring evidence is incomplete; see reasons above.")


if __name__ == "__main__":
    main()
