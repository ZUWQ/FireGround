"""Run the bundled S1R Oracle condition with a private copy of the UE build."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/s1r_oracle"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def preflight():
    from EmbodiedMAS.experiments.run_once import load_run_config
    from EmbodiedMAS.agent.config.environment_config import load_environment_config
    from EmbodiedMAS.environment.resources import validate_oracle_resources
    from EmbodiedMAS.agent.text_agent import TextAgent
    from EmbodiedMAS.runners.sa import SARunner
    from openai import OpenAI

    task = load_run_config(EXAMPLE / "task.json")
    os.environ["EMBODIED_ENVIRONMENT_CONFIG"] = str(EXAMPLE / "environment_config.json")
    env = load_environment_config(reload=True)
    scene = read(EXAMPLE / "scene_config.json")
    if (scene != task["scene_config"]
            or scene["OutfireMultiply"] != 4 or env["SafeArea"] != [scene["SafeArea"]]
            or env["burn_time"] != task["burn_time"] or env["time_limit_sec"] != task["time_limit_sec"]):
        raise ValueError("Inconsistent S1R configuration")
    from EmbodiedMAS.agent.prompt.emas_prompt import build_initial_prompt
    if not build_initial_prompt("text", "FR", "none", observation_scope="local",
                                record_mode=task["record_mode"]):
        raise ValueError("Empty Oracle prompt")
    validate_oracle_resources()


def wait_ready(server, command, env, log):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("UE exited during startup. See ue.log.")
        try:
            probe = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=20)
            if probe.returncode == 0 and server.poll() is None:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
    raise TimeoutError("UE readiness timed out. See readiness.log.")


async def check_ignition(context, task):
    import tongsim as ts
    from EmbodiedMAS.environment.startup import initialize_environment

    rpc = ts.UnaryAPI.get_burned_area
    async with asyncio.timeout(60):
        before = await getattr(rpc, "__wrapped__", rpc)(context.conn)
        if before["burning_num"] != 0:
            raise RuntimeError("Expected a fresh scene before ignition.")
        state = await initialize_environment(context, start_fire=True, burn_time=task["burn_time"],
                                             require_initial_fire=True)
    return {"status": "passed", "preburn_seconds": task["burn_time"],
            "burning_before": before["burning_num"],
            "burning_after_preburn": state["initial_fire_burning_num"],
            "fire_initialized": state["fire_initialized"]}


def run(build, out, *, fire_check=False):
    if not fire_check:
        from EmbodiedMAS.agent.config.llm_config import LLMConfig
        config = LLMConfig()
        config.get_client_kwargs()
        model = config.get_model()
    binary = Path("TongSim_Lite/Binaries/Linux/TongSim_Lite-Linux-Shipping")
    if not (build / binary).is_file() or not os.access(build / binary, os.X_OK):
        raise ValueError(f"Missing executable at {build / binary}")
    if out.is_relative_to(build):
        raise ValueError("Choose an output directory outside the UE build.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 5726))
    out.mkdir(parents=True, exist_ok=False)
    for name in ("task.json", "scene_config.json", "environment_config.json"):
        shutil.copyfile(EXAMPLE / name, out / name)
    if not fire_check:
        (out / "model.json").write_text(json.dumps({"model": model}) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("EMBODIED_")}
    env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT, ROOT / "src"))),
               PYTHONDONTWRITEBYTECODE="1", EMBODIED_ENVIRONMENT_CONFIG=str(out / "environment_config.json"),
               EMBODIED_BENCHMARK_DATA_ROOT=str(out), EMBODIED_AGENT_LOG_DIR=str(out / "sessions"),
               EMBODIED_BENCHMARK_LOG_DIR=str(out / "sessions"))
    command = [sys.executable, "-B", "-m", "EmbodiedMAS.experiments.run_once", "--endpoint",
               "127.0.0.1:5726", "--config", str(out / "task.json")]
    with tempfile.TemporaryDirectory(prefix="ue-", dir=out) as staging:
        private = Path(staging)
        subprocess.run(["cp", "-RL", "--preserve=mode,timestamps", "--reflink=auto",
                        str(build) + "/.", str(private)], check=True)
        scene_path = private / "TongSim_Lite/SceneConfig.json"
        scene_path.parent.chmod(scene_path.parent.stat().st_mode | 0o700)
        scene_path.unlink(missing_ok=True)
        scene = (out / "scene_config.json").read_bytes()
        scene_path.write_bytes(scene)
        if scene_path.read_bytes() != scene:
            raise RuntimeError("Failed to stage the bundled S1R scene.")
        with (out / "ue.log").open("w") as ue_log, (out / "readiness.log").open("w") as ready_log:
            server = subprocess.Popen([private / binary, "TongSim_Lite", "-RenderOffscreen",
                                       "-Unattended", "-nosound", "-stdout", "-FullStdOutLogOutput"],
                                      cwd=private, stdout=ue_log, stderr=subprocess.STDOUT,
                                      env={k:v for k,v in env.items() if not k.startswith("OPENAI_")})
            try:
                wait_ready(server, command + ["--check-ready"], env, ready_log)
                if fire_check:
                    import tongsim as ts
                    with ts.TongSim(grpc_endpoint="127.0.0.1:5726") as ue:
                        result = ue.context.sync_run(check_ignition(ue.context, read(out / "task.json")))
                    result["scene_sha256"] = hashlib.sha256(scene).hexdigest()
                    report = json.dumps(result, indent=2) + "\n"
                    (out / "ignition.json").write_text(report)
                    print(report, end="", flush=True)
                    return
                with (out / "episode.log").open("w") as episode_log:
                    subprocess.run(command + ["--output-dir", str(out), "--environment-config",
                                   str(out / "environment_config.json")], env=env, cwd=ROOT,
                                   stdout=episode_log, stderr=subprocess.STDOUT, timeout=900, check=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    summary_file = out / "evaluation/summary.json"
    summary = read(summary_file)
    if summary.get("termination_reason") != "time_limit":
        raise RuntimeError(f"Episode ended early. See {summary_file}")
    if not summary.get("model_call_count") or not summary.get("physical_action_count"):
        raise RuntimeError(f"No model calls or actions recorded. See {summary_file}")
    print(f"Episode {summary['run_id']} completed with {summary['model_call_count']} model calls "
          f"and {summary['physical_action_count']} actions. Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ue-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-fire", action="store_true",
                        help="Start S1R, ignite and verify fire, then close UE without a model")
    checks.add_argument("--check-only", action="store_true", help="Offline import and configuration checks")
    args = parser.parse_args()
    preflight()
    if args.check_only:
        print("S1R Oracle dependencies, inputs and catalog passed offline preflight.")
        return
    if args.ue_root is None or args.output_dir is None:
        parser.error("--ue-root and --output-dir are required for live execution")
    run(args.ue_root.resolve(strict=True), args.output_dir.resolve(), fire_check=args.check_fire)


if __name__ == "__main__":
    main()
