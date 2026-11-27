# FireGround

[Project website](https://zuwq.github.io/FireGround/) · [Dataset](https://huggingface.co/datasets/EMAS4Rescue/FireGround)

## S1R Oracle example

Run one firefighting robot in the S1R rescue scene using local text observations. The simulator preburns for 5 seconds, then gives the robot 300 seconds to act. The three JSON files in `examples/s1r_oracle/` configure the scene, agent task and runtime. A fresh run generates its own trajectories.

## Download the UE environment

The S1R ignition check and agent episode use the UE 3.6 Linux Shipping build. The compressed archive is approximately **2 GB**. [Download the UE environment from Google Drive](https://drive.google.com/file/d/142bx8gi3KW1_9uYDwlMmb3jAImTkM21g/view?usp=drive_link).

Extract the archive and set `--ue-root` in the commands below to the directory containing `TongSim_Lite/Binaries/Linux/TongSim_Lite-Linux-Shipping`. The launcher loads the S1R scene and starts UE automatically.

## Install

From the repository root on Linux x86-64, use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -B tools/run_episode.py --check-only
```

The last command verifies imports, prompt construction and the three configuration files locally. It requires no GPU, UE process or model credentials.

## Start S1R and verify ignition (no model)

Run as a regular user on a machine with a Vulkan-capable GPU. Keep port 5726 free and choose a new output directory outside the build. The launcher makes a temporary private build copy using GNU cp; allow space for the extracted build if your filesystem cannot use reflinks.

Start with UE closed on this port. The command below automatically writes the bundled
`examples/s1r_oracle/scene_config.json` into the **temporary build copy** at
`TongSim_Lite/SceneConfig.json` before starting UE. Leave the scene file in your
downloaded build as supplied.

```bash
python -B tools/run_episode.py --check-fire \
  --ue-root /path/to/extracted/Linux \
  --output-dir ./runs/s1r-fire-check
```

The check waits for a stable scene with two NPCs, confirms zero burning objects,
ignites the scene, and checks for a positive burning count after the five-second
preburn. It writes the counts and scene checksum to `ignition.json` in the output
directory and closes its UE instance. This command needs no model endpoint or API key.

## Run an agent episode

The agent command below uses the same automatic scene setup and starts a new UE instance.
Set the endpoint and model for your OpenAI-compatible Chat Completions service. The live run sends observations to the service and can incur provider charges.

```bash
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_MODEL='your-model-id'
read -rsp 'Model API key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY

python -B tools/run_episode.py \
  --ue-root /path/to/extracted/Linux \
  --output-dir ./runs/s1r-001
```

The launcher starts UE, waits for the scene to load, runs the agent and prints the run ID, model call count, action count and location of `evaluation/summary.json`. The output directory also contains the trajectory and the UE, readiness and episode logs.

## Optional scoring

To calculate the four benchmark metrics for a completed episode, supply the matching S1R free-burning reference (run ID `20261113_234218_892581_772a5a1a`). Pass its directory containing `scene_config.json` and `evaluation/`.

```bash
python -B tools/score_episode.py --episode ./runs/s1r-001 --baseline /path/to/reference-run
```

The runnable example covers S1R local Oracle. Reproducing other paper conditions requires the corresponding scenario, runner and experiment configuration.

## Project website

The React/Vite website is in `web/`. To preview it locally with Node.js 22:

```bash
cd web
npm ci
npm run dev
```

To preview the production build, run `npm run build` followed by `npm run preview` in the same directory. GitHub Pages publishes `web/dist/` when changes reach `main`.
