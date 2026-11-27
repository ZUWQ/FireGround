"""Build a central-controller prompt without changing individual role prompts."""
from __future__ import annotations

import copy
import json
from typing import Any

from EmbodiedMAS.agent.prompt.visual_backend import action_space_for_backend, resolve_visual_backend
from EmbodiedMAS.agent.prompt.src.config_loader import ConfigLoader
from EmbodiedMAS.agent.prompt.src.initial_prompt_builder import PromptBuilder

_PROFILES = {"FR": "firefighting_agent", "RR": "rescue_agent", "HF": "human_firefighter", "ER": "exploration_agent"}


def build_central_prompt(*, team: list[dict[str, Any]], modality: str = "text",
                         observation_scope: str = "local", record_mode: str | None = None,
                         time_limit_sec: float | None = 300,
                         episode_briefing: dict[str, Any] | None = None,
                         vision_backend: str | None = None) -> str:
    if modality not in {"text", "vision"} or observation_scope not in {"local", "global"}:
        raise ValueError("Unsupported central modality or observation scope")
    if not team or any(not isinstance(entry, dict) or entry.get("role") not in _PROFILES
                       or not isinstance(entry.get("agent_id"), str) or not entry["agent_id"] for entry in team):
        raise ValueError("A central team requires robot IDs and supported roles")
    if len({entry["agent_id"] for entry in team}) != len(team):
        raise ValueError("Central robot IDs must be unique")
    selected_backend = resolve_visual_backend(modality, vision_backend)
    raw = modality == "vision" and selected_backend == "raw_vlm"
    hybrid = modality == "vision" and selected_backend == "grounded_sam_pixel"
    loader = ConfigLoader()
    builder = PromptBuilder(loader)
    builder.resolve_record_mode(observation_scope, modality, record_mode)  # Validate caller configuration.
    mode = "full_snapshot"  # CMAS sends complete latest boundaries in every modality.
    brief = copy.deepcopy(episode_briefing or {})
    brief.update(team=copy.deepcopy(team), time_limit_sec=time_limit_sec)
    mission = builder._render_task_prior(loader.load_prompt("task_prior.md"),
        loader.load_scenario_config("scenario_example.json"), brief)
    if observation_scope == "global":
        scope_guidance = (
            "current_observation.robots contains exactly one source robot's latest complete global snapshot "
            "for the whole team. Its robot key identifies the collector, not a visibility restriction. "
            "robot_states and previous_execution_feedback still cover every robot. "
            "Use this same snapshot for current world facts and entity action targets.")
        record_guidance = (
            "Only initial or post-action global snapshots are eligible. Periodic observations are discarded. "
            "At each central decision, only the newest complete snapshot across the team is supplied; "
            "older workers' full-world copies are omitted, including when the newest state contains no live entities. "
            "Read the source robot, action ID and capture time as provenance; robot states are sampled separately.")
    else:
        scope_guidance = (
            "The center may share observations legitimately acquired by any team member. "
            "A union of local views is not global truth; hidden scene state is unavailable. "
            "A robot may execute a target grounded in another member's permitted observation, "
            "but its reported position is historical and can be stale.")
        record_guidance = (
            "Each robot contributes only its latest complete four-view boundary observation, "
            "from initialization or action completion/interruption. Periodic observations are discarded, "
            "including front views collected while the center is thinking. A busy robot retains its last "
            "complete boundary; it is never forced to rotate for a central decision. Images, known_objects "
            "and action targets come from that same boundary. Source robot, action ID and capture time "
            "identify records; different timestamps are not one atomic scene snapshot.")
    blocks = ["""# Central controller
You are the single decision maker controlling the entire robot team in an indoor emergency. Robots sense and execute commands; they have no individual LLM, plans or cognitive memory. Choose exactly one command for each robot at each central decision. All model-visible positions and room anchors use world XY in meters; orientation vectors are dimensionless.
Use robot_states, current_observation, signal_channel, memory_state, previous_execution_feedback and task_time as evidence. Distinguish direct observations, retained history, intentions and confirmed effects. Missing, null and unknown information is unavailable; never fill it with assumed facts.""",
        "# Mission and public briefing\n" + mission,
        "# Robot capabilities\n" + "\n".join(
            f"{entry['agent_id']}: role={entry['role']}; allowed physical actions=" +
            json.dumps([a["name"] for a in loader.load_action_space(_PROFILES[entry["role"]] + "_actions.json")["allowed_actions"]])
            for entry in team),
        """# Asynchronous execution and immediate reassignment
Only one central model call runs at a time. Any robot finishing its action and its own final observation/effect check can trigger the next decision; other robots do not wait for it. Input is a timestamped snapshot. Robots keep executing during sensing and central deliberation, and a robot may finish before your reply arrives. Finished robots remain idle while awaiting commands.
A continue command keeps the current execution state: a running action continues without restarting; a finished or idle robot stays idle. continue is not a physical action, does not last ten seconds and creates no completion event. Consider that continue can leave a robot idle if its previous action finishes during deliberation.
Every new physical action means immediate reassignment. If that robot is busy, its old action is cancelled and cleaned up before the new command starts; if it has finished, the new command starts directly. There is no future-action reservation or queue. Repeating the same physical command still starts a new action; use continue to retain the existing one. Preserve productive actions when their benefit exceeds that of interrupting them; do not reassign merely because an action might finish soon.
execution_status is idle, executing or finalizing. current_action includes its action_id, name, subtype when applicable and parameters. action_elapsed_sec is elapsed time, not a predicted completion time. Action completion, observed effect and completion of the emergency objective are different claims. When all robots are idle and all commands are continue, the controller will recheck after two seconds.
Each robot has a separate action_id sequence; the central step is a decision number, not a shared action round. Completion or failure of one action does not imply completion of any other robot's action. Use the latest available positions and feedback, allowing for movement during inference.
RR cannot suppress fire. Use a fire-capable robot when suppression is needed; the central controller allocates the work directly. No individual robot communication or super-agent assignment output is available.""",
        "# Observation scope\nobservation_scope=" + observation_scope + "\n" +
            loader.load_json(f"configs/observation_scopes/{observation_scope}.json")["scope_prompt"] +
            "\n" + scope_guidance,
        "# Observation records\nentity_record_mode=" + mode + "\n" +
            loader.load_json(f"configs/record_modes/{mode}.json")["record_prompt"] +
            "\n" + record_guidance,
        "# Decision guidance\n" + loader.load_prompt("cognitive_protocol.md").replace("weigh exploration or communication", "weigh exploration or coordination"),
        "# Feedback\n" + "Read status together" + loader.load_prompt("feedback.md").split("Read status together", 1)[-1]]
    if raw:
        blocks.append("# Visual input\n" + loader.load_input_modality("raw_vlm_observation.json")["modality_prompt"] +
            "\nImages are grouped by source robot and capture in the same order as the multimodal input. Copy an actionable frame_ref from the commanded robot\'s own ending views; another robot\'s images may inform reasoning but cannot supply its image target. No detector list or oracle target coordinates are supplied.")
    elif hybrid:
        blocks.append("# Visual input\n" + loader.load_input_modality("sam_pixel_observation.json")["modality_prompt"] +
            "\nImages are grouped by source robot and capture: route RGB only, ending RGB then annotation when supplied. "
            "Entity actions may use exact currently grounded source-scoped target_ref values from any team member. "
            "distance_xy_m is relative only to the frame\'s source robot at its capture time. "
            "For proximity of a commanded robot, use distances_by_robot_m[robot_id] for that robot, "
            "not another robot\'s source distance; null means unknown. "
            "These estimates combine the target\'s frame captured_at with that robot\'s "
            "state snapshot marked robot_states[robot_id].captured_at, so they are not an atomic scene measurement. "
            "Account for stale target observations and robot movement during execution or deliberation; "
            "a small reported distance alone does not guarantee current interaction range. "
            "For image actions copy an actionable frame_ref from the commanded robot's own ending views; "
            "another robot's images may inform reasoning but cannot supply its image target. "
            "suppress_fire.target is either a current grounded target_ref string or {frame_ref,pixel_xy}. "
            "navigate/image uses the same image target object. No oracle target coordinates are supplied.")
    elif modality == "vision":
        visual_guidance = loader.load_input_modality("vision_observation.json")["modality_prompt"].split("\n", 1)[0]
        blocks.append("# Visual input\n" + visual_guidance + "\n" +
            "Images are grouped by source robot and capture: route RGB only, ending RGB then annotation when supplied. "
            "Each image path is listed with the same order as the actual multimodal input. "
            "Entity actions require exact source-scoped target_ref values with grounding_available=true; "
            "never invent references or coordinates. A frame label from another camera or frame is not automatically the same instance. "
            "A fire's attached_to is a visual association with an object/person reference, not a confirmed hidden fire state. "
            "No oracle entity coordinates or precise health are supplied by images. "
            "Visual suppression effects can be unknown even when its command executes.")
    else:
        blocks.append("# Text input\n" + loader.load_input_modality("text_observation.json")["modality_prompt"])
    for role in dict.fromkeys(entry["role"] for entry in team):
        blocks.append("# Physical actions for " + role + "\n" + builder._format_action_space(
            action_space_for_backend(loader.load_action_space(_PROFILES[role] + "_actions.json"), raw=raw, hybrid=hybrid),
            entity_reference="target_ref" if modality == "vision" else "entity_name"))
    example = {"decision_basis": "<team objective and decisive evidence>",
               "robot_actions": {entry["agent_id"]: {"name": "continue", "parameters": {}} for entry in team},
               "memory_update": {"note": None}}
    blocks.append("""# Central response contract
Return exactly one JSON object with exactly decision_basis, robot_actions and memory_update; no Markdown or surrounding prose. decision_basis is one concise team-level sentence. robot_actions must contain every controlled robot ID exactly once, with no extra or missing robots.
Each robot command uses name and parameters. navigate and instruct_civilian also require subtype beside name and parameters, never inside parameters. Other physical actions omit subtype. continue requires exactly {"name":"continue","parameters":{}}. Copy target references and room_name values exactly. Each command must belong to that robot's role capability set or be continue.
No communication, individual decisions, high_level_plan or assignment field is allowed. memory_update contains only note, a string or null. Read memory_state.note as the current episode-local team notebook. memory_state.short_term keeps a fixed window of accepted central decisions with robot_actions and actual available execution feedback; each entry is a team decision, not one robot's action. Entries are removed by age, without an importance policy.
Return note=null to keep the notebook unchanged, note="" to clear it, or a new complete string to replace it. When rewriting, retain still-useful information from the old note and remove obsolete content. Keep the new note within memory_state.note_max_chars characters. This note is part of the same decision response, not a request for a separate summarization call. Observation facts and action records are maintained separately; do not repeat the observation registry. Distinguish observations, attributed reports, inferences and intentions; plans do not establish execution or confirmed outcomes. No cross-episode memory is supplied.
If new SOS arrives during deliberation, only commands for the affected receiving robots are withheld; other valid robot commands can proceed. The pending SOS and reconciliation feedback are shown at the next decision. Repeated delivery of the same signal during one action is coalesced, and a signal you have already read does not itself invalidate your reply. A withheld command does not imply the previous action stopped.
Malformed or role-invalid batches are rejected before any new action or memory update is committed; running actions keep going. previous_decision_feedback identifies errors for the next attempt. Completion during inference alone does not invalidate the whole reply. The following example shows structure only; choose commands using the actual evidence:
""" + json.dumps(example, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(blocks)
