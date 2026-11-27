from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from .config_loader import ConfigLoader
    from ...spatial_units import finite_xy, world_xy_m
except ImportError:  # pragma: no cover - direct script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from EmbodiedMAS.agent.prompt.src.config_loader import ConfigLoader
    from EmbodiedMAS.agent.spatial_units import finite_xy, world_xy_m


class PromptBuilder:
    """Assemble the selected English mission, observation and action contracts."""

    def __init__(self, loader: ConfigLoader | None = None) -> None:
        self.loader = loader or ConfigLoader()

    def build_initial_prompt(
        self,
        *,
        scenario_config: dict[str, Any],
        agent_profile: dict[str, Any],
        coordination_mode: dict[str, Any],
        action_space: dict[str, Any],
        input_modality: dict[str, Any] | None = None,
        observation_scope: str = "local",
        record_mode: str | None = None,
        episode_briefing: dict[str, Any] | None = None,
        common_system_path: str | Path = "common_system.md",
        task_prior_path: str | Path = "task_prior.md",
        input_modality_path: str | Path = "input_modality.md",
        cognitive_protocol_path: str | Path = "cognitive_protocol.md",
        output_protocol_path: str | Path = "output_protocol.md",
    ) -> str:
        scope = str(observation_scope).strip().lower()
        if scope not in {"local", "global"}:
            raise ValueError(f"Unknown observation_scope: {observation_scope!r}")
        modality = (input_modality or {}).get("input_modality", "text_observation")
        mode = self.resolve_record_mode(scope, modality, record_mode)
        scope_config = self.loader.load_json(f"configs/observation_scopes/{scope}.json")
        record_config = self.loader.load_json(f"configs/record_modes/{mode}.json")
        task_prior = self._render_task_prior(
            self.loader.load_prompt(str(task_prior_path)), scenario_config, episode_briefing,
        )
        blocks = [
            ("Role and decision loop", self.loader.load_prompt(str(common_system_path))),
            ("Mission and public briefing", task_prior),
            ("Agent role", self._format_agent_profile(agent_profile)),
            ("Coordination", self._format_coordination_mode(coordination_mode)),
            ("Observation scope", f"observation_scope={scope}\n{scope_config['scope_prompt']}"),
            ("Observation format", self._render_input_modality(
                self.loader.load_prompt(str(input_modality_path)), input_modality)),
            ("Record semantics", f"entity_record_mode={mode}\n{record_config['record_prompt']}"),
            ("Allowed actions", self._format_action_space(action_space,
                entity_reference="target_ref" if modality in {"vision", "vision_observation"} else "entity_name")),
            ("Decision guidance", self.loader.load_prompt(str(cognitive_protocol_path))),
            ("Action feedback", self.loader.load_prompt("feedback.md")),
            ("Response contract", self.loader.load_prompt(str(output_protocol_path))),
        ]
        return "\n\n".join(f"# {title}\n{body.strip()}" for title, body in blocks)

    @staticmethod
    def resolve_record_mode(scope: str, modality: str, mode: str | None) -> str:
        visual = modality in {"vision", "vision_observation"}
        if scope == "global" and visual:
            raise NotImplementedError("Global visual observations are not implemented")
        selected = (str(mode).strip().lower() if mode is not None else
                    "full_snapshot" if scope == "global" or visual else
                    "action_first_seen_or_fire_state_change")
        if selected not in {"full_snapshot", "action_first_seen_or_fire_state_change"}:
            raise ValueError(f"Unknown record_mode: {mode!r}")
        if selected != "full_snapshot" and (scope == "global" or visual):
            raise ValueError("Global and visual observations require full_snapshot")
        return selected

    def _render_task_prior(self, template: str, scenario_config: dict[str, Any],
                           episode_briefing: dict[str, Any] | None = None) -> str:
        replacements = {
            "SEMANTIC_MAPS": self._format_semantic_maps(scenario_config.get("semantic_maps", [])),
            "EPISODE_BRIEFING": self._format_episode_briefing(episode_briefing),
        }
        for key, value in replacements.items():
            template = template.replace("{" + key + "}", value)
        return template

    @staticmethod
    def _format_episode_briefing(briefing: dict[str, Any] | None) -> str:
        if briefing is not None and not isinstance(briefing, dict):
            raise ValueError("episode_briefing must be a JSON object")
        data = briefing or {}

        def count(key: str, legacy: str) -> str:
            value = data.get(key, data.get(legacy))
            return str(value) if type(value) is int and value >= 0 else "unknown"

        lines = [f"Initial civilians: {count('initial_civilians', 'num_civilians')}; "
                 f"initial fire sources: {count('initial_fires', 'num_fires')}. "
                 "These are starting counts, not current totals or undiscovered target locations."]
        limit = data.get("time_limit_sec")
        if (isinstance(limit, (int, float)) and not isinstance(limit, bool)
                and math.isfinite(limit) and limit > 0):
            lines.append(f"Time limit: {limit:g} seconds. All sensing, deliberation, communication and action time counts. "
                         "task_time.remaining_sec is the budget remaining when the input was assembled; "
                         "it continues to decrease. The runtime ends the episode at the deadline.")
        else:
            lines.append("No time limit was supplied. Use only timing information available at runtime.")
        if data.get("team") is not None:
            lines.append("Team: " + json.dumps(data["team"], ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines)

    def _format_semantic_maps(self, semantic_maps: Any) -> str:
        if not isinstance(semantic_maps, list) or not semantic_maps:
            return "No room map was supplied."
        blocks = []
        for semantic_map in semantic_maps:
            if not isinstance(semantic_map, dict):
                continue
            source_unit = semantic_map.get("coordinate_unit", "uu")
            source_unit = source_unit.strip().lower() if isinstance(source_unit, str) else None
            coordinate_frame = semantic_map.get("coordinate_frame", "world_xy")
            supported = source_unit in {"uu", "m"} and coordinate_frame == "world_xy"
            lines = [
                f"Scene {semantic_map.get('scene_id', 'unknown')}: "
                "coordinate_frame=world_xy; coordinate_unit=m; units_per_meter=1",
                "room_name | navigation_anchor[x,y] | polygon_source",
            ]
            if not supported:
                lines.append("Source coordinates are unsupported; navigation anchors are unavailable.")
            rooms = semantic_map.get("rooms", [])
            if isinstance(rooms, list):
                for room in rooms:
                    if not isinstance(room, dict):
                        continue
                    name = room.get("room_name", "unknown_room")
                    xy = None
                    if supported:
                        xy = (finite_xy(room) if source_unit == "m" else world_xy_m(room))
                    anchor = json.dumps(xy, ensure_ascii=False, separators=(",", ":"))
                    source = room.get("polygon_source", "unknown") if room.get("polygon_xy") else "unknown"
                    lines.append(f"{name} | {anchor} | {source}")
            if len(lines) > 2:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "No room map was supplied."

    @staticmethod
    def _format_agent_profile(agent_profile: dict[str, Any]) -> str:
        return (f"Role: {agent_profile.get('agent_type', 'unknown_agent')}\n"
                f"{agent_profile.get('role_prompt', 'Choose an allowed action for this role.')}")

    @staticmethod
    def _format_coordination_mode(coordination_mode: dict[str, Any]) -> str:
        mode = str(coordination_mode.get("coordination_mode", "none"))
        body = f"Mode: {mode}\n{coordination_mode.get('coordination_prompt', '')}"
        if mode != "none":
            body += ('\nTo submit a message, use {"type":"<type>","priority":"<priority>","message":"<text>"}. '
                     'Choose type from discovery, request_help, status_update and priority from urgent, important, useful. '
                     'Use null when there is nothing useful to send. Submission does not confirm delivery.')
        return body

    @staticmethod
    def _render_input_modality(template: str, input_modality: dict[str, Any] | None) -> str:
        config = input_modality or {}
        return (template.replace("{INPUT_MODALITY}", str(config.get("input_modality", "unspecified")))
                .replace("{INPUT_MODALITY_DESCRIPTION}", str(config.get("modality_prompt", "Use the observations supplied at runtime."))))

    def _format_action_space(self, action_space: dict[str, Any], *,
                             entity_reference: str = "target_ref") -> str:
        lines = [f"Action set: {action_space.get('action_space_id', 'unknown_action_space')}"]
        for action in action_space.get("allowed_actions", []):
            if not isinstance(action, dict):
                continue
            lines.append(f"- {action.get('name', 'unknown_action')}: {action.get('description', '')}")
            subtypes = action.get("subtypes", [])
            if isinstance(subtypes, list) and subtypes:
                for subtype in subtypes:
                    if isinstance(subtype, dict):
                        lines.append(f"  - subtype={subtype.get('name', 'unknown_subtype')}: "
                                     f"{subtype.get('description', '')} "
                                     f"Parameters: {self._format_parameters(subtype.get('parameters', {}))}.")
            else:
                lines[-1] += f" Parameters: {self._format_parameters(action.get('parameters', {}))}."
        return "\n".join(lines).replace("target_ref", entity_reference)

    @staticmethod
    def _format_parameters(parameters: Any) -> str:
        if not isinstance(parameters, dict) or not parameters:
            return "{}"
        return ", ".join(f"{key}={value}" for key, value in parameters.items())

    @staticmethod
    def action_space_name_for_agent(agent_type: str) -> str:
        return f"{agent_type}_actions"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the selected English agent prompt.")
    parser.add_argument("--profile", default="rescue_agent.json")
    parser.add_argument("--coordination", default="none.json")
    parser.add_argument("--input-modality", default="text_observation.json")
    parser.add_argument("--observation-scope", choices=("local", "global"), default="local")
    parser.add_argument("--record-mode", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    loader = ConfigLoader()
    builder = PromptBuilder(loader)
    profile = loader.load_agent_profile(args.profile)
    prompt = builder.build_initial_prompt(
        scenario_config=loader.load_scenario_config("scenario_example.json"),
        agent_profile=profile,
        coordination_mode=loader.load_coordination_mode(args.coordination),
        action_space=loader.load_action_space(f"{builder.action_space_name_for_agent(profile['agent_type'])}.json"),
        input_modality=loader.load_input_modality(args.input_modality),
        observation_scope=args.observation_scope, record_mode=args.record_mode,
    )
    print(f"chars={len(prompt)}")
    if args.show:
        print(prompt)


if __name__ == "__main__":
    main()
