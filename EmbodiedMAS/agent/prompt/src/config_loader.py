from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigLoader:
    """Load EMAS prompt templates and configs from disk."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]

    def load_json(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.root_dir / relative_path
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return data

    def load_text(self, relative_path: str | Path) -> str:
        path = self.root_dir / relative_path
        return path.read_text(encoding="utf-8")

    def load_prompt(self, name: str) -> str:
        return self.load_text(Path("templates") / name)

    def load_agent_profile(self, name: str) -> dict[str, Any]:
        return self.load_json(Path("configs") / "agent_profiles" / name)

    def load_action_space(self, name: str) -> dict[str, Any]:
        return self.load_json(Path("configs") / "action_spaces" / name)

    def load_coordination_mode(self, name: str) -> dict[str, Any]:
        return self.load_json(Path("configs") / "coordination_modes" / name)

    def load_input_modality(self, name: str) -> dict[str, Any]:
        return self.load_json(Path("configs") / "input_modalities" / name)

    def load_scenario_config(self, name: str) -> dict[str, Any]:
        relative_path = Path("configs") / "scenario_configs" / name
        data = self.load_json(relative_path)
        if "room_config" in data:
            from EmbodiedMAS.environment.room_config import measured_room_map

            source = data["room_config"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("room_config must name a measured RoomConfig JSON file")
            measured = measured_room_map(self.load_json(relative_path.parent / source))
            additional_maps = data.get("semantic_maps", [])
            if not isinstance(additional_maps, list):
                raise ValueError("semantic_maps must be a list")
            data["semantic_maps"] = [measured, *additional_maps]
        return data
