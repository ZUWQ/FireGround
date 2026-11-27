"""Explicit construction of robot bodies and perception components.

Vision imports stay inside their branch, so a text agent does not load model
libraries or initialize camera perception.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def create_components(context: Any, *, agent_type: str, modality: str,
                      log_dir: Optional[Path] = None, robot: Any = None,
                      perception: Any = None, interval_sec: float = 2.0,
                      vision_backend: str = "grounded_sam",
                      observation_scope: str = "local", record_mode: str | None = None):
    agent_type = str(agent_type).upper()
    modality = str(modality).lower()
    if agent_type not in {"FR", "RR", "HF", "ER"}:
        raise ValueError(f"Unsupported agent_type: {agent_type!r}")
    if modality not in {"text", "vision"}:
        raise ValueError(f"Unsupported modality: {modality!r}")
    from EmbodiedMAS.agent.perception.adapters import ObservationAdapter
    # Validate the full capability combination before constructing bodies/models.
    ObservationAdapter(modality, observation_scope=observation_scope, record_mode=record_mode)
    if robot is None:
        from EmbodiedMAS.agent.robot import RobotAgent
        robot = RobotAgent(context, agent_type=agent_type, modality=modality, log_dir=log_dir)
    if perception is None:
        if modality == "text":
            if observation_scope == "global":
                from EmbodiedMAS.agent.perception.global_oracle import GlobalOraclePerceptionModule
                perception = GlobalOraclePerceptionModule(context, interval_sec=interval_sec)
            else:
                from EmbodiedMAS.agent.perception.text import PerceptionModule
                perception = PerceptionModule(context, interval_sec=interval_sec)
        else:
            from EmbodiedMAS.agent.perception.vision import VisionPerceptionModule
            perception = VisionPerceptionModule(context, interval_sec=interval_sec, backend=vision_backend)
    return robot, perception


def create_agent(context: Any, *, agent_type: str = "FR", modality: str = "text",
                 log_dir: Optional[Path] = None, vision_backend: str | None = None,
                 observation_scope: str | None = None, record_mode: str | None = None,
                 cooperation_type: str | None = None, time_limit_sec: float | None = None,
                 episode_briefing: dict[str, Any] | None = None, message_sender: Any = None,
                 vision_grace_pause: bool | None = None):
    if vision_grace_pause is not None and type(vision_grace_pause) is not bool:
        raise ValueError("vision_grace_pause must be a boolean")
    if modality.lower() == "text" and vision_grace_pause:
        raise ValueError("vision_grace_pause requires vision; Oracle is unchanged")
    if modality.lower() == "text":
        from EmbodiedMAS.agent.text_agent import TextAgent
        return TextAgent(context, agent_type=agent_type, log_dir=log_dir,
                         observation_scope=observation_scope, record_mode=record_mode,
                         cooperation_type=cooperation_type, time_limit_sec=time_limit_sec,
                         episode_briefing=episode_briefing, message_sender=message_sender)
    if modality.lower() == "vision":
        from EmbodiedMAS.agent.vision_agent import VisionAgent
        return VisionAgent(context, agent_type=agent_type, log_dir=log_dir, vision_backend=vision_backend,
                           **({"vision_grace_pause": vision_grace_pause} if vision_grace_pause is not None else {}),
                           observation_scope=observation_scope, record_mode=record_mode,
                           cooperation_type=cooperation_type, time_limit_sec=time_limit_sec,
                         episode_briefing=episode_briefing, message_sender=message_sender)
    raise ValueError(f"Unsupported modality: {modality!r}")
