"""Text agent assembly; shared behavior lives in agent.runtime.loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from EmbodiedMAS.agent.runtime.loop import AgentRuntime


class TextAgent(AgentRuntime):
    def __init__(self, context: Any, *, agent_type: str, log_dir: Optional[Path] = None,
                 observation_scope: str | None = None, record_mode: str | None = None,
                 cooperation_type: str | None = None, time_limit_sec: float | None = None,
                 episode_briefing: dict[str, Any] | None = None, message_sender: Any = None) -> None:
        super().__init__(context, agent_type=agent_type, modality="text", log_dir=log_dir,
                         observation_scope=observation_scope, record_mode=record_mode,
                         cooperation_type=cooperation_type, time_limit_sec=time_limit_sec,
                         episode_briefing=episode_briefing, message_sender=message_sender)


class TextFireAgent(TextAgent):
    def __init__(self, context: Any, *, log_dir: Optional[Path] = None) -> None:
        super().__init__(context, agent_type="FR", log_dir=log_dir)

class TextSaveAgent(TextAgent):
    def __init__(self, context: Any, *, log_dir: Optional[Path] = None) -> None:
        super().__init__(context, agent_type="RR", log_dir=log_dir)

class TextFirefighterAgent(TextAgent):
    def __init__(self, context: Any, *, log_dir: Optional[Path] = None) -> None:
        super().__init__(context, agent_type="HF", log_dir=log_dir)

class TextExploreAgent(TextAgent):
    def __init__(self, context: Any, *, log_dir: Optional[Path] = None) -> None:
        super().__init__(context, agent_type="ER", log_dir=log_dir)
