"""Lightweight session logs, independent of evaluation and simulator metrics."""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_benchmark_log_dir() -> Path:
    """
    目录用于 LLM memory 文本日志。

    优先级：EMBODIED_AGENT_LOG_DIR → EMBODIED_BENCHMARK_LOG_DIR → DATA_ROOT/memory_logs → outputs/sessions
    （与 tongsim_automation 下 task 工作区内的 llm_tokens 同级：均在 DATA_ROOT 下。）
    """
    explicit = os.environ.get("EMBODIED_AGENT_LOG_DIR") or os.environ.get("EMBODIED_BENCHMARK_LOG_DIR")
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        root = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
        if root:
            p = Path(root).expanduser().resolve() / "memory_logs"
        else:
            p = Path(__file__).resolve().parents[2] / "outputs" / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_agent_id(agent_id: Optional[str]) -> str:
    aid = agent_id or "agent"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in aid)


def session_memory_path(
    log_dir: Path,
    agent_id: Optional[str],
    session_id: str,
) -> Path:
    """Append-only memory snapshot file (same naming idea as before)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_aid = _safe_agent_id(agent_id)
    return log_dir / f"memory_{safe_aid}_{session_id}.txt"


def append_memory_block(
    path: Path,
    *,
    log_round: int,
    memory_lines: list[str],
    session_header: Optional[str] = None,
) -> None:
    recorded_at = datetime.now().isoformat()
    parts: list[str] = [
        f"===== step={log_round} recorded_at={recorded_at} =====",
    ]
    if session_header:
        parts.append(session_header)
    parts.append("## MEMORY")
    parts.extend(memory_lines if memory_lines else ["(empty)"])
    parts.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(parts))


def session_decision_output_path(log_dir: Path, agent_id: Optional[str], session_id: str) -> Path:
    """Raw output and repair evidence, separate from model-visible memory."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"decision_outputs_{_safe_agent_id(agent_id)}_{session_id}.jsonl"


def append_decision_output(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")


def session_sos_event_path(log_dir: Path, agent_id: Optional[str], session_id: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"sos_events_{_safe_agent_id(agent_id)}_{session_id}.jsonl"


def append_sos_event(path: Path, record: dict) -> None:
    record = dict(record, recorded_at=datetime.now().isoformat())
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
