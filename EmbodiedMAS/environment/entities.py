"""Agent 初始化信息：逻辑命名与引擎 spawn 身份。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class SpawnedActor:
    """引擎 spawn 返回的 actor 身份，RPC 统一使用 guid。"""

    guid: str
    name: Optional[str] = None
    class_path: Optional[str] = None

    @classmethod
    def from_spawn_dict(cls, raw: dict[str, Any]) -> SpawnedActor:
        guid = raw.get("id") or raw.get("guid")
        if not guid:
            raise ValueError("spawn actor dict missing 'id' or 'guid' field")
        return cls(
            guid=str(guid),
            name=raw.get("name"),
            class_path=raw.get("class_path"),
        )

    def to_spawn_dict(self) -> dict[str, Any]:
        return {
            "id": self.guid,
            "name": self.name,
            "class_path": self.class_path,
        }


def format_agent_name(
    *,
    modality: Literal["text", "vision"],
    agent_type: str,
    index: int = 1,
) -> str:
    """生成逻辑 agent 名：TextAgent_FR_01 / VisionAgent_RR_02。"""
    prefix = "TextAgent" if modality == "text" else "VisionAgent"
    return f"{prefix}_{str(agent_type).upper()}_{index:02d}"


def validate_agent_name(name: str) -> str:
    """校验非空 name（deploy 必填）。"""
    validated = str(name).strip()
    if not validated:
        raise ValueError("agent name must be a non-empty string")
    return validated


async def spawn_entity(context: Any, *, blueprint: str, name: str, location: Any = None,
                       rotation: Any = None, tags: Any = None, timeout: float = 5.0) -> dict | None:
    """Create one actor; entity identity is owned by its RobotAgent."""
    import tongsim as ts
    transform = ts.Transform(location=location, rotation=rotation) if rotation else ts.Transform(location=location)
    raw = await ts.UnaryAPI.spawn_actor(context.conn, blueprint=blueprint, transform=transform,
                                        name=validate_agent_name(name), tags=tags, timeout=timeout)
    if raw and not raw.get("name"):
        raw = dict(raw, name=name)
    return raw


async def get_entity_state(context: Any, actor_id: str) -> dict | None:
    """Return the existing compact position/forward state."""
    import tongsim as ts
    import time
    from .rpc_diagnostics import clock_details, error_details
    from EmbodiedMAS.evaluation.monitor import record_event
    started = time.monotonic()
    try:
        rpc = ts.UnaryAPI.get_actor_transform
        # Preserve the original SDK timeout; unwrap only its error-swallowing decorator.
        tf = await getattr(rpc, '__wrapped__', rpc)(context.conn, actor_id)
        if not tf:
            raise RuntimeError('get_actor_transform returned no agent pose')
    except Exception as exc:
        completed = time.monotonic()
        try:
            record_event(context, 'observations', dict(event='rpc_failed', valid=False,
                rpc='get_actor_transform', engine_agent_id=actor_id,
                requested_at_monotonic=started, capture_elapsed_sec=completed-started,
                **clock_details(context, at=completed), **error_details(exc)))
        except Exception:
            pass
        raise
    forward = forward_vector(tf.rotation)
    return {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z, "forward": forward}


def forward_vector(rotation: Any) -> Any:
    import tongsim as ts
    v = rotation * ts.Vector3(1.0, 0.0, 0.0)
    return ts.Vector3(*(0.0 if abs(x) < 1e-4 else x for x in (v.x, v.y, v.z)))


def agent_id_to_fguid_bytes(agent_id: str | bytes) -> bytes:
    """Translate SDK spawn IDs to the FGuid byte layout for camera attachment."""
    if isinstance(agent_id, bytes):
        if len(agent_id) != 16:
            raise ValueError(f"agent id must be 16 bytes, got {len(agent_id)}")
        return agent_id
    text = str(agent_id).replace("-", "").strip()
    if len(text) != 32:
        raise ValueError(f"invalid agent id: {agent_id!r}")
    raw = bytes.fromhex(text)
    return raw[0:4][::-1] + raw[4:6][::-1] + raw[6:8][::-1] + raw[8:16]


