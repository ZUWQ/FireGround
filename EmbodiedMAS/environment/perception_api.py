"""Engine perception and the existing entity filtering contract."""
from __future__ import annotations

import asyncio
import copy
import re
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import tongsim as ts
from tongsim.core.world_context import WorldContext

from EmbodiedMAS.environment.entity_names import simplify_npc_name

_EMBODIED_MAS_DIR = Path(__file__).resolve().parent
_OBJECT_LIST: dict[str, dict[str, str]] | None = None


def _load_object_list() -> dict[str, dict[str, str]]:
    global _OBJECT_LIST
    if _OBJECT_LIST is not None:
        return _OBJECT_LIST
    from .resources import load_object_catalog
    # Never cache a missing/malformed catalog as an apparently valid empty view.
    mapping = load_object_catalog(_EMBODIED_MAS_DIR / "ObjectList.csv")
    _OBJECT_LIST = mapping
    return mapping


_INSTANCE_SUFFIX_RE = re.compile(r"_C_\d+$")


def _lookup_object_list_row(
    nm: str,
    object_list: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    """按 object_name 匹配 ObjectList；若无则去掉末尾 _C_<数字> 再匹配。"""
    if nm in object_list:
        return object_list[nm]
    base = _INSTANCE_SUFFIX_RE.sub("", nm)
    if base != nm and base in object_list:
        return object_list[base]
    return None


def _transform_actor_entry(
    entry: dict[str, Any],
    object_list: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    """按 ObjectList 过滤/重命名 actor，并扁平化输出。"""
    if not isinstance(entry, dict):
        return None
    actor = entry.get("actor")
    if not isinstance(actor, dict):
        return None

    raw_name = actor.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    row = _lookup_object_list_row(raw_name.strip(), object_list)
    if row is None or row.get("read") != "1":
        return None

    simplified_name = row.get("simplified_name", "")
    if not simplified_name:
        return None

    out = copy.deepcopy(actor)
    out.update(copy.deepcopy({key: value for key, value in entry.items() if key != "actor"}))
    out.update(name=simplified_name, raw_name=raw_name.strip(), label=row.get("label", ""))
    return out


def _transform_npc_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Simplify NPC names; retain raw identity and all simulator state fields."""
    if not isinstance(entry, dict) or not isinstance(entry.get("object_info"), dict):
        return None
    out = copy.deepcopy(entry["object_info"])
    out.update(copy.deepcopy({key: value for key, value in entry.items()
                              if key not in {"object_info", "position"}}))
    out["raw_name"] = out.get("name", "")
    out["name"] = simplify_npc_name(out["raw_name"])
    out["location"] = copy.deepcopy(entry.get("position"))
    return out


def _postprocess_embodied_perception(result: dict) -> dict:
    """按 ObjectList 处理 actor；简化 NPC 名称，保留内部身份与完整状态。"""
    object_list = _load_object_list()

    raw_actors = result.get("actor_info") or []
    result["actor_info"] = [
        transformed
        for entry in raw_actors
        if (transformed := _transform_actor_entry(entry, object_list)) is not None
    ]

    raw_npcs = result.get("npc_info") or []
    result["npc_info"] = [
        transformed
        for entry in raw_npcs
        if (transformed := _transform_npc_entry(entry)) is not None
    ]

    return result


def _filter_destroyed_from_embodied_perception(result: dict, destroyed_names: frozenset[str]) -> None:
    """Filter exact engine instance names, never an entire simplified category."""
    for kind in ("actor_info", "npc_info"):
        result[kind] = [entry for entry in result.get(kind, [])
                        if str(entry.get("raw_name", entry.get("name", ""))).strip()
                        not in destroyed_names]


def _evaluation_identity(value: Any) -> str | None:
    """Canonicalize UE FGuid identity without confusing it with frame labels."""
    if isinstance(value, bytes):
        value = str(UUID(bytes_le=value)) if len(value) == 16 else value.hex()
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        identity = UUID(text)
        return str(identity) if identity.int else None
    except ValueError:
        return text


def record_visible_entities(context: Any, agent_id: Any, result: dict | None, *,
                            requested_at_monotonic: float | None = None,
                            error: Exception | None = None, **metadata: Any) -> None:
    """Save evaluation-only visibility; never publish it through a sensor frame.

    Object IDs and NPC IDs are separate: exploration of objects must not count
    residents, detector frame labels, or the scene's initial registry as views.
    """
    if getattr(context, "_emas_monitor", None) is None:
        return
    from EmbodiedMAS.evaluation.monitor import record_event

    def identities(kind: str) -> list[str]:
        ids = set()
        for row in (result or {}).get(kind, []):
            if not isinstance(row, dict):
                continue
            identity = _evaluation_identity(row.get("entity_id") or row.get("id") or row.get("guid"))
            if identity is not None:
                ids.add(identity)
        return sorted(ids)

    completed = time.monotonic()
    objects = identities("actor_info") if error is None else None
    npcs = identities("npc_info") if error is None else None
    record_event(context, "observations", {
        "type": "visibility_sample", "engine_agent_id": _evaluation_identity(agent_id),
        "scope": "local", "source": "ue_visible_entities",
        "requested_at_monotonic": requested_at_monotonic,
        "completed_at_monotonic": completed,
        "captured_at_monotonic": completed,
        **metadata, "valid": error is None,
        "visible_object_ids": objects, "visible_entity_ids": objects,
        "visible_npc_ids": npcs,
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
    })


class PerceptionCaptureError(RuntimeError):
    """An unavailable or invalid sensor response, distinct from a valid empty view."""


class PerceptionInfo:
    """获取 Agent 周围的 Actor 和 NPC 感知信息。"""
    
    def __init__(self, context: WorldContext):
        self._context = context
        self._conn = context.conn
        self._last_destroyed_perception_names: frozenset[str] = frozenset()

    def filter_destroyed_cache(self, cache: dict[str, Any]) -> None:
        _filter_destroyed_from_embodied_perception(cache, self._last_destroyed_perception_names)

    async def get_perception(
        self,
        agent_id: bytes | str,
        timeout: float = 5.0,
        *,
        filter_destroyed: bool = True,
    ) -> dict:
        """获取指定 Agent 周围的感知信息。``agent_id`` 须为引擎 GUID 字符串或 16 字节 FGuid。"""
        self._last_destroyed_perception_names = frozenset()
        try:
            # The SDK decorator turns RPC errors into an empty view. Call its
            # wrapped implementation so a sensor failure cannot look successful.
            rpc = ts.UnaryAPI.get_embodied_perception
            result = await getattr(rpc, "__wrapped__", rpc)(self._conn, agent_id, timeout=timeout)
            if not isinstance(result, dict) or any(
                not isinstance(result.get(kind), list) for kind in ("actor_info", "npc_info")
            ):
                raise ValueError("Invalid local perception response")
            result = _postprocess_embodied_perception(copy.deepcopy(result))
            if filter_destroyed:
                rpc = ts.UnaryAPI.get_destroyed_objects
                destroyed = await getattr(rpc, "__wrapped__", rpc)(self._conn)
                if not isinstance(destroyed, list):
                    raise ValueError("Invalid destroyed-object response")
                self._last_destroyed_perception_names = frozenset(
                    str(name).strip() for name in destroyed if str(name).strip()
                )
                self.filter_destroyed_cache(result)
            return result
        except Exception as exc:
            raise PerceptionCaptureError(f"Local perception capture failed: {exc}") from exc

    async def capture_evaluation_visibility(self, agent_id: Any, *, timeout: float = 5.0,
                                            **metadata: Any) -> None:
        """Query the current visual field for statistics without touching caches.

        This side channel is enabled only by an episode monitor. The response is
        reduced to IDs and discarded; no labels, target bindings, or agent-visible
        observations are populated from it. Sampling failure is a missing sample,
        not a successful empty view and not a visual perception failure.
        """
        if getattr(self._context, "_emas_monitor", None) is None:
            return
        started = time.monotonic()
        try:
            rpc = ts.UnaryAPI.get_embodied_perception
            result = await getattr(rpc, "__wrapped__", rpc)(self._conn, agent_id, timeout=timeout)
            if not isinstance(result, dict) or any(
                not isinstance(result.get(kind), list) for kind in ("actor_info", "npc_info")
            ):
                raise ValueError("Invalid evaluation visibility response")
            result = _postprocess_embodied_perception(copy.deepcopy(result))
        except Exception as exc:
            record_visible_entities(self._context, agent_id, None,
                                    requested_at_monotonic=started, error=exc, **metadata)
        else:
            record_visible_entities(self._context, agent_id, result,
                                    requested_at_monotonic=started, **metadata)

    def sos_diagnostics_for(self, packet: dict) -> dict:
        """Side-channel audit metadata; never add diagnostics to sensor packets."""
        if packet is getattr(self, "_sos_last_packet", None):
            return dict(getattr(self, "_sos_last_trace", {}))
        return {}

    async def receive_sos(self, agent_ids: list[bytes | str | dict]):
        """Own/drain the real stream and optionally audit raw transport evidence.

        rpc_created is local call creation, not a server subscription acknowledgment.
        The callback and side-channel trace never change the sensor packet schema.
        """
        from uuid import uuid4
        from tongsim_lite_protobuf.embodied_mas_pb2 import NPCSOSRequest
        from tongsim_lite_protobuf.embodied_mas_pb2_grpc import EmbodiedMASStub
        from tongsim_lite_protobuf.object_pb2 import ObjectId
        from EmbodiedMAS.environment.entities import agent_id_to_fguid_bytes

        stream_id = uuid4().hex
        def audit(event, **details):
            callback = getattr(self, "_sos_event_callback", None)
            if callback is not None:
                try:
                    callback(event, dict(sos_stream_id=stream_id, **details))
                except Exception as exc:
                    print(f"[SOSListener] Transport audit unavailable: {exc}")

        ids = [value.get("guid") if isinstance(value, dict) else value for value in agent_ids]
        request = NPCSOSRequest(agent_id=[ObjectId(guid=agent_id_to_fguid_bytes(value)) for value in ids])
        call = self._conn.get_stub(EmbodiedMASStub).NPCSOSInfo(request, timeout=100.0)
        audit("rpc_created", timeout_sec=100.0, server_acknowledged=False,
              requested_receiver_bytes_le_hex=[item.guid.hex() for item in request.agent_id])
        response_index = 0
        try:
            async for response in call:
                response_index += 1
                audit("raw_response", response_index=response_index,
                      signal_count=len(response.npc_sos_info), empty_response=not response.npc_sos_info)
                for signal_index, signal in enumerate(response.npc_sos_info):
                    trace = dict(sos_stream_id=stream_id, response_index=response_index,
                                 signal_index=signal_index,
                                 signal_id=f"{stream_id}:{response_index}:{signal_index}")
                    audit("raw_signal", **{key: value for key, value in trace.items() if key != "sos_stream_id"},
                          receiver_bytes_le_hex=signal.agent_id.guid.hex(),
                          orientation_count=len(signal.orientation), distance_count=len(signal.distance))
                    packet = {
                        "agent_id": signal.agent_id.guid,
                        "orientations": [{"roll": item.roll_deg, "pitch": item.pitch_deg,
                                          "yaw": item.yaw_deg} for item in signal.orientation],
                        "distances": list(signal.distance),
                    }
                    self._sos_last_packet, self._sos_last_trace = packet, trace
                    yield packet
        except asyncio.CancelledError:
            audit("rpc_cancelled")
            raise
        except Exception as exc:
            audit("rpc_error", reason=type(exc).__name__, detail=str(exc))
            raise
        finally:
            call.cancel()
            try:
                async with asyncio.timeout(2.0):
                    code = await call.code()
                audit("rpc_closed", status_code=str(code))
            except TimeoutError:
                audit("rpc_close_timeout")
