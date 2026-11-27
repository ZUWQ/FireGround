"""Observed entity facts, independent of model-authored cognitive memory."""
from __future__ import annotations

import copy
from typing import Any
from uuid import UUID

from EmbodiedMAS.agent.perception.records import ObservationFrame


def stable_entity_id(entity: dict[str, Any]) -> str | None:
    """Use explicit simulator identity, never a display name or frame label."""
    for field in ("entity_id", "id", "guid"):
        value = entity.get(field)
        if isinstance(value, bytes):
            # TongSim exposes UE FGuid bytes in Windows GUID (little-endian) layout.
            value = str(UUID(bytes_le=value)) if len(value) == 16 else value.hex()
        elif isinstance(value, UUID):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value.replace("-", "") == "0" * 32:
            continue
        # Canonicalize real GUIDs; test/custom sensors may use opaque IDs.
        try:
            identity = UUID(value)
            if identity.int:
                return str(identity)
        except ValueError:
            return value
    # A GUID-only target_ref is stable too; arbitrary visual/frame labels are not.
    reference = entity.get("target_ref")
    if isinstance(reference, str):
        try:
            identity = UUID(reference.strip())
            return str(identity) if identity.int else None
        except ValueError:
            pass
    return None


class SeenObjects:
    """Keep emitted text facts by alias; retain every visual observation.

    Text facts come exclusively from an action's reduced model-visible output.
    Public positions are already in meters and must not be rescaled here.
    Missing entities retain their last emitted state and observation action.
    """

    def __init__(self, modality: str):
        if modality not in {"text", "vision"}:
            raise ValueError(f"Unsupported modality: {modality!r}")
        self.modality = modality
        self._entries: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._recorded_frames: set[tuple[Any, ...]] = set()

    def record_observation(self, payload: dict[str, Any]) -> None:
        """Accumulate only retained text records, in their observation order."""
        if self.modality != "text":
            raise ValueError("Visual facts must be recorded with record(frame)")
        action_id = payload["action_id"]
        frames = [*payload.get("during_action", []), *payload.get("surroundings", [])]
        for frame in frames:
            for room, members in frame.get("rooms", {}).items():
                for group, fields in (("objects", ("location", "burning", "remaining_percent")),
                                      ("npcs", ("location", "health", "fire_affected", "is_safe"))):
                    for name, facts in members.get(group, {}).items():
                        key = (group, name)
                        previous = self._entries.get(key)
                        if (previous is not None
                                and isinstance(previous["last_seen_action"], int)
                                and isinstance(action_id, int)
                                and previous["last_seen_action"] > action_id):
                            # A retry of an older action must not roll back memory.
                            continue
                        self._entries[key] = {
                            "room": room,
                            "facts": {field: copy.deepcopy(facts.get(field)) for field in fields},
                            "last_seen_action": action_id,
                        }

    def record(self, frame: ObservationFrame) -> None:
        """Retain the existing per-frame visual history without text raw facts."""
        if self.modality != "vision":
            raise ValueError("Text facts must be recorded with record_observation(payload)")
        frame_key = (frame.tick, frame.action_id, frame.batch_id,
                     frame.capture_kind, frame.view)
        if frame_key in self._recorded_frames:
            return
        for index, entity in enumerate(frame.entities):
            key = ("observation", *frame_key, index)
            raw = copy.deepcopy(entity)
            reference = entity.get("target_ref") or f"obs_t{frame.tick}_entity_{index}"
            name = entity.get("name") or entity.get("entity_name") or ""
            self._entries[key] = {
                "id": reference,
                "name": name,
                "first_observation": copy.deepcopy(raw),
                "latest_observation": raw,
                "first_seen_action": frame.action_id,
                "last_seen_action": frame.action_id,
            }
        self._recorded_frames.add(frame_key)

    def snapshot(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return detached facts grouped by room for text, or visual records."""
        if self.modality == "vision":
            return copy.deepcopy(list(self._entries.values()))
        rooms: dict[str, Any] = {}
        for (group, name), entry in self._entries.items():
            members = rooms.setdefault(entry["room"], {"objects": {}, "npcs": {}})
            members[group][name] = {
                **copy.deepcopy(entry["facts"]),
                "last_seen_action": entry["last_seen_action"],
            }
        return rooms
