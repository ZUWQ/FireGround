"""Resolve decision-bound references without conflating same-name instances."""
from __future__ import annotations

from typing import Any, Iterable
import tongsim as ts

from EmbodiedMAS.serialization import location_from_perception
from EmbodiedMAS.agent.actions.results import action_result
from EmbodiedMAS.environment.regions import resolve_room_location


class DestroyedTargetError(ValueError):
    """An observed named entity is confirmed destroyed by its engine identity."""

    def __init__(self, target: str):
        self.target = target
        super().__init__(f"Target {target!r} has been destroyed and no longer exists.")

    def result(self) -> dict[str, Any]:
        return action_result("failed", "target_destroyed", {
            "target_ref": self.target, "destroyed": True, "execution_started": False,
            "message": str(self),
        })


class StaleVisualTargetError(ValueError):
    """A remembered/detected reference is not actionable in the current view."""

    code = "visual_target_unavailable"

    def __init__(self, target: str):
        self.public_message = (
            f"Target {target!r} is not grounded in the latest observation. "
            "Reobserve and select an exact current target_ref with valid grounding; "
            "do not reuse a reference from memory or an earlier frame."
        )
        super().__init__(self.public_message)


def validate_current_visual_target(action: dict[str, Any], targets: dict, *,
                                   semantic_maps: Iterable[dict[str, Any]] = ()) -> None:
    """Check exact current vision bindings without resolving hidden world state.

    Image-pixel targets have their own batch/agent validator. Static named rooms
    remain valid navigation destinations, but are not flame targets. A detection
    without a grounded coordinate is deliberately absent from ``targets``.
    """
    name = action.get("name")
    if name not in {"navigate", "suppress_fire"}:
        return
    if name == "navigate" and action.get("subtype") != "entity":
        return
    target = action.get("parameters", {}).get("target")
    if not isinstance(target, str):
        return
    needle = target.strip().casefold()
    if needle and any(str(ref).casefold() == needle for ref in targets):
        return
    if name == "navigate" and resolve_room_location(list(semantic_maps), target) is not None:
        return
    raise StaleVisualTargetError(target)


class TargetResolver:
    def __init__(self):
        self._coordinates: dict[str, tuple[float, float, float]] = {}
        self._entities: tuple[dict[str, Any], ...] = ()
        self._known_entities: dict[str, dict[str, str]] = {}

    def bind_targets(self, targets: dict[str, tuple[float, float, float]], *,
                     entities: Iterable[dict[str, Any]] = ()) -> None:
        """Replace the current decision binding; never merge by display name."""
        self._coordinates = {str(key): tuple(value) for key, value in targets.items()}
        self._entities = tuple(dict(entity) for entity in entities)
        self.remember_entities(self._entities)

    def remember_entities(self, entities: Iterable[dict[str, Any]]) -> None:
        """Keep observed identities for destruction checks, never old coordinates."""
        fields = ("target_ref", "entity_id", "id", "name", "entity_name", "label", "raw_name")
        for entity in entities:
            raw_name = entity.get("raw_name")
            ref = entity.get("entity_id") or entity.get("id") or entity.get("target_ref")
            if isinstance(raw_name, str) and raw_name.strip() and isinstance(ref, str) and ref.strip():
                self._known_entities[ref.casefold()] = {
                    key: value.strip() for key in fields
                    if isinstance(value := entity.get(key), str) and value.strip()
                }

    def _matching_known_entities(self, name: str) -> list[dict[str, str]]:
        key = name.strip().casefold()
        exact, aliases = [], []
        for entity in self._known_entities.values():
            if any(entity.get(field, "").casefold() == key for field in ("target_ref", "entity_id", "id")):
                exact.append(entity)
            if any(entity.get(field, "").casefold() == key for field in ("name", "entity_name", "label")):
                aliases.append(entity)
        return exact or aliases

    def raw_names_for(self, name: str) -> set[str]:
        """Resolve only observed identity metadata, including removed targets."""
        return {entity["raw_name"] for entity in self._matching_known_entities(name)}

    def entity_ids_for(self, name: str) -> set[str]:
        return {str(identity).replace("-", "").casefold()
                for entity in self._matching_known_entities(name)
                if (identity := entity.get("entity_id") or entity.get("id"))}

    def resolve(self, name: str, *, object_list: dict | None = None,
                semantic_maps: list[dict[str, Any]] | None = None) -> Any:
        needle = name.strip()
        if not needle:
            raise ValueError("Target reference must be non-empty")
        key = needle.casefold()
        # Stable references and explicit IDs take precedence over display names.
        coordinates = {ref.casefold(): coordinate for ref, coordinate in self._coordinates.items()}
        if key in coordinates:
            return ts.Vector3(*coordinates[key])

        entries = list(self._entities)
        for collection in ("actor_info", "npc_info"):
            entries.extend(item for item in (object_list or {}).get(collection, [])
                           if isinstance(item, dict))

        def identity(item: dict[str, Any]) -> str:
            return str(item.get("target_ref") or item.get("id") or item.get("entity_id")
                       or item.get("entity_name") or item.get("name") or "").strip()

        def location(item: dict[str, Any]):
            for field in ("target_ref", "id", "entity_id"):
                ref = str(item.get(field) or "").strip().casefold()
                if ref in coordinates:
                    return coordinates[ref]
            value = item.get("location")
            if isinstance(value, (list, tuple)):
                return tuple(float(component) for component in value)
            if value is not None:
                return tuple(location_from_perception(value))
            return None

        exact = {}
        aliases = {}
        for item in entries:
            coordinate = location(item)
            if coordinate is None:
                continue
            ref = identity(item).casefold()
            if any(str(item.get(field) or "").strip().casefold() == key
                   for field in ("target_ref", "id", "entity_id")):
                exact.setdefault(ref, coordinate)
            if any(str(item.get(field) or "").strip().casefold() == key
                   for field in ("entity_name", "name", "label")):
                # Current bindings occur before text history, so duplicate IDs
                # use current coordinates rather than stale historical geometry.
                aliases.setdefault(ref, coordinate)
        candidates = exact or aliases
        if len(candidates) == 1:
            return ts.Vector3(*next(iter(candidates.values())))
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous target {name!r}; use target_ref or ID")
        coordinate = resolve_room_location(semantic_maps or [], needle)
        if coordinate is not None:
            return ts.Vector3(*coordinate)
        raise ValueError(f"Unknown target entity: {name!r}")
