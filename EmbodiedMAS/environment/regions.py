"""Semantic-map room target lookup."""
from __future__ import annotations
from typing import Any

from .room_geometry import finite_number, polygon_from_room, polygon_contains

def _find_room(semantic_maps: list[dict[str, Any]], room_name: str
               ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(room_name, str):
        raise ValueError("room_name must be a string")
    needle = room_name.strip().casefold()
    if not needle:
        return None
    matches = []
    for semantic_map in semantic_maps:
        if not isinstance(semantic_map, dict):
            continue
        rooms = semantic_map.get("rooms", [])
        if not isinstance(rooms, list):
            continue
        for room in rooms:
            if not isinstance(room, dict):
                continue
            name = room.get("room_name")
            if isinstance(name, str) and name.strip().casefold() == needle:
                matches.append((room, semantic_map))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous room: {room_name!r}")
    return matches[0] if matches else None


def _coordinate_metadata(semantic_map: dict[str, Any]) -> dict[str, Any]:
    if semantic_map.get("coordinate_frame", "world_xy") != "world_xy":
        raise ValueError("Rooms require the world_xy coordinate frame")
    unit = semantic_map.get("coordinate_unit", "uu")
    unit = unit.strip().lower() if isinstance(unit, str) else None
    if unit not in {"uu", "m"}:
        raise ValueError("coordinate_unit must be 'uu' or 'm'")
    scale = finite_number(semantic_map.get("units_per_meter", 100.0),
                          "units_per_meter", positive=True)
    return {"coordinate_unit": unit, "units_per_meter": scale}


def resolve_room_location(
    semantic_maps: list[dict[str, Any]],
    room_name: str,
    *,
    z: float = 1000.0,
) -> tuple[float, float, float] | None:
    """Resolve a unique navigation anchor to finite UE world-unit coordinates.

    Unknown names return None; ambiguous rooms and invalid coordinates raise
    ValueError. An anchor-only target remains navigable.
    """
    match = _find_room(semantic_maps, room_name)
    if match is None:
        return None
    room, semantic_map = match
    metadata = _coordinate_metadata(semantic_map)
    multiplier = metadata["units_per_meter"] if metadata["coordinate_unit"] == "m" else 1.0
    xy = [finite_number(finite_number(room.get(axis), f"room.{axis}") * multiplier,
                        f"world.{axis}") for axis in ("x", "y")]
    return xy[0], xy[1], finite_number(z, "navigation.z")


def resolve_room_spec(
    semantic_maps: list[dict[str, Any]], room_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a unique exploration room with valid geometry and metadata."""
    import copy
    match = _find_room(semantic_maps, room_name)
    if match is None:
        raise ValueError(f"Unknown exploration room: {room_name!r}")
    room, semantic_map = match
    metadata = _coordinate_metadata(semantic_map)
    multiplier = metadata["units_per_meter"] if metadata["coordinate_unit"] == "m" else 1.0
    # Navigation-only anchors must not be accepted as exploration rooms.
    polygon_from_room(room, multiplier)
    return copy.deepcopy(room), metadata


def locate_room(semantic_maps: list[dict[str, Any]], location: Any) -> str | None:
    """Find a unique measured room containing a UE world-unit XY location.

    Location can be an x/y mapping, sequence, or vector with x/y attributes.
    Only measured, valid world_xy polygons are authoritative. Navigation anchors,
    mock_square_2m and unknown sources never establish room membership. Maps may
    use uu or m, matching the existing room geometry contract. Shared boundaries
    and overlapping polygons are ambiguous and return None; Z is not inferred.
    """
    try:
        if isinstance(location, dict):
            xy = (location["x"], location["y"])
        elif isinstance(location, (list, tuple)):
            xy = (location[0], location[1])
        else:
            xy = (location.x, location.y)
        point = tuple(finite_number(value, "location") for value in xy)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(semantic_maps, (list, tuple)):
        return None
    matches = []
    for semantic_map in semantic_maps:
        if not isinstance(semantic_map, dict):
            continue
        try:
            metadata = _coordinate_metadata(semantic_map)
        except ValueError:
            continue
        multiplier = metadata["units_per_meter"] if metadata["coordinate_unit"] == "m" else 1.0
        rooms = semantic_map.get("rooms", [])
        if not isinstance(rooms, list):
            continue
        for room in rooms:
            if not isinstance(room, dict) or room.get("polygon_source") != "measured":
                continue
            name = room.get("room_name")
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                vertices = polygon_from_room(room, multiplier)
            except ValueError:
                continue
            if polygon_contains(vertices, point):
                matches.append(name.strip())
                if len(matches) > 1:
                    return None
    return matches[0] if matches else None
