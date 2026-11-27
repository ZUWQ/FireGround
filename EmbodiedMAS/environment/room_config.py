"""Load room boundaries and explicit navigation anchors into semantic maps."""
from __future__ import annotations

from typing import Any

from .room_geometry import finite_number, polygon_from_room, polygon_contains


def measured_room_map(config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        raise ValueError("RoomConfig must contain measured rooms")
    rooms = []
    for name, outline in config.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("RoomConfig entries require a room name")
        anchor = None
        if isinstance(outline, dict):
            points = outline.get("vertices")
            point = outline.get("reachable_point")
            if not isinstance(point, dict):
                raise ValueError(f"RoomConfig {name}.reachable_point must contain x and y")
            try:
                anchor = tuple(finite_number(point[axis], f"{name}.reachable_point.{axis}")
                               for axis in ("x", "y"))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid RoomConfig reachable_point for {name}: {exc}") from exc
        else:
            # Older vertex-only files retain their interior-anchor fallback.
            points = outline
        if not isinstance(points, list):
            raise ValueError(f"RoomConfig {name}.vertices must be a vertex list")
        if name == "Safe_Zone" and points == [] and anchor is not None:
            rooms.append({"room_name": name, "x": anchor[0], "y": anchor[1],
                          "polygon_source": "unknown"})
            continue
        try:
            polygon = [[point["x"], point["y"]] for point in points]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid RoomConfig outline for {name}") from exc
        try:
            vertices = polygon_from_room({"polygon_xy": polygon}, 1.0)
        except ValueError as exc:
            raise ValueError(f"Invalid RoomConfig outline for {name}: {exc}") from exc
        if anchor is not None and not polygon_contains(vertices, anchor):
            raise ValueError(f"RoomConfig {name}.reachable_point must lie inside its vertices")
        if anchor is None:
            # Use the area centroid when it is inside the polygon. Concave outlines
            # whose centroid lies outside use an interior scanline span instead.
            edges = list(zip(vertices, vertices[1:] + vertices[:1]))
            cross = [a[0] * b[1] - b[0] * a[1] for a, b in edges]
            area6 = 3 * sum(cross)
            anchor = tuple(sum((a[axis] + b[axis]) * weight
                               for (a, b), weight in zip(edges, cross)) / area6
                           for axis in (0, 1))
            if not polygon_contains(vertices, anchor):
                levels = sorted({p[1] for p in vertices})
                spans = []
                for low, high in zip(levels, levels[1:]):
                    y = (low + high) / 2
                    xs = sorted(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                                for a, b in edges if (a[1] > y) != (b[1] > y))
                    spans.extend((right - left, (left + right) / 2, y)
                                 for left, right in zip(xs[::2], xs[1::2]))
                _, x, y = max(spans)
                anchor = (x, y)
        rooms.append({"room_name": name, "x": anchor[0], "y": anchor[1],
                      "polygon_xy": [list(point) for point in vertices],
                      "polygon_source": "measured"})
    return {"scene_id": "world", "coordinate_frame": "world_xy", "coordinate_unit": "uu",
            "units_per_meter": 100.0, "rooms": rooms}
