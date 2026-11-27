"""Pure room geometry in UE world units; no simulator calls."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable

Point = tuple[float, float]


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


@dataclass(frozen=True)
class RoomRectangle:
    """An ordered, nondegenerate rectangle, including rotated rectangles."""
    vertices: tuple[Point, ...]
    origin: Point
    axis_u: Point
    axis_v: Point
    width: float
    height: float
    inset: float

    def contains(self, point: Point, *, tolerance: float = 1e-6) -> bool:
        delta = _sub(point, self.origin)
        u, v = _dot(delta, self.axis_u), _dot(delta, self.axis_v)
        return (-tolerance <= u <= self.width + tolerance
                and -tolerance <= v <= self.height + tolerance)

    def contains_path(self, points: Iterable[Point]) -> bool:
        # Rectangles are convex: if every polyline vertex is inside, every
        # connecting segment is inside too. This is not a general polygon test.
        values = tuple(points)
        return bool(values) and all(self.contains(point) for point in values)

    def inset_corners_clockwise(self) -> tuple[Point, ...]:
        m = self.inset
        local = ((m, m), (self.width - m, m),
                 (self.width - m, self.height - m), (m, self.height - m))
        result = tuple((self.origin[0] + u * self.axis_u[0] + v * self.axis_v[0],
                        self.origin[1] + u * self.axis_u[1] + v * self.axis_v[1])
                       for u, v in local)
        cross = self.axis_u[0] * self.axis_v[1] - self.axis_u[1] * self.axis_v[0]
        return result if cross < 0 else (result[0], result[3], result[2], result[1])


def rectangle_from_room(room: dict[str, Any], *, inset_m: float,
                        coordinate_unit: str = "uu", units_per_meter: float = 100.0) -> RoomRectangle:
    scale = finite_number(units_per_meter, "units_per_meter", positive=True)
    inset = finite_number(inset_m, "inset_m", positive=True) * scale
    if coordinate_unit not in {"uu", "m"}:
        raise ValueError("coordinate_unit must be 'uu' or 'm'")
    multiplier = scale if coordinate_unit == "m" else 1.0
    polygon = room.get("polygon_xy")
    if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
        raise ValueError("explore requires an ordered four-vertex rectangular polygon_xy")
    vertices = []
    for index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon_xy vertices must each contain x and y")
        vertices.append(tuple(finite_number(value, f"polygon_xy[{index}]") * multiplier
                              for value in point))
    points = tuple(vertices)
    edge_u, edge_v = _sub(points[1], points[0]), _sub(points[3], points[0])
    width, height = math.hypot(*edge_u), math.hypot(*edge_v)
    tolerance = max(1e-6, max(width, height) * 1e-7)
    if width <= tolerance or height <= tolerance:
        raise ValueError("rectangle edges must have nonzero length")
    if abs(_dot(edge_u, edge_v)) > width * height * 1e-7:
        raise ValueError("polygon_xy must be a rectangle with ordered adjacent vertices")
    expected = (points[1][0] + points[3][0] - points[0][0],
                points[1][1] + points[3][1] - points[0][1])
    if math.dist(points[2], expected) > tolerance:
        raise ValueError("polygon_xy must be an ordered rectangle, not a crossed or arbitrary quadrilateral")
    if width - 2 * inset <= tolerance or height - 2 * inset <= tolerance:
        raise ValueError("room_too_small_for_inset")
    return RoomRectangle(points, points[0],
                         (edge_u[0] / width, edge_u[1] / width),
                         (edge_v[0] / height, edge_v[1] / height),
                         width, height, inset)


def _cross(a: tuple[float, float], b: tuple[float, float],
           c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(point: tuple[float, float], a: tuple[float, float],
                b: tuple[float, float]) -> bool:
    return (abs(_cross(a, b, point)) <= 1e-8
            and min(a[0], b[0]) - 1e-8 <= point[0] <= max(a[0], b[0]) + 1e-8
            and min(a[1], b[1]) - 1e-8 <= point[1] <= max(a[1], b[1]) + 1e-8)


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((ab_c * ab_d < 0 and cd_a * cd_b < 0)
            or _on_segment(c, a, b) or _on_segment(d, a, b)
            or _on_segment(a, c, d) or _on_segment(b, c, d))


def polygon_from_room(room: dict[str, Any], multiplier: float) -> list[tuple[float, float]]:
    """Validate a simple polygon and convert it to UE world units."""
    polygon = room.get("polygon_xy")
    if not isinstance(polygon, (list, tuple)):
        raise ValueError("polygon_xy must be a vertex sequence")
    vertices = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon_xy vertices must contain x and y")
        vertices.append(tuple(finite_number(finite_number(value, "polygon_xy") * multiplier,
                                            "world vertex") for value in point))
    # Accept conventional explicitly closed polygons as well as implicit closure.
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3 or len(set(vertices)) != len(vertices):
        raise ValueError("polygon_xy must have at least three distinct vertices")
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    if abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)) <= 1e-8:
        raise ValueError("polygon_xy must have nonzero area")
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _segments_intersect(a, b, *edges[j]):
                raise ValueError("polygon_xy must not self-intersect")
    return vertices


def polygon_contains(vertices: list[tuple[float, float]], point: tuple[float, float]) -> bool:
    x, y = point
    inside = False
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if _on_segment(point, a, b):
            return True
        if ((a[1] > y) != (b[1] > y)
                and x < a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])):
            inside = not inside
    return inside


def polygon_contains_path(vertices: list[Point], points: Iterable[Point]) -> bool:
    """Check entire segments, including crossings of a concave room's cutouts."""
    points = tuple(points)
    if not points or not all(polygon_contains(vertices, p) for p in points):
        return False
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    for a, b in zip(points, points[1:]):
        delta = _sub(b, a)
        length_sq = _dot(delta, delta)
        if length_sq <= 1e-16:
            continue
        cuts = {0.0, 1.0}
        for c, d in edges:
            edge = _sub(d, c)
            denominator = delta[0] * edge[1] - delta[1] * edge[0]
            if abs(denominator) <= 1e-8:
                if abs(_cross(a, b, c)) <= 1e-8:
                    cuts.update(max(0.0, min(1.0, _dot(_sub(p, a), delta) / length_sq))
                                for p in (c, d))
                continue
            offset = _sub(c, a)
            t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
            u = (offset[0] * delta[1] - offset[1] * delta[0]) / denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                cuts.add(t)
        ordered = sorted(cuts)
        for lo, hi in zip(ordered, ordered[1:]):
            t = (lo + hi) / 2
            if not polygon_contains(vertices, (a[0] + t * delta[0], a[1] + t * delta[1])):
                return False
    return True


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    edge = _sub(b, a)
    t = max(0.0, min(1.0, _dot(_sub(point, a), edge) / _dot(edge, edge)))
    return math.dist(point, (a[0] + t * edge[0], a[1] + t * edge[1]))


@dataclass(frozen=True)
class RoomPolygon:
    """A simple orthogonal room and its validated inward perimeter."""
    vertices: tuple[Point, ...]
    inset_vertices: tuple[Point, ...]

    def contains(self, point: Point) -> bool:
        return polygon_contains(list(self.vertices), point)

    def contains_path(self, points: Iterable[Point]) -> bool:
        return polygon_contains_path(list(self.vertices), points)

    def inset_corners_clockwise(self) -> tuple[Point, ...]:
        points = self.inset_vertices
        area = sum(_cross((0, 0), a, b) for a, b in zip(points, points[1:] + points[:1]))
        return points if area < 0 else (points[0], *reversed(points[1:]))


def _orthogonal_inset(points: list[Point], inset: float) -> tuple[Point, ...]:
    """Exact square erosion of a small orthogonal polygon, without dependencies.

    Offset edges can vanish when a narrow branch collapses. Decompose the
    remaining area at all vertex +/- inset coordinates instead of preserving an
    invalid one-to-one correspondence between source and offset corners. A cell
    survives exactly when the inset square around its centre is inside the room.
    Keep every surviving cell; disconnected or pinched results are rejected.
    """
    def levels(axis: int) -> list[float]:
        values = sorted({point[axis] + sign * inset for point in points for sign in (-1, 1)})
        result = []
        for value in values:
            if not result or value - result[-1] > 1e-7:
                result.append(value)
        return result

    xs, ys = levels(0), levels(1)
    cells = set()
    for i, (left, right) in enumerate(zip(xs, xs[1:])):
        for j, (bottom, top) in enumerate(zip(ys, ys[1:])):
            x, y = (left + right) / 2, (bottom + top) / 2
            square = [(x - inset, y - inset), (x + inset, y - inset),
                      (x + inset, y + inset), (x - inset, y + inset)]
            if polygon_contains_path(points, square + square[:1]):
                cells.add((i, j))
    if not cells:
        raise ValueError("room_too_small_for_inset")

    pending = set(cells)
    stack = [pending.pop()]
    while stack:
        i, j = stack.pop()
        for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if neighbour in pending:
                pending.remove(neighbour)
                stack.append(neighbour)
    if pending:
        raise ValueError("inset_splits_room")

    boundary = set()
    for i, j in cells:
        corners = [(xs[i], ys[j]), (xs[i + 1], ys[j]),
                   (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if (b, a) in boundary:
                boundary.remove((b, a))
            else:
                boundary.add((a, b))
    following = dict(boundary)
    if len(following) != len(boundary) or len(set(following.values())) != len(boundary):
        raise ValueError("inset_splits_room")
    start = min(following)
    ring, current = [], start
    while True:
        ring.append(current)
        current = following[current]
        if current == start:
            break
    if len(ring) != len(boundary):
        raise ValueError("inset_has_multiple_boundaries")
    # Grid intersections on straight edges are not exploration waypoints.
    return tuple(b for a, b, c in zip(ring[-1:] + ring[:-1], ring, ring[1:] + ring[:1])
                 if abs(_cross(a, b, c)) > 1e-8)


def exploration_region_from_room(room: dict[str, Any], *, inset_m: float,
                                 coordinate_unit: str = "uu", units_per_meter: float = 100.0
                                 ) -> RoomRectangle | RoomPolygon:
    """Parallel inward edges; preserve connected remnants when narrow branches vanish."""
    scale = finite_number(units_per_meter, "units_per_meter", positive=True)
    inset = finite_number(inset_m, "inset_m", positive=True) * scale
    if coordinate_unit not in {"uu", "m"}:
        raise ValueError("coordinate_unit must be 'uu' or 'm'")
    points = polygon_from_room(room, scale if coordinate_unit == "m" else 1.0)
    if len(points) == 4:
        return rectangle_from_room(dict(room, polygon_xy=points), inset_m=inset_m,
                                   units_per_meter=scale)
    edges = list(zip(points, points[1:] + points[:1]))
    for a, b in edges:
        dx, dy = _sub(b, a)
        if (dx == 0) == (dy == 0):
            raise ValueError("nonrectangular exploration rooms require orthogonal edges")
    corners = _orthogonal_inset(points, inset)
    polygon_from_room({"polygon_xy": corners}, 1.0)
    if not polygon_contains_path(points, list(corners) + [corners[0]]):
        raise ValueError("room_too_small_for_inset")
    for c, d in zip(corners, corners[1:] + corners[:1]):
        for e, f in edges:
            distance = (0.0 if _segments_intersect(c, d, e, f) else min(
                _point_segment_distance(c, e, f), _point_segment_distance(d, e, f),
                _point_segment_distance(e, c, d), _point_segment_distance(f, c, d)))
            if distance < inset - 1e-6:
                raise ValueError("room_too_small_for_inset")
    return RoomPolygon(tuple(points), tuple(corners))
