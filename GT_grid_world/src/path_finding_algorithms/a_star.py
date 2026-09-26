"""Deterministic single-agent A* search for the MLAPD replication layer.

This module deliberately has no dependency on the simulator's ``Graph`` class.
Any grid-like object exposing ``get_neighbors(node, ignore_robots)`` and
``get_if_obstacle(node)`` can be searched.  The next step, space-time A*, will
reuse the same movement and heuristic conventions while adding time-indexed
reservations.
"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Protocol, TypeAlias


Coordinate: TypeAlias = tuple[int, int]


class GridGraph(Protocol):
    """Minimum graph interface required by :func:`astar_path`."""

    def get_if_obstacle(self, node: Coordinate) -> bool: ...

    def get_neighbors(
        self, node: Coordinate, ignore_robots: bool = False
    ) -> list[Coordinate]: ...


def manhattan_distance(first: Coordinate, second: Coordinate) -> int:
    """Return the admissible unit-cost heuristic for a 4-connected grid."""

    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def astar_path(
    graph: GridGraph,
    start: Coordinate,
    goal: Coordinate,
    *,
    ignore_robots: bool = True,
) -> list[Coordinate] | None:
    """Find a shortest path from ``start`` to ``goal``.

    The returned path includes both endpoints; therefore its movement cost is
    ``len(path) - 1``.  ``None`` means no path exists or an endpoint is blocked.
    Equal-priority nodes retain the order returned by ``graph.get_neighbors``,
    making a fixed map and start/goal pair reproducible across runs.

    Args:
        graph: A 4-connected grid graph.
        start: Initial grid coordinate.
        goal: Destination grid coordinate.
        ignore_robots: When true, use static obstacles only.  This is the
            intended mode for assignment-cost estimates; dynamic conflicts are
            handled later by space-time A*.
    """
    if graph.get_if_obstacle(start) or graph.get_if_obstacle(goal):
        return None
    if start == goal:
        return [start]

    # (f-score, h-score, insertion-order, node).  The counter makes tie
    # breaking explicit instead of relying on the iteration order of a set.
    frontier: list[tuple[int, int, int, Coordinate]] = []
    order = count()
    start_h = manhattan_distance(start, goal)
    heappush(frontier, (start_h, start_h, next(order), start))

    came_from: dict[Coordinate, Coordinate] = {}
    g_score: dict[Coordinate, int] = {start: 0}
    expanded: set[Coordinate] = set()

    while frontier:
        _, _, _, current = heappop(frontier)
        if current in expanded:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)

        expanded.add(current)
        current_cost = g_score[current]
        for neighbor in graph.get_neighbors(current, ignore_robots):
            if neighbor in expanded:
                continue

            candidate_cost = current_cost + 1
            if candidate_cost >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = candidate_cost
            heuristic = manhattan_distance(neighbor, goal)
            heappush(
                frontier,
                (candidate_cost + heuristic, heuristic, next(order), neighbor),
            )

    return None


def _reconstruct_path(
    came_from: dict[Coordinate, Coordinate], current: Coordinate
) -> list[Coordinate]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
