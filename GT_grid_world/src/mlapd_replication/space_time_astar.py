"""Deterministic space-time A* and sequential prioritized routing.

The static A* implementation estimates travel distance for task assignment.
This module adds a time coordinate and reserves vertices and directed edges so
agents cannot occupy one cell together or swap across one edge in opposite
directions during the same tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Protocol

from ..path_finding_algorithms.a_star import manhattan_distance
from .models import Coordinate, WorldState


class StaticGraph(Protocol):
    """The static-map interface needed by space-time A*."""

    def get_if_obstacle(self, node: Coordinate) -> bool: ...

    def get_neighbors(
        self, node: Coordinate, ignore_robots: bool = True
    ) -> list[Coordinate]: ...


@dataclass(frozen=True, slots=True)
class TimedPath:
    """A path whose index is its offset from ``start_time``."""

    agent_id: int
    start_time: int
    locations: tuple[Coordinate, ...]

    @property
    def end_time(self) -> int:
        return self.start_time + len(self.locations) - 1

    @property
    def travel_distance(self) -> int:
        return sum(
            first != second for first, second in zip(self.locations, self.locations[1:])
        )

    def location_at(self, time: int) -> Coordinate:
        if time < self.start_time:
            raise ValueError("time precedes this path")
        offset = min(time - self.start_time, len(self.locations) - 1)
        return self.locations[offset]


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Routes reserved in agent-ID priority order for one planning batch."""

    paths: dict[int, TimedPath]
    planning_seconds: float


def space_time_astar(
    graph: StaticGraph,
    world: WorldState,
    *,
    agent_id: int,
    start: Coordinate,
    goal: Coordinate,
    start_time: int,
    max_time: int,
    allow_partial: bool = False,
) -> list[Coordinate] | None:
    """Find a shortest conflict-free path from ``start`` to ``goal``.

    The result includes its start and goal. A move costs one tick; waiting in
    place is also allowed. Reservations already stored on ``world`` belong to
    higher-priority agents. ``max_time`` is an absolute time bound.  With
    ``allow_partial=True``, a route that cannot reach its goal by that bound
    ends at its closest reachable position instead of failing outright.
    """
    if start_time > max_time:
        raise ValueError("start_time must not exceed max_time")
    if graph.get_if_obstacle(start) or graph.get_if_obstacle(goal):
        return None
    if not _vertex_is_free(world, agent_id, start, start_time):
        return None

    # (f, h, insertion_order, time, location). The insertion counter gives
    # deterministic behaviour when the other priorities tie.
    frontier: list[tuple[int, int, int, int, Coordinate]] = []
    order = count()
    initial_heuristic = manhattan_distance(start, goal)
    heappush(
        frontier,
        (initial_heuristic, initial_heuristic, next(order), start_time, start),
    )
    came_from: dict[tuple[Coordinate, int], tuple[Coordinate, int]] = {}
    visited: set[tuple[Coordinate, int]] = {(start, start_time)}

    best_partial: tuple[int, int, tuple[Coordinate, int]] | None = None
    while frontier:
        _, heuristic, order_value, time, location = heappop(frontier)
        state = (location, time)
        if location == goal:
            return _reconstruct_timed_path(came_from, state)
        if time == max_time:
            candidate = (heuristic, order_value, state)
            if best_partial is None or candidate[:2] < best_partial[:2]:
                best_partial = candidate
            continue

        next_time = time + 1
        # Waiting is last to preserve a simple N, E, S, W, wait tie order.
        for next_location in (*graph.get_neighbors(location, True), location):
            if not _transition_is_free(
                world,
                agent_id=agent_id,
                previous=location,
                next_location=next_location,
                next_time=next_time,
            ):
                continue
            next_state = (next_location, next_time)
            if next_state in visited:
                continue
            visited.add(next_state)
            came_from[next_state] = state
            heuristic = manhattan_distance(next_location, goal)
            heappush(
                frontier,
                (
                    next_time - start_time + heuristic,
                    heuristic,
                    next(order),
                    next_time,
                    next_location,
                ),
            )

    if allow_partial and best_partial is not None:
        return _reconstruct_timed_path(came_from, best_partial[2])
    return None


def plan_prioritized_paths(
    graph: StaticGraph,
    world: WorldState,
    *,
    start_time: int,
    end_time: int,
) -> RoutingResult | None:
    """Plan every agent's scheduled stops in deterministic ID order.

    Each completed route waits at its final cell until ``end_time``. Those
    waits are reservations too, so later agents cannot plan through a parked
    agent. The function clears stale reservations before planning; callers
    therefore receive one self-contained reservation table per batch.
    """
    if start_time > end_time:
        raise ValueError("start_time must not exceed end_time")

    started = perf_counter()
    world.vertex_occupancy.clear()
    world.edge_occupancy.clear()
    paths: dict[int, TimedPath] = {}

    for agent_id, agent in sorted(world.agents.items()):
        route = [agent.location]
        current_location = agent.location
        current_time = start_time

        for event in agent.schedule:
            segment = space_time_astar(
                graph,
                world,
                agent_id=agent_id,
                start=current_location,
                goal=event.location,
                start_time=current_time,
                max_time=end_time,
                allow_partial=True,
            )
            if segment is None:
                return None
            route.extend(segment[1:])
            current_location = segment[-1]
            current_time += len(segment) - 1
            if current_location != event.location:
                # The remaining horizon cannot reach this stop.  The route has
                # already been extended through end_time, so future stops must
                # wait for the next batch replan.
                break

        # An idle agent remains where it is. Reserve the final position to make
        # the occupancy model explicit for the other agents.
        route.extend([current_location] * (end_time - current_time))
        world.reserve_path(agent_id, route, start_time)
        paths[agent_id] = TimedPath(
            agent_id=agent_id,
            start_time=start_time,
            locations=tuple(route),
        )

    return RoutingResult(paths=paths, planning_seconds=perf_counter() - started)


def _vertex_is_free(
    world: WorldState, agent_id: int, location: Coordinate, time: int
) -> bool:
    owner = world.vertex_occupancy.get(time, {}).get(location)
    return owner is None or owner == agent_id


def _transition_is_free(
    world: WorldState,
    *,
    agent_id: int,
    previous: Coordinate,
    next_location: Coordinate,
    next_time: int,
) -> bool:
    if not _vertex_is_free(world, agent_id, next_location, next_time):
        return False
    reverse_owner = world.edge_occupancy.get(next_time - 1, {}).get(
        (next_location, previous)
    )
    return reverse_owner is None or reverse_owner == agent_id


def _reconstruct_timed_path(
    came_from: dict[tuple[Coordinate, int], tuple[Coordinate, int]],
    current: tuple[Coordinate, int],
) -> list[Coordinate]:
    path = [current[0]]
    while current in came_from:
        current = came_from[current]
        path.append(current[0])
    path.reverse()
    return path
