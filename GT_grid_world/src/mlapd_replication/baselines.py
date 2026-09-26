"""Paper baseline task-allocation policies.

The paper compares CBDC with two lightweight sequential baselines.  This file
implements their documented allocation rules only; both use the same
space-time prioritized router from :mod:`space_time_astar` later in the
experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..path_finding_algorithms.a_star import astar_path
from .models import Agent, Coordinate, Task, WorldState


class StaticGraph(Protocol):
    def get_if_obstacle(self, node: Coordinate) -> bool: ...

    def get_neighbors(
        self, node: Coordinate, ignore_robots: bool = True
    ) -> list[Coordinate]: ...


@dataclass(frozen=True, slots=True)
class AssignmentDecision:
    """One task accepted by a baseline at the current batch boundary."""

    task_id: int
    agent_id: int
    estimated_cost: int


def assign_tpmc(world: WorldState, graph: StaticGraph) -> tuple[AssignmentDecision, ...]:
    """Assign each pending task to its nearest feasible agent.

    TPMC is the paper's nearest-feasible-agent baseline.  It estimates only
    the static distance from the agent's *current* position to a task pickup,
    then appends that task to the agent's schedule.  Ties use agent ID so a
    frozen trace always produces the same decisions.
    """
    decisions: list[AssignmentDecision] = []
    for task_id in _pending_in_arrival_order(world):
        task = world.tasks[task_id]
        candidates: list[tuple[int, int]] = []
        for agent_id, agent in sorted(world.agents.items()):
            if task.weight > world.remaining_capacity(agent_id):
                continue
            pickup_distance = _shortest_distance(graph, agent.location, task.pickup)
            delivery_distance = _shortest_distance(graph, task.pickup, task.delivery)
            if pickup_distance is None or delivery_distance is None:
                continue
            candidates.append((pickup_distance, agent_id))
        if not candidates:
            continue

        pickup_distance, agent_id = min(candidates)
        world.assign(task_id, agent_id)
        decisions.append(
            AssignmentDecision(task_id, agent_id, pickup_distance)
        )
    return tuple(decisions)


def assign_rmca(world: WorldState, graph: StaticGraph) -> tuple[AssignmentDecision, ...]:
    """Assign each pending task to the feasible agent with least added route cost.

    RMCA differs from TPMC by measuring from the final stop already on an
    agent's schedule, rather than from its current location.  The estimate is
    the added static travel needed to reach this task's pickup and delivery.
    This is an independent implementation of the paper's minimum-estimated-
    cost rule; its route ordering is explicitly tail insertion.
    """
    decisions: list[AssignmentDecision] = []
    for task_id in _pending_in_arrival_order(world):
        task = world.tasks[task_id]
        candidates: list[tuple[int, int]] = []
        for agent_id, agent in sorted(world.agents.items()):
            if task.weight > world.remaining_capacity(agent_id):
                continue
            added_cost = _appended_route_cost(graph, agent, task)
            if added_cost is not None:
                candidates.append((added_cost, agent_id))
        if not candidates:
            continue

        added_cost, agent_id = min(candidates)
        world.assign(task_id, agent_id)
        decisions.append(AssignmentDecision(task_id, agent_id, added_cost))
    return tuple(decisions)


def _pending_in_arrival_order(world: WorldState) -> tuple[int, ...]:
    return tuple(
        sorted(
            world.pending_task_ids,
            key=lambda task_id: (
                world.tasks[task_id].release_time,
                world.tasks[task_id].deadline,
                task_id,
            ),
        )
    )


def _appended_route_cost(
    graph: StaticGraph, agent: Agent, task: Task
) -> int | None:
    last_stop = agent.schedule[-1].location if agent.schedule else agent.location
    to_pickup = _shortest_distance(graph, last_stop, task.pickup)
    pickup_to_delivery = _shortest_distance(graph, task.pickup, task.delivery)
    if to_pickup is None or pickup_to_delivery is None:
        return None
    return to_pickup + pickup_to_delivery


def _shortest_distance(
    graph: StaticGraph, start: Coordinate, goal: Coordinate
) -> int | None:
    path = astar_path(graph, start, goal, ignore_robots=True)
    return None if path is None else len(path) - 1
