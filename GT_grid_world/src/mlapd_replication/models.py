"""Paper-specific MLAPD state models.

These types model the paper's multi-load semantics instead of reusing the
legacy simulator's single-SKU agent state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .trace import ExperimentTrace


Coordinate: TypeAlias = tuple[int, int]


class TaskStatus(str, Enum):
    UNRELEASED = "unreleased"
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    COMPLETED = "completed"
    EXPIRED = "expired"


class ScheduleEventKind(str, Enum):
    PICKUP = "pickup"
    DELIVERY = "delivery"


@dataclass(frozen=True, slots=True)
class Task:
    """One online pickup-and-delivery request from the paper's formulation."""

    task_id: int
    release_time: int
    deadline: int
    pickup: Coordinate
    delivery: Coordinate
    weight: int = 1

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.release_time < 0:
            raise ValueError("release_time must be non-negative")
        if self.deadline < self.release_time:
            raise ValueError("deadline must not precede release_time")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True, slots=True)
class ScheduleEvent:
    """A pickup or delivery stop in one agent's interleaved schedule."""

    task_id: int
    kind: ScheduleEventKind
    location: Coordinate


@dataclass(slots=True)
class Agent:
    """A multi-load agent with a mutable route schedule."""

    agent_id: int
    location: Coordinate
    capacity: int
    current_load: int = 0
    assigned_task_ids: list[int] = field(default_factory=list)
    schedule: list[ScheduleEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.agent_id < 0:
            raise ValueError("agent_id must be non-negative")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 <= self.current_load <= self.capacity:
            raise ValueError("current_load must be between zero and capacity")

    def copy(self) -> "Agent":
        return Agent(
            agent_id=self.agent_id,
            location=self.location,
            capacity=self.capacity,
            current_load=self.current_load,
            assigned_task_ids=self.assigned_task_ids.copy(),
            schedule=self.schedule.copy(),
        )


@dataclass(slots=True)
class WorldState:
    """Discrete-time MLAPD state shared by every compared algorithm."""

    time: int
    tasks: dict[int, Task]
    agents: dict[int, Agent]
    task_status: dict[int, TaskStatus]
    task_agent: dict[int, int] = field(default_factory=dict)
    vertex_occupancy: dict[int, dict[Coordinate, int]] = field(default_factory=dict)
    edge_occupancy: dict[int, dict[tuple[Coordinate, Coordinate], int]] = field(
        default_factory=dict
    )

    @classmethod
    def from_trace(cls, trace: "ExperimentTrace") -> "WorldState":
        statuses = {
            task.task_id: (
                TaskStatus.PENDING if task.release_time == 0 else TaskStatus.UNRELEASED
            )
            for task in trace.tasks
        }
        return cls(
            time=0,
            tasks={task.task_id: task for task in trace.tasks},
            agents={agent.agent_id: agent.copy() for agent in trace.agents},
            task_status=statuses,
        )

    @property
    def pending_task_ids(self) -> tuple[int, ...]:
        return tuple(
            task_id
            for task_id, status in self.task_status.items()
            if status == TaskStatus.PENDING
        )

    def advance_to(self, new_time: int) -> None:
        """Advance time, release new requests, and expire unassigned requests."""
        if new_time < self.time:
            raise ValueError("time cannot move backwards")
        for tick in range(self.time + 1, new_time + 1):
            for task_id, task in self.tasks.items():
                if task.release_time == tick:
                    self.task_status[task_id] = TaskStatus.PENDING
                if self.task_status[task_id] == TaskStatus.PENDING and tick > task.deadline:
                    self.task_status[task_id] = TaskStatus.EXPIRED
        self.time = new_time

    def remaining_capacity(self, agent_id: int) -> int:
        agent = self._agent(agent_id)
        reserved_weight = sum(
            self.tasks[task_id].weight
            for task_id in agent.assigned_task_ids
            if self.task_status[task_id] in {TaskStatus.ASSIGNED, TaskStatus.PICKED_UP}
        )
        return agent.capacity - reserved_weight

    def assign(self, task_id: int, agent_id: int) -> None:
        """Assign a pending task and append pickup/delivery schedule events."""
        task = self._task(task_id)
        agent = self._agent(agent_id)
        if self.task_status[task_id] != TaskStatus.PENDING:
            raise ValueError(f"task {task_id} is not pending")
        if task.weight > self.remaining_capacity(agent_id):
            raise ValueError(f"agent {agent_id} lacks capacity for task {task_id}")

        self.task_status[task_id] = TaskStatus.ASSIGNED
        self.task_agent[task_id] = agent_id
        agent.assigned_task_ids.append(task_id)
        agent.schedule.extend(
            (
                ScheduleEvent(task_id, ScheduleEventKind.PICKUP, task.pickup),
                ScheduleEvent(task_id, ScheduleEventKind.DELIVERY, task.delivery),
            )
        )

    def mark_picked_up(self, task_id: int) -> None:
        task = self._task(task_id)
        agent = self._assigned_agent(task_id)
        if self.task_status[task_id] != TaskStatus.ASSIGNED:
            raise ValueError(f"task {task_id} is not awaiting pickup")
        if agent.current_load + task.weight > agent.capacity:
            raise ValueError(f"pickup of task {task_id} would exceed capacity")
        agent.current_load += task.weight
        self.task_status[task_id] = TaskStatus.PICKED_UP

    def mark_completed(self, task_id: int) -> None:
        task = self._task(task_id)
        agent = self._assigned_agent(task_id)
        if self.task_status[task_id] != TaskStatus.PICKED_UP:
            raise ValueError(f"task {task_id} has not been picked up")
        agent.current_load -= task.weight
        self.task_status[task_id] = TaskStatus.COMPLETED

    def reserve_path(self, agent_id: int, path: list[Coordinate], start_time: int) -> None:
        """Reserve path vertices and transitions, rejecting vertex and swap conflicts."""
        self._agent(agent_id)
        if not path:
            raise ValueError("a reserved path must contain at least one vertex")
        if start_time < self.time:
            raise ValueError("cannot reserve a path in the past")

        # Validate the whole path before recording it, so an invalid path never
        # leaves partial reservations behind.
        for offset, location in enumerate(path):
            timestamp = start_time + offset
            occupied = self.vertex_occupancy.get(timestamp, {})
            owner = occupied.get(location)
            if owner is not None and owner != agent_id:
                raise ValueError(f"vertex conflict at time {timestamp}: {location}")

            if offset == 0:
                continue
            previous = path[offset - 1]
            transitions = self.edge_occupancy.get(timestamp - 1, {})
            reverse_owner = transitions.get((location, previous))
            if reverse_owner is not None and reverse_owner != agent_id:
                raise ValueError(
                    f"edge-swap conflict at time {timestamp - 1}: {previous} <-> {location}"
                )

        for offset, location in enumerate(path):
            timestamp = start_time + offset
            self.vertex_occupancy.setdefault(timestamp, {})[location] = agent_id
            if offset == 0:
                continue
            previous = path[offset - 1]
            transitions = self.edge_occupancy.setdefault(timestamp - 1, {})
            transitions[(previous, location)] = agent_id

    def _task(self, task_id: int) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError as error:
            raise KeyError(f"unknown task {task_id}") from error

    def _agent(self, agent_id: int) -> Agent:
        try:
            return self.agents[agent_id]
        except KeyError as error:
            raise KeyError(f"unknown agent {agent_id}") from error

    def _assigned_agent(self, task_id: int) -> Agent:
        if task_id not in self.task_agent:
            raise ValueError(f"task {task_id} has no assigned agent")
        return self._agent(self.task_agent[task_id])
