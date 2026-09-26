"""Deterministic, serializable experiment inputs for MLAPD comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from .models import Agent, Coordinate, Task


@dataclass(frozen=True, slots=True)
class ExperimentTrace:
    """All immutable inputs that must be shared by compared algorithms."""

    map_path: str
    map_sha256: str
    horizon: int
    seed: int
    agents: tuple[Agent, ...]
    tasks: tuple[Task, ...]
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len({agent.agent_id for agent in self.agents}) != len(self.agents):
            raise ValueError("agent IDs must be unique")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task IDs must be unique")

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "map_path": self.map_path,
            "map_sha256": self.map_sha256,
            "horizon": self.horizon,
            "seed": self.seed,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "location": list(agent.location),
                    "capacity": agent.capacity,
                }
                for agent in self.agents
            ],
            "tasks": [
                {
                    "task_id": task.task_id,
                    "release_time": task.release_time,
                    "deadline": task.deadline,
                    "pickup": list(task.pickup),
                    "delivery": list(task.delivery),
                    "weight": task.weight,
                }
                for task in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExperimentTrace":
        if payload.get("format_version") != 1:
            raise ValueError("unsupported experiment trace format")
        return cls(
            map_path=payload["map_path"],
            map_sha256=payload["map_sha256"],
            horizon=payload["horizon"],
            seed=payload["seed"],
            agents=tuple(
                Agent(
                    agent_id=agent["agent_id"],
                    location=tuple(agent["location"]),
                    capacity=agent["capacity"],
                )
                for agent in payload["agents"]
            ),
            tasks=tuple(
                Task(
                    task_id=task["task_id"],
                    release_time=task["release_time"],
                    deadline=task["deadline"],
                    pickup=tuple(task["pickup"]),
                    delivery=tuple(task["delivery"]),
                    weight=task["weight"],
                )
                for task in payload["tasks"]
            ),
        )


def load_grid_map(path: str | Path) -> tuple[tuple[Coordinate, ...], tuple[Coordinate, ...]]:
    """Read traversable cells and designated robot starts from a legacy map."""
    map_path = Path(path)
    rows = map_path.read_text(encoding="utf-8").splitlines()
    if len(rows) < 3:
        raise ValueError(f"map {map_path} has no grid data")
    height, width, *_ = (int(value.strip()) for value in rows[0].split(","))
    grid = [row.rstrip() for row in rows[2:] if row.strip()]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"map {map_path} does not match its declared dimensions")

    traversable: list[Coordinate] = []
    starts: list[Coordinate] = []
    for row_index, row in enumerate(grid):
        for column_index, cell in enumerate(row):
            if cell != "@":
                traversable.append((row_index, column_index))
            if cell == "r":
                starts.append((row_index, column_index))
    return tuple(traversable), tuple(starts)


def generate_experiment_trace(
    *,
    map_path: str | Path,
    horizon: int,
    seed: int,
    task_count: int,
    agent_count: int,
    capacity: int,
    deadline_slack: int,
) -> ExperimentTrace:
    """Generate one deterministic unit-weight trace and freeze its inputs."""
    if task_count < 0:
        raise ValueError("task_count must be non-negative")
    if deadline_slack < 0:
        raise ValueError("deadline_slack must be non-negative")

    resolved_map_path = Path(map_path)
    traversable, starts = load_grid_map(resolved_map_path)
    if agent_count > len(starts):
        raise ValueError(
            f"map has {len(starts)} designated starts, fewer than {agent_count} agents"
        )
    if len(traversable) < 2 and task_count:
        raise ValueError("a task trace needs at least two traversable cells")

    random_source = random.Random(seed)
    selected_starts = random_source.sample(starts, agent_count)
    agents = tuple(
        Agent(agent_id=index, location=location, capacity=capacity)
        for index, location in enumerate(selected_starts)
    )

    tasks = []
    for task_id in range(task_count):
        pickup, delivery = random_source.sample(traversable, 2)
        release_time = random_source.randrange(horizon)
        tasks.append(
            Task(
                task_id=task_id,
                release_time=release_time,
                deadline=release_time + deadline_slack,
                pickup=pickup,
                delivery=delivery,
            )
        )
    tasks.sort(key=lambda task: (task.release_time, task.task_id))

    return ExperimentTrace(
        map_path=str(resolved_map_path),
        map_sha256=hashlib.sha256(resolved_map_path.read_bytes()).hexdigest(),
        horizon=horizon,
        seed=seed,
        agents=agents,
        tasks=tuple(tasks),
    )


def save_trace(trace: ExperimentTrace, destination: str | Path) -> None:
    """Write an immutable trace in a stable, reviewable JSON representation."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_trace(path: str | Path) -> ExperimentTrace:
    """Load a previously generated trace without re-sampling any input."""
    return ExperimentTrace.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen MLAPD experiment trace")
    parser.add_argument("--map", required=True, help="Path to a legacy grid map")
    parser.add_argument("--output", required=True, help="Destination JSON trace")
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--task-count", required=True, type=int)
    parser.add_argument("--agent-count", required=True, type=int)
    parser.add_argument("--capacity", required=True, type=int)
    parser.add_argument("--deadline-slack", required=True, type=int)
    arguments = parser.parse_args()
    trace = generate_experiment_trace(
        map_path=arguments.map,
        horizon=arguments.horizon,
        seed=arguments.seed,
        task_count=arguments.task_count,
        agent_count=arguments.agent_count,
        capacity=arguments.capacity,
        deadline_slack=arguments.deadline_slack,
    )
    save_trace(trace, arguments.output)


if __name__ == "__main__":
    main()
