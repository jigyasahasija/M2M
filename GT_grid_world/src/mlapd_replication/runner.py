"""Batch experiment runner for the reproducible MLAPD baselines.

This is an intentionally small, inspectable benchmark harness.  It runs one
frozen trace at a time and records exactly which baseline made each decision.
It is not labelled as a paper-result reproduction until the paper's original
maps, traces, and tie-breaking details are available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

from .baselines import AssignmentDecision, assign_rmca, assign_tpmc
from .grid import StaticGrid
from .models import ScheduleEventKind, TaskStatus, WorldState
from .space_time_astar import RoutingResult, plan_prioritized_paths
from .trace import ExperimentTrace, load_trace


BaselineName = Literal["tpmc", "rmca"]
Allocator = Callable[[WorldState, StaticGrid], tuple[AssignmentDecision, ...]]


@dataclass(frozen=True, slots=True)
class BatchRecord:
    """The allocation and routing work done at one batch boundary."""

    time: int
    assignments: tuple[AssignmentDecision, ...]
    allocation_seconds: float
    path_planning_seconds: float
    path_planning_succeeded: bool


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Metrics and decision log produced by one baseline on one frozen trace."""

    baseline: BaselineName
    trace_seed: int
    horizon: int
    batch_size: int
    completed_task_ids: tuple[int, ...]
    expired_task_ids: tuple[int, ...]
    completion_rate: float
    travel_distance: int
    delay_cost: int
    total_cost: float
    average_batch_path_planning_seconds: float
    failed_path_planning_batches: int
    batches: tuple[BatchRecord, ...]
    final_task_statuses: dict[int, TaskStatus]

    def to_dict(self) -> dict:
        """Return a stable, JSON-ready representation for review and plotting."""
        payload = asdict(self)
        payload["final_task_statuses"] = {
            str(task_id): status.value
            for task_id, status in self.final_task_statuses.items()
        }
        return payload


def run_baseline(
    trace: ExperimentTrace,
    *,
    baseline: BaselineName,
    batch_size: int,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> ExperimentResult:
    """Run TPMC or RMCA on one trace using sequential space-time routing.

    At every batch boundary, tasks released so far are allocated, all agent
    routes are planned in a fixed agent-ID priority order, then those plans are
    followed until the next batch.  The next batch replans from the true agent
    locations.  This is the paper-baseline behaviour to compare against a
    later CBDC implementation, not the paper's dynamic-CBS implementation.

    ``delay_cost`` sums ``max(0, completion_time - deadline)`` only for tasks
    that finish inside the finite horizon.  Unfinished and expired tasks remain
    visible through the completion rate and final statuses instead of receiving
    an invented penalty.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if alpha < 0 or beta < 0:
        raise ValueError("cost weights must be non-negative")

    allocator = _allocator_for(baseline)
    grid = StaticGrid.from_legacy_map(trace.map_path)
    world = WorldState.from_trace(trace)
    completions: dict[int, int] = {}
    travel_distance = 0
    records: list[BatchRecord] = []

    for batch_start in range(0, trace.horizon, batch_size):
        # Previous loop iterations finish exactly at this time.  Keep this
        # guard for clarity if the execution policy is changed later.
        world.advance_to(batch_start)
        allocation_start = perf_counter()
        decisions = allocator(world, grid)
        allocation_seconds = perf_counter() - allocation_start
        _process_arrivals(world, completions, batch_start)

        planning_start = perf_counter()
        routing = plan_prioritized_paths(
            grid,
            world,
            start_time=batch_start,
            end_time=trace.horizon,
        )
        planning_seconds = perf_counter() - planning_start
        records.append(
            BatchRecord(
                time=batch_start,
                assignments=decisions,
                allocation_seconds=allocation_seconds,
                path_planning_seconds=planning_seconds,
                path_planning_succeeded=routing is not None,
            )
        )

        batch_end = min(batch_start + batch_size, trace.horizon)
        for time in range(batch_start + 1, batch_end + 1):
            if routing is not None:
                travel_distance += _advance_agents_one_tick(world, routing, time)
            world.advance_to(time)
            _process_arrivals(world, completions, time)

    completed_task_ids = tuple(sorted(completions))
    expired_task_ids = tuple(
        sorted(
            task_id
            for task_id, status in world.task_status.items()
            if status == TaskStatus.EXPIRED
        )
    )
    delay_cost = sum(
        max(0, completion_time - world.tasks[task_id].deadline)
        for task_id, completion_time in completions.items()
    )
    planning_times = [record.path_planning_seconds for record in records]
    completion_rate = len(completed_task_ids) / len(trace.tasks) if trace.tasks else 1.0

    return ExperimentResult(
        baseline=baseline,
        trace_seed=trace.seed,
        horizon=trace.horizon,
        batch_size=batch_size,
        completed_task_ids=completed_task_ids,
        expired_task_ids=expired_task_ids,
        completion_rate=completion_rate,
        travel_distance=travel_distance,
        delay_cost=delay_cost,
        total_cost=alpha * travel_distance + beta * delay_cost,
        average_batch_path_planning_seconds=(
            sum(planning_times) / len(planning_times) if planning_times else 0.0
        ),
        failed_path_planning_batches=sum(
            not record.path_planning_succeeded for record in records
        ),
        batches=tuple(records),
        final_task_statuses=world.task_status.copy(),
    )


def save_result(result: ExperimentResult, destination: str | Path) -> None:
    """Write one baseline result without changing the frozen input trace."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _allocator_for(baseline: BaselineName) -> Allocator:
    if baseline == "tpmc":
        return assign_tpmc
    if baseline == "rmca":
        return assign_rmca
    raise ValueError(f"unknown baseline: {baseline}")


def _advance_agents_one_tick(
    world: WorldState, routing: RoutingResult, time: int
) -> int:
    moves = 0
    for agent_id, agent in world.agents.items():
        destination = routing.paths[agent_id].location_at(time)
        moves += destination != agent.location
        agent.location = destination
    return moves


def _process_arrivals(
    world: WorldState, completions: dict[int, int], time: int
) -> None:
    """Apply every pickup or delivery event reached at the current time."""
    for agent in world.agents.values():
        while agent.schedule and agent.schedule[0].location == agent.location:
            event = agent.schedule[0]
            if event.kind == ScheduleEventKind.PICKUP:
                world.mark_picked_up(event.task_id)
            else:
                world.mark_completed(event.task_id)
                completions[event.task_id] = time
            agent.schedule.pop(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen MLAPD baseline trace")
    parser.add_argument("--trace", required=True, help="Frozen trace JSON")
    parser.add_argument("--baseline", required=True, choices=("tpmc", "rmca"))
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--output", required=True, help="Result JSON destination")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    arguments = parser.parse_args()

    result = run_baseline(
        load_trace(arguments.trace),
        baseline=arguments.baseline,
        batch_size=arguments.batch_size,
        alpha=arguments.alpha,
        beta=arguments.beta,
    )
    save_result(result, arguments.output)


if __name__ == "__main__":
    main()
