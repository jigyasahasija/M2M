"""Tests for the MLAPD replication state and trace boundary."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlapd_replication.models import Agent, Task, TaskStatus, WorldState
from src.mlapd_replication.trace import (
    ExperimentTrace,
    generate_experiment_trace,
    load_trace,
    save_trace,
)


class ReplicationModelTests(unittest.TestCase):
    def _trace(self) -> ExperimentTrace:
        return ExperimentTrace(
            map_path="test-map",
            map_sha256="0" * 64,
            horizon=10,
            seed=7,
            agents=(Agent(agent_id=0, location=(0, 0), capacity=1),),
            tasks=(
                Task(0, release_time=0, deadline=1, pickup=(0, 1), delivery=(0, 2)),
                Task(1, release_time=2, deadline=4, pickup=(1, 0), delivery=(1, 1)),
            ),
        )

    def test_release_expiry_and_capacity_follow_the_task_lifecycle(self):
        world = WorldState.from_trace(self._trace())
        self.assertEqual(world.pending_task_ids, (0,))

        world.assign(0, 0)
        self.assertEqual(world.remaining_capacity(0), 0)
        with self.assertRaises(ValueError):
            world.assign(1, 0)

        world.mark_picked_up(0)
        self.assertEqual(world.agents[0].current_load, 1)
        world.mark_completed(0)
        self.assertEqual(world.agents[0].current_load, 0)

        world.advance_to(2)
        self.assertEqual(world.task_status[0], TaskStatus.COMPLETED)
        self.assertEqual(world.task_status[1], TaskStatus.PENDING)

    def test_pending_tasks_expire_after_their_deadline(self):
        world = WorldState.from_trace(self._trace())

        world.advance_to(2)

        self.assertEqual(world.task_status[0], TaskStatus.EXPIRED)
        self.assertEqual(world.task_status[1], TaskStatus.PENDING)

    def test_vertex_and_edge_swap_conflicts_are_rejected(self):
        trace = ExperimentTrace(
            map_path="test-map",
            map_sha256="0" * 64,
            horizon=3,
            seed=1,
            agents=(
                Agent(agent_id=0, location=(0, 0), capacity=1),
                Agent(agent_id=1, location=(0, 1), capacity=1),
            ),
            tasks=(),
        )
        world = WorldState.from_trace(trace)
        world.reserve_path(0, [(0, 0), (0, 1)], start_time=0)

        with self.assertRaisesRegex(ValueError, "vertex conflict"):
            world.reserve_path(1, [(1, 1), (0, 1)], start_time=0)
        self.assertNotIn((1, 1), world.vertex_occupancy[0])
        with self.assertRaisesRegex(ValueError, "edge-swap conflict"):
            world.reserve_path(1, [(0, 1), (0, 0)], start_time=0)

    def test_saved_trace_reloads_exactly_and_generation_is_seeded(self):
        map_file = Path(__file__).parents[2] / "data/maps/small_test"
        first = generate_experiment_trace(
            map_path=map_file,
            horizon=30,
            seed=9,
            task_count=4,
            agent_count=2,
            capacity=2,
            deadline_slack=8,
        )
        second = generate_experiment_trace(
            map_path=map_file,
            horizon=30,
            seed=9,
            task_count=4,
            agent_count=2,
            capacity=2,
            deadline_slack=8,
        )
        self.assertEqual(first, second)

        with TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            save_trace(first, trace_path)
            self.assertEqual(load_trace(trace_path), first)


if __name__ == "__main__":
    unittest.main()
