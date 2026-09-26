"""Tests for the two paper baselines and the batch experiment runner."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlapd_replication.baselines import assign_rmca, assign_tpmc
from src.mlapd_replication.grid import StaticGrid
from src.mlapd_replication.models import Agent, Task, TaskStatus, WorldState
from src.mlapd_replication.runner import run_baseline
from src.mlapd_replication.trace import ExperimentTrace


class ReplicationBaselineTests(unittest.TestCase):
    def _allocation_world(self) -> WorldState:
        trace = ExperimentTrace(
            map_path="test-map",
            map_sha256="0" * 64,
            horizon=20,
            seed=2,
            agents=(
                Agent(agent_id=0, location=(0, 0), capacity=2),
                Agent(agent_id=1, location=(0, 3), capacity=2),
            ),
            tasks=(
                Task(0, release_time=0, deadline=19, pickup=(0, 10), delivery=(0, 10)),
                Task(1, release_time=0, deadline=19, pickup=(0, 1), delivery=(0, 2)),
            ),
        )
        world = WorldState.from_trace(trace)
        world.assign(0, 0)
        return world

    def test_tpmc_uses_current_location_but_rmca_accounts_for_schedule_tail(self):
        grid = StaticGrid(frozenset((0, column) for column in range(11)))

        tpmc_world = self._allocation_world()
        tpmc = assign_tpmc(tpmc_world, grid)
        self.assertEqual(len(tpmc), 1)
        self.assertEqual(tpmc[0].task_id, 1)
        self.assertEqual(tpmc[0].agent_id, 0)

        rmca_world = self._allocation_world()
        rmca = assign_rmca(rmca_world, grid)
        self.assertEqual(len(rmca), 1)
        self.assertEqual(rmca[0].task_id, 1)
        self.assertEqual(rmca[0].agent_id, 1)

    def test_runner_completes_a_small_frozen_trace(self):
        with TemporaryDirectory() as directory:
            map_path = Path(directory) / "tiny.map"
            map_path.write_text("2,3,0,0,2\n\nr.r\n...\n", encoding="utf-8")
            trace = ExperimentTrace(
                map_path=str(map_path),
                map_sha256="0" * 64,
                horizon=6,
                seed=3,
                agents=(
                    Agent(agent_id=0, location=(0, 0), capacity=1),
                    Agent(agent_id=1, location=(0, 2), capacity=1),
                ),
                tasks=(
                    Task(
                        0,
                        release_time=0,
                        deadline=4,
                        pickup=(0, 1),
                        delivery=(1, 1),
                    ),
                ),
            )

            result = run_baseline(trace, baseline="tpmc", batch_size=3)

        self.assertEqual(result.completed_task_ids, (0,))
        self.assertEqual(result.completion_rate, 1.0)
        self.assertEqual(result.travel_distance, 2)
        self.assertEqual(result.delay_cost, 0)
        self.assertEqual(result.failed_path_planning_batches, 0)
        self.assertEqual(result.final_task_statuses[0], TaskStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
