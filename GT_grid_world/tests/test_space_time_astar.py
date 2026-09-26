"""Unit tests for time-indexed conflict-free routing."""

from __future__ import annotations

import unittest

from src.mlapd_replication.grid import StaticGrid
from src.mlapd_replication.models import Agent, WorldState
from src.mlapd_replication.space_time_astar import space_time_astar
from src.mlapd_replication.trace import ExperimentTrace


class SpaceTimeAStarTests(unittest.TestCase):
    def _world(self) -> WorldState:
        trace = ExperimentTrace(
            map_path="test-map",
            map_sha256="0" * 64,
            horizon=4,
            seed=1,
            agents=(
                Agent(agent_id=0, location=(0, 1), capacity=1),
                Agent(agent_id=1, location=(0, 0), capacity=1),
            ),
            tasks=(),
        )
        return WorldState.from_trace(trace)

    def test_waits_until_a_reserved_goal_becomes_free(self):
        grid = StaticGrid(frozenset({(0, 0), (0, 1), (0, 2)}))
        world = self._world()
        world.reserve_path(0, [(0, 1), (0, 2), (0, 2)], start_time=0)

        path = space_time_astar(
            grid,
            world,
            agent_id=1,
            start=(0, 0),
            goal=(0, 2),
            start_time=0,
            max_time=3,
        )

        self.assertEqual(path, [(0, 0), (0, 1), (0, 1), (0, 2)])

    def test_returns_safe_partial_progress_at_a_horizon_boundary(self):
        grid = StaticGrid(frozenset((0, column) for column in range(5)))
        world = self._world()

        path = space_time_astar(
            grid,
            world,
            agent_id=1,
            start=(0, 0),
            goal=(0, 4),
            start_time=0,
            max_time=2,
            allow_partial=True,
        )

        self.assertEqual(path, [(0, 0), (0, 1), (0, 2)])


if __name__ == "__main__":
    unittest.main()
