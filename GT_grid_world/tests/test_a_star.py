"""Unit tests for the replication layer's static-grid A* primitive."""

from __future__ import annotations

import unittest

from src.path_finding_algorithms.a_star import astar_path


class TinyGrid:
    """Dependency-free 4-connected grid used to test A* behavior."""

    def __init__(self, height, width, *, obstacles=(), occupied=()):
        self.height = height
        self.width = width
        self.obstacles = set(obstacles)
        self.occupied = set(occupied)

    def get_if_obstacle(self, node):
        return node in self.obstacles

    def get_neighbors(self, node, ignore_robots=False):
        candidates = (
            (node[0] - 1, node[1]),
            (node[0], node[1] + 1),
            (node[0] + 1, node[1]),
            (node[0], node[1] - 1),
        )
        return [
            candidate
            for candidate in candidates
            if 0 <= candidate[0] < self.height
            and 0 <= candidate[1] < self.width
            and candidate not in self.obstacles
            and (ignore_robots or candidate not in self.occupied)
        ]


class AStarPathTests(unittest.TestCase):
    def test_finds_a_shortest_detour_with_stable_tie_breaking(self):
        graph = TinyGrid(3, 3, obstacles={(1, 1)})

        path = astar_path(graph, (0, 0), (2, 2))

        self.assertEqual(path, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
        self.assertEqual(len(path) - 1, 4)

    def test_returns_a_singleton_path_when_already_at_goal(self):
        graph = TinyGrid(2, 2)

        self.assertEqual(astar_path(graph, (1, 1), (1, 1)), [(1, 1)])

    def test_returns_none_for_blocked_or_unreachable_endpoints(self):
        blocked_start = TinyGrid(2, 2, obstacles={(0, 0)})
        enclosed_goal = TinyGrid(3, 3, obstacles={(0, 1), (1, 0), (1, 2), (2, 1)})

        self.assertIsNone(astar_path(blocked_start, (0, 0), (1, 1)))
        self.assertIsNone(astar_path(enclosed_goal, (0, 0), (1, 1)))

    def test_can_either_respect_or_ignore_current_robot_positions(self):
        graph = TinyGrid(1, 3, occupied={(0, 1)})

        self.assertIsNone(astar_path(graph, (0, 0), (0, 2), ignore_robots=False))
        self.assertEqual(
            astar_path(graph, (0, 0), (0, 2), ignore_robots=True),
            [(0, 0), (0, 1), (0, 2)],
        )


if __name__ == "__main__":
    unittest.main()
