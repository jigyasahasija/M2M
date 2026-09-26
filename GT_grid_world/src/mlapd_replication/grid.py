"""A lightweight static grid used by the paper-replication components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Coordinate
from .trace import load_grid_map


@dataclass(frozen=True, slots=True)
class StaticGrid:
    """Immutable four-connected grid with a deterministic neighbour order.

    This intentionally models only static walls.  Robot conflicts belong to
    :mod:`space_time_astar`, which reasons about a separate time axis.
    """

    traversable: frozenset[Coordinate]

    @classmethod
    def from_legacy_map(cls, path: str | Path) -> "StaticGrid":
        traversable, _ = load_grid_map(path)
        return cls(frozenset(traversable))

    def get_if_obstacle(self, node: Coordinate) -> bool:
        return node not in self.traversable

    def get_neighbors(
        self, node: Coordinate, ignore_robots: bool = True
    ) -> list[Coordinate]:
        """Return N, E, S, W neighbours that are not static walls."""
        del ignore_robots  # The static grid deliberately has no robot state.
        row, column = node
        candidates = (
            (row - 1, column),
            (row, column + 1),
            (row + 1, column),
            (row, column - 1),
        )
        return [candidate for candidate in candidates if candidate in self.traversable]
