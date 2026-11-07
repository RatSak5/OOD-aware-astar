"""Shared movement rules and ground-truth transition costs for grid worlds."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import TypeAlias

import numpy as np

Coord: TypeAlias = tuple[int, int]
GridArray: TypeAlias = np.ndarray

# Stable ordering makes equal-cost Dijkstra paths reproducible.
_ORTHOGONAL_MOVES: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_MOVES: tuple[Coord, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBOR_MOVES: tuple[Coord, ...] = _ORTHOGONAL_MOVES + _DIAGONAL_MOVES


@dataclass(frozen=True, slots=True)
class GroundTruthCostConfig:
    """Ground-truth path-cost convention used by every search baseline.

    ``unit_move_count`` reproduces the supplied Neural A* benchmark convention:
    entering any passable neighbor costs one, including a diagonal neighbor.
    Diagonal corner-cutting is disallowed as a project-specific graph rule.
    """

    model: str = "unit_move_count"
    prevent_diagonal_corner_cutting: bool = True

    def __post_init__(self) -> None:
        if self.model != "unit_move_count":
            raise ValueError(
                "Only the documented 'unit_move_count' ground-truth model is supported."
            )
        if not self.prevent_diagonal_corner_cutting:
            raise ValueError("Diagonal corner-cutting must remain disabled in this project.")


DEFAULT_COST_CONFIG = GroundTruthCostConfig()


def validate_map(grid: GridArray) -> GridArray:
    """Return a NumPy view after validating the 2-D 0=free, 1=obstacle map."""

    array = np.asarray(grid)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("A grid map must be a non-empty two-dimensional array.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Grid values must be 0 (traversable) or 1 (obstacle).")
    return array


def _coord(value: Coord, shape: tuple[int, int]) -> Coord | None:
    """Normalize an integral coordinate, returning None when out of bounds."""

    if len(value) != 2:
        return None
    row, col = value
    if isinstance(row, (bool, np.bool_)) or isinstance(col, (bool, np.bool_)):
        return None
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        return None
    row_i, col_i = int(row), int(col)
    if not (0 <= row_i < shape[0] and 0 <= col_i < shape[1]):
        return None
    return row_i, col_i


def is_legal_transition(
    grid: GridArray,
    source: Coord,
    destination: Coord,
    config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> bool:
    """Return whether a single eight-connected transition is legal.

    A diagonal is legal only if both orthogonal side cells are traversable.
    This prevents the moving point from passing through a blocked corner.
    """

    array = np.asarray(grid)
    if array.ndim != 2 or array.size == 0:
        return False
    if not isinstance(config, GroundTruthCostConfig):
        raise TypeError("config must be a GroundTruthCostConfig instance.")
    src = _coord(source, array.shape)
    dst = _coord(destination, array.shape)
    if src is None or dst is None or src == dst:
        return False
    if array[src] != 0 or array[dst] != 0:
        return False
    if array[src] not in (0, 1) or array[dst] not in (0, 1):
        return False

    dr, dc = dst[0] - src[0], dst[1] - src[1]
    if abs(dr) > 1 or abs(dc) > 1:
        return False
    if dr != 0 and dc != 0:
        side_a = (src[0] + dr, src[1])
        side_b = (src[0], src[1] + dc)
        if array[side_a] != 0 or array[side_b] != 0:
            return False
    return True


def get_neighbors(
    grid: GridArray,
    source: Coord,
    config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> list[Coord]:
    """Return all legal traversable eight-connected neighbors of ``source``.

    Neighbors are ordered orthogonal-first, then diagonal, in a fixed order.
    An out-of-bounds source raises ``ValueError``; an obstacle source has no
    neighbors. All planning algorithms should use this shared function.
    """

    array = np.asarray(grid)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("A grid map must be a non-empty two-dimensional array.")
    src = _coord(source, array.shape)
    if src is None:
        raise ValueError(f"Source coordinate {source!r} is outside the map.")
    if not isinstance(config, GroundTruthCostConfig):
        raise TypeError("config must be a GroundTruthCostConfig instance.")
    if array[src] != 0:
        return []

    result: list[Coord] = []
    for dr, dc in NEIGHBOR_MOVES:
        destination = (src[0] + dr, src[1] + dc)
        # The caller's map is fully validated at its data boundary. Checking
        # only adjacent cells here avoids rescanning the whole map at each A* /
        # Dijkstra expansion.
        dst = _coord(destination, array.shape)
        if dst is None or array[dst] not in (0, 1) or array[dst] != 0:
            continue
        if dr != 0 and dc != 0:
            side_a = (src[0] + dr, src[1])
            side_b = (src[0], src[1] + dc)
            if (
                array[side_a] not in (0, 1)
                or array[side_b] not in (0, 1)
                or array[side_a] != 0
                or array[side_b] != 0
            ):
                continue
        if abs(dr) <= 1 and abs(dc) <= 1:
            result.append(destination)
    return result


def get_transition_cost(
    map: GridArray,
    source: Coord,
    destination: Coord,
    config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> float:
    """Return the shared ground-truth cost of a legal transition.

    The benchmark convention is one unit per entered cell for every legal
    orthogonal or diagonal move. Obstacles, corner-cutting diagonals, non-adjacent
    moves, and invalid coordinates have infinite cost. This is distinct from the
    Neural A* learned guidance value assigned to each cell.
    """

    if not isinstance(config, GroundTruthCostConfig):
        raise TypeError("config must be a GroundTruthCostConfig instance.")
    if not is_legal_transition(map, source, destination, config):
        return inf
    return 1.0
