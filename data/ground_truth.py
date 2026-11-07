"""Ground-truth shortest paths and distance-to-go maps for generated grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from algorithms.dijkstra import dijkstra
from algorithms.movement import Coord, DEFAULT_COST_CONFIG, GroundTruthCostConfig, get_transition_cost, validate_map


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Ground-truth values for a start-goal map instance."""

    start: Coord
    goal: Coord
    distance_map: np.ndarray
    optimal_path: tuple[Coord, ...] | None
    optimal_path_cost: float
    reachable_mask: np.ndarray


def compute_ground_truth(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> GroundTruth:
    """Compute cost-to-go from every reachable cell to ``goal`` and the path.

    Dijkstra starts at the goal. The grid's legal transitions are undirected
    and have symmetric unit cost, so the resulting distance map is a distance-
    to-go map. Unreachable cells retain ``np.inf`` and are false in the mask.
    """

    array = validate_map(grid)
    for label, coordinate in (("start", start), ("goal", goal)):
        if len(coordinate) != 2:
            raise ValueError(f"{label} must be a (row, column) coordinate.")
        row, col = coordinate
        if (
            isinstance(row, (bool, np.bool_))
            or isinstance(col, (bool, np.bool_))
            or not isinstance(row, (int, np.integer))
            or not isinstance(col, (int, np.integer))
        ):
            raise ValueError(f"{label} coordinates must be integers.")
        if not (0 <= int(row) < array.shape[0] and 0 <= int(col) < array.shape[1]):
            raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
        if array[int(row), int(col)] != 0:
            raise ValueError(f"{label} cell must be traversable.")
    start_coord = (int(start[0]), int(start[1]))
    goal_coord = (int(goal[0]), int(goal[1]))

    result = dijkstra(array, goal_coord, config)
    cost = float(result.distances[start_coord])
    if not np.isfinite(cost):
        return GroundTruth(
            start=start_coord,
            goal=goal_coord,
            distance_map=result.distances,
            optimal_path=None,
            optimal_path_cost=inf,
            reachable_mask=result.reachable_mask,
        )

    path: list[Coord] = [start_coord]
    current = start_coord
    maximum_steps = array.size
    while current != goal_coord:
        parent_values = result.predecessors[current]
        if parent_values[0] < 0 or len(path) > maximum_steps:
            raise RuntimeError("Dijkstra predecessor map could not reconstruct the path.")
        following = (int(parent_values[0]), int(parent_values[1]))
        if not np.isfinite(get_transition_cost(array, current, following, config)):
            raise RuntimeError("Dijkstra produced an invalid transition in its path.")
        path.append(following)
        current = following

    reconstructed_cost = sum(
        get_transition_cost(array, source, destination, config)
        for source, destination in zip(path, path[1:])
    )
    if reconstructed_cost != cost:
        raise RuntimeError(
            f"Reconstructed path cost {reconstructed_cost} does not match Dijkstra cost {cost}."
        )

    return GroundTruth(
        start=start_coord,
        goal=goal_coord,
        distance_map=result.distances,
        optimal_path=tuple(path),
        optimal_path_cost=cost,
        reachable_mask=result.reachable_mask,
    )
