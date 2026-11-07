"""Dijkstra shortest paths using the project's shared grid movement rules."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from time import perf_counter

import numpy as np

from algorithms.movement import (
    Coord,
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    get_transition_cost,
    validate_map,
)


@dataclass(frozen=True, slots=True)
class DijkstraResult:
    """Distance and predecessor maps plus observational search counters.

    Expanded nodes are non-stale states removed from OPEN; the source and all
    reachable nodes, including a supplied target if one is later selected by a
    caller, follow the same pop-based count convention used by Phase B.
    """

    source: Coord
    distances: np.ndarray
    predecessors: np.ndarray
    reachable_mask: np.ndarray
    expanded_nodes: int = 0
    generated_nodes: int = 0
    reexpansions: int = 0
    max_open_size: int = 0
    max_closed_size: int = 0
    final_open_size: int = 0
    closed_size: int = 0
    search_time_seconds: float = 0.0


def dijkstra(
    grid: np.ndarray,
    source: Coord,
    config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    *,
    target: Coord | None = None,
) -> DijkstraResult:
    """Compute shortest distances from ``source`` to every reachable cell.

    Distances are stored as float64; unreachable cells remain ``np.inf``.
    ``predecessors[row, col]`` stores the preceding coordinate on a shortest
    source-to-cell route, or ``(-1, -1)`` when the cell has no predecessor.
    """

    started_at = perf_counter()
    array = validate_map(grid)
    if len(source) != 2:
        raise ValueError("source must be a (row, column) coordinate.")
    row, col = source
    if (
        isinstance(row, (bool, np.bool_))
        or isinstance(col, (bool, np.bool_))
        or not isinstance(row, (int, np.integer))
        or not isinstance(col, (int, np.integer))
    ):
        raise ValueError("source coordinates must be integers.")
    source_coord = (int(row), int(col))
    if not (0 <= source_coord[0] < array.shape[0] and 0 <= source_coord[1] < array.shape[1]):
        raise ValueError(f"Source coordinate {source!r} is outside the map.")
    if array[source_coord] != 0:
        raise ValueError("Dijkstra source must be traversable.")
    target_coord: Coord | None = None
    if target is not None:
        if len(target) != 2:
            raise ValueError("Target must be a (row, column) coordinate.")
        target_row, target_col = target
        if (
            isinstance(target_row, (bool, np.bool_))
            or isinstance(target_col, (bool, np.bool_))
            or not isinstance(target_row, (int, np.integer))
            or not isinstance(target_col, (int, np.integer))
        ):
            raise ValueError("Target coordinates must be integers.")
        target_coord = (int(target_row), int(target_col))
        if not (0 <= target_coord[0] < array.shape[0] and 0 <= target_coord[1] < array.shape[1]):
            raise ValueError(f"Target coordinate {target!r} is outside the map.")
        if array[target_coord] != 0:
            raise ValueError("Dijkstra target must be traversable.")
    if not isinstance(config, GroundTruthCostConfig):
        raise TypeError("config must be a GroundTruthCostConfig instance.")

    distances = np.full(array.shape, inf, dtype=np.float64)
    predecessors = np.full((*array.shape, 2), -1, dtype=np.int32)
    distances[source_coord] = 0.0
    frontier: list[tuple[float, int, int]] = [(0.0, *source_coord)]
    open_states = {source_coord}
    discovered = {source_coord}
    closed_states: set[Coord] = set()
    expansion_counts: dict[Coord, int] = {}
    expanded_nodes = 0
    reexpansions = 0
    max_open_size = 1
    max_closed_size = 0

    while frontier:
        current_distance, row, col = heappop(frontier)
        current = (row, col)
        if current_distance != distances[current]:
            continue
        open_states.discard(current)
        expanded_nodes += 1
        if expansion_counts.get(current, 0) > 0:
            reexpansions += 1
        expansion_counts[current] = expansion_counts.get(current, 0) + 1
        closed_states.add(current)
        max_closed_size = max(max_closed_size, len(closed_states))
        if current == target_coord:
            break
        for neighbor in get_neighbors(array, current, config):
            step_cost = get_transition_cost(array, current, neighbor, config)
            candidate = current_distance + step_cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = current
                discovered.add(neighbor)
                open_states.add(neighbor)
                max_open_size = max(max_open_size, len(open_states))
                heappush(frontier, (candidate, *neighbor))

    reachable = np.isfinite(distances)
    return DijkstraResult(
        source=source_coord,
        distances=distances,
        predecessors=predecessors,
        reachable_mask=reachable,
        expanded_nodes=expanded_nodes,
        generated_nodes=len(discovered),
        reexpansions=reexpansions,
        max_open_size=max_open_size,
        max_closed_size=max_closed_size,
        final_open_size=len(open_states),
        closed_size=len(closed_states),
        search_time_seconds=float(perf_counter() - started_at),
    )
