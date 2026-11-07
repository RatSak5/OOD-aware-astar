"""Conventional A* search on the shared eight-connected grid graph."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
from time import perf_counter
from typing import Callable

import numpy as np

from algorithms.movement import (
    Coord,
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    get_transition_cost,
    validate_map,
)

HeuristicFunction = Callable[[Coord, Coord], float]


def zero_heuristic(node: Coord, goal: Coord) -> float:
    """Return zero everywhere; A* then behaves like Dijkstra's algorithm."""

    del node, goal
    return 0.0


def chebyshev_heuristic(node: Coord, goal: Coord) -> float:
    """Return the admissible, consistent heuristic for unit-cost 8-way moves.

    The unobstructed minimum move count is ``max(|dr|, |dc|)``. Every legal
    transition changes this L-infinity distance by at most one and costs 1, so
    ``h(u) <= c(u,v) + h(v)`` on every legal edge. Obstacles can only increase
    the true cost-to-go. Thus this heuristic is both consistent and admissible
    for the project's unit-move ground-truth model, including its corner rule.
    """

    return float(max(abs(node[0] - goal[0]), abs(node[1] - goal[1])))


def euclidean_heuristic(node: Coord, goal: Coord) -> float:
    """Return Euclidean distance for the explicitly non-admissible baseline.

    Diagonal transitions cost one in this project, so Euclidean distance may
    overestimate the true remaining move count. This function is provided as
    an experimental comparison and carries no optimality guarantee.
    """

    dr = float(node[0] - goal[0])
    dc = float(node[1] - goal[1])
    return float(np.hypot(dr, dc))


@dataclass(frozen=True, slots=True)
class AStarResult:
    """Search outcome and performance counters for one start-goal query.

    ``expanded_nodes`` counts valid OPEN removals, including the goal when it
    is selected. This common convention is used by the Phase B result writer.
    ``generated_nodes`` counts unique traversable states first inserted into
    OPEN, including the start. ``reexpansions`` counts states expanded more
    than once after being reopened following a better path. OPEN and CLOSED
    sizes are unique state counts; stale heap entries do not contribute.
    """

    found: bool
    path: tuple[Coord, ...] | None
    found_ground_truth_cost: float
    expanded_nodes: int
    generated_nodes: int
    reexpansions: int
    max_open_size: int
    max_closed_size: int
    final_open_size: int
    closed_size: int
    search_time_seconds: float
    heuristic_name: str
    mean_open_size: float = 0.0


def _validate_endpoint(grid: np.ndarray, coordinate: Coord, label: str) -> Coord:
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
    normalized = (int(row), int(col))
    if not (0 <= normalized[0] < grid.shape[0] and 0 <= normalized[1] < grid.shape[1]):
        raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
    if grid[normalized] != 0:
        raise ValueError(f"{label} cell must be traversable.")
    return normalized


def _reconstruct_path(parents: dict[Coord, Coord], start: Coord, goal: Coord) -> tuple[Coord, ...]:
    reverse_path = [goal]
    current = goal
    while current != start:
        parent = parents.get(current)
        if parent is None:
            raise RuntimeError("A* parent map could not reconstruct a found path.")
        reverse_path.append(parent)
        if len(reverse_path) > len(parents) + 1:
            raise RuntimeError("A* parent map contains a cycle.")
        current = parent
    reverse_path.reverse()
    return tuple(reverse_path)


def astar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    heuristic: HeuristicFunction = chebyshev_heuristic,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    *,
    cost_tolerance: float = 1e-12,
) -> AStarResult:
    """Find a path using ``f(n) = g(n) + h(n)`` and a min-priority queue.

    ``heuristic(node, goal)`` is called once per discovered cell and cached.
    Pass ``zero_heuristic`` to obtain the Dijkstra-equivalent search. A future
    CNN can be exposed through this callable interface, but only admissible and
    consistent predictions carry the usual A* optimality guarantee. The
    implementation reopens a CLOSED state when it discovers a better ``g``.

    ``cost_tolerance`` is an absolute tolerance for float64 path comparisons.
    The implemented ground-truth model uses unit costs, but all accumulated
    costs and priorities remain floating-point values.
    """

    started_at = perf_counter()
    array = validate_map(grid)
    start_coord = _validate_endpoint(array, start, "start")
    goal_coord = _validate_endpoint(array, goal, "goal")
    if not isinstance(cost_config, GroundTruthCostConfig):
        raise TypeError("cost_config must be a GroundTruthCostConfig instance.")
    if not callable(heuristic):
        raise TypeError("heuristic must be a callable accepting (node, goal).")
    if not isfinite(cost_tolerance) or cost_tolerance < 0:
        raise ValueError("cost_tolerance must be finite and non-negative.")
    heuristic_name = getattr(heuristic, "__name__", heuristic.__class__.__name__)

    if start_coord == goal_coord:
        return AStarResult(
            found=True,
            path=(start_coord,),
            found_ground_truth_cost=0.0,
            expanded_nodes=0,
            generated_nodes=1,
            reexpansions=0,
            max_open_size=1,
            max_closed_size=0,
            final_open_size=0,
            closed_size=0,
            search_time_seconds=perf_counter() - started_at,
            heuristic_name=heuristic_name,
            mean_open_size=1.0,
        )

    heuristic_cache: dict[Coord, float] = {}

    def h_value(node: Coord) -> float:
        if node not in heuristic_cache:
            value = float(heuristic(node, goal_coord))
            if not isfinite(value):
                raise ValueError(f"Heuristic returned a non-finite value for {node}.")
            heuristic_cache[node] = value
        return heuristic_cache[node]

    g_score: dict[Coord, float] = {start_coord: 0.0}
    parents: dict[Coord, Coord] = {}
    discovered: set[Coord] = {start_coord}
    open_states: set[Coord] = {start_coord}
    closed_states: set[Coord] = set()
    expansion_counts: dict[Coord, int] = {}
    sequence = count()
    frontier: list[tuple[float, int, float, Coord]] = [
        (h_value(start_coord), next(sequence), 0.0, start_coord)
    ]
    expanded_nodes = 0
    reexpansions = 0
    max_open_size = 1
    max_closed_size = 0
    open_size_total = 0
    open_iterations = 0
    found_path: tuple[Coord, ...] | None = None
    found_cost = inf

    while frontier:
        _, _, queued_g, current = heappop(frontier)
        best_g = g_score.get(current, inf)
        if queued_g > best_g + cost_tolerance or current not in open_states:
            continue

        open_size_total += len(open_states)
        open_iterations += 1
        open_states.remove(current)
        expanded_nodes += 1
        if current == goal_coord:
            found_path = _reconstruct_path(parents, start_coord, goal_coord)
            found_cost = float(best_g)
            measured_cost = sum(
                get_transition_cost(array, source, destination, cost_config)
                for source, destination in zip(found_path, found_path[1:])
            )
            if abs(measured_cost - found_cost) > cost_tolerance:
                raise RuntimeError("A* path cost does not match its accumulated ground-truth cost.")
            break

        if expansion_counts.get(current, 0) > 0:
            reexpansions += 1
        expansion_counts[current] = expansion_counts.get(current, 0) + 1
        closed_states.add(current)
        max_closed_size = max(max_closed_size, len(closed_states))

        for neighbor in get_neighbors(array, current, cost_config):
            step_cost = get_transition_cost(array, current, neighbor, cost_config)
            candidate_g = best_g + step_cost
            previous_g = g_score.get(neighbor, inf)
            if not isfinite(previous_g) or candidate_g < previous_g - cost_tolerance:
                g_score[neighbor] = candidate_g
                parents[neighbor] = current
                if neighbor in closed_states:
                    closed_states.remove(neighbor)
                if neighbor not in discovered:
                    discovered.add(neighbor)
                open_states.add(neighbor)
                priority = candidate_g + h_value(neighbor)
                heappush(frontier, (priority, next(sequence), candidate_g, neighbor))
                max_open_size = max(max_open_size, len(open_states))

    return AStarResult(
        found=found_path is not None,
        path=found_path,
        found_ground_truth_cost=float(found_cost),
        expanded_nodes=expanded_nodes,
        generated_nodes=len(discovered),
        reexpansions=reexpansions,
        max_open_size=max_open_size,
        max_closed_size=max_closed_size,
        final_open_size=len(open_states),
        closed_size=len(closed_states),
        search_time_seconds=perf_counter() - started_at,
        heuristic_name=heuristic_name,
        mean_open_size=(open_size_total / open_iterations if open_iterations else 0.0),
    )
