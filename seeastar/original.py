"""Faithful SeeA* search with the paper's three candidate strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from time import perf_counter
from typing import Callable

import numpy as np

from algorithms.astar import HeuristicFunction, chebyshev_heuristic
from algorithms.movement import (
    Coord,
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    get_transition_cost,
    validate_map,
)
from seeastar.sampling import (
    CandidateSampler,
    CompetitiveClusterSampler,
    UCTLikeSampler,
    UniformSampler,
    lowest_f_candidate,
)


@dataclass(frozen=True, slots=True)
class SeeAStarConfig:
    """Fixed per-search SeeA* settings; K does not adapt during a search."""

    k: int = 50
    sampling_strategy: str = "uniform"
    max_expansions: int | None = None
    num_clusters: int = 5
    eta: float = 0.15
    cb: float = 0.35
    seed: int = 0
    cost_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("K must be at least one.")
        if self.sampling_strategy not in {"uniform", "cluster", "uct"}:
            raise ValueError("sampling_strategy must be uniform, cluster, or uct.")
        if self.max_expansions is not None and self.max_expansions < 1:
            raise ValueError("max_expansions must be positive or None.")
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be positive.")
        if not isfinite(self.eta) or not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be finite and in [0, 1].")
        if not isfinite(self.cb) or self.cb < 0.0:
            raise ValueError("cb must be finite and non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isfinite(self.cost_tolerance) or self.cost_tolerance < 0.0:
            raise ValueError("cost_tolerance must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class SeeAStarResult:
    """One search result with search and selective-sampling measurements."""

    found: bool
    path: tuple[Coord, ...] | None
    found_ground_truth_cost: float
    expanded_nodes: int
    generated_nodes: int
    reexpansions: int
    max_open_size: int
    final_open_size: int
    mean_open_size: float
    closed_size: int
    sampling_time_seconds: float
    search_time_seconds: float
    iterations: int
    eligible_open_sizes: tuple[int, ...]
    candidate_sizes: tuple[int, ...]
    expanded_order: tuple[Coord, ...]
    stop_reason: str
    sampling_strategy: str


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


def _make_sampler(config: SeeAStarConfig, shape: tuple[int, int]) -> CandidateSampler:
    if config.sampling_strategy == "uniform":
        return UniformSampler(config.k, config.seed)
    if config.sampling_strategy == "cluster":
        return CompetitiveClusterSampler(
            config.k,
            shape,
            num_clusters=config.num_clusters,
            eta=config.eta,
            seed=config.seed,
        )
    return UCTLikeSampler(config.k, config.cb)


def _reconstruct_path(
    parents: dict[Coord, Coord], start: Coord, goal: Coord
) -> tuple[Coord, ...]:
    reverse_path = [goal]
    current = goal
    while current != start:
        parent = parents.get(current)
        if parent is None:
            raise RuntimeError("SeeA* parent map could not reconstruct a found path.")
        reverse_path.append(parent)
        if len(reverse_path) > len(parents) + 1:
            raise RuntimeError("SeeA* parent map contains a cycle.")
        current = parent
    reverse_path.reverse()
    return tuple(reverse_path)


def seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    heuristic: HeuristicFunction = chebyshev_heuristic,
    config: SeeAStarConfig = SeeAStarConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> SeeAStarResult:
    """Run the generic SeeA* algorithm using one published sampling strategy.

    The paper defines SeeA* as candidate generation ``D=F(OPEN,K)`` followed
    by minimum-f selection from D (Algorithm 1). Strategy-specific candidate
    construction is implemented in :mod:`seeastar.sampling`. Grid coordinates
    are graph states, and all costs come from the shared movement module.
    """

    started_at = perf_counter()
    array = validate_map(grid)
    start_coord = _validate_endpoint(array, start, "start")
    goal_coord = _validate_endpoint(array, goal, "goal")
    if not isinstance(cost_config, GroundTruthCostConfig):
        raise TypeError("cost_config must be a GroundTruthCostConfig instance.")
    if not callable(heuristic):
        raise TypeError("heuristic must be a callable accepting (node, goal).")

    sampler = _make_sampler(config, array.shape)
    heuristic_cache: dict[Coord, float] = {}

    def h_value(node: Coord) -> float:
        if node not in heuristic_cache:
            value = float(heuristic(node, goal_coord))
            if not isfinite(value):
                raise ValueError(f"Heuristic returned a non-finite value for {node}.")
            heuristic_cache[node] = value
        return heuristic_cache[node]

    g_score: dict[Coord, float] = {start_coord: 0.0}
    f_score: dict[Coord, float] = {start_coord: h_value(start_coord)}
    parents: dict[Coord, Coord] = {}
    depths: dict[Coord, int] = {start_coord: 0}
    discovered: set[Coord] = {start_coord}
    open_states: set[Coord] = {start_coord}
    closed_states: set[Coord] = set()
    expansion_counts: dict[Coord, int] = {}
    sampling_seconds = 0.0
    expanded_order: list[Coord] = []
    candidate_sizes: list[int] = []
    eligible_open_sizes: list[int] = []
    sampler_update_started = perf_counter()
    sampler.on_add(start_coord)
    sampling_seconds += perf_counter() - sampler_update_started

    expanded_nodes = 0
    reexpansions = 0
    max_open_size = 1
    iterations = 0
    open_size_total = 0
    found_path: tuple[Coord, ...] | None = None
    found_cost = inf
    stop_reason = "open_exhausted"

    while open_states:
        if config.max_expansions is not None and expanded_nodes >= config.max_expansions:
            stop_reason = "max_expansions"
            break
        selection_started = perf_counter()
        ordered_open = sorted(open_states)
        sampler.assert_open_consistent(ordered_open)
        open_size_total += len(ordered_open)
        iterations += 1
        eligible_open_sizes.append(len(ordered_open))
        candidates = sampler.select(ordered_open, f_score, depths)
        if not candidates:
            raise RuntimeError("Candidate sampler returned an empty set for non-empty OPEN.")
        if len(candidates) != len(set(candidates)):
            raise RuntimeError("Candidate sampler returned duplicate states.")
        if not set(candidates).issubset(open_states):
            raise RuntimeError("Candidate sampler returned a state outside OPEN.")
        candidate_sizes.append(len(candidates))
        current = lowest_f_candidate(candidates, f_score)
        sampling_seconds += perf_counter() - selection_started

        open_states.remove(current)
        sampler_update_started = perf_counter()
        sampler.on_remove(current)
        sampling_seconds += perf_counter() - sampler_update_started
        if current == goal_coord:
            found_path = _reconstruct_path(parents, start_coord, goal_coord)
            found_cost = float(g_score[current])
            measured_cost = sum(
                get_transition_cost(array, source, destination, cost_config)
                for source, destination in zip(found_path, found_path[1:])
            )
            if abs(measured_cost - found_cost) > config.cost_tolerance:
                raise RuntimeError("SeeA* path cost differs from accumulated transition costs.")
            stop_reason = "goal"
            break

        if expansion_counts.get(current, 0) > 0:
            reexpansions += 1
        expansion_counts[current] = expansion_counts.get(current, 0) + 1
        expanded_order.append(current)
        closed_states.add(current)
        expanded_nodes += 1

        for neighbor in get_neighbors(array, current, cost_config):
            step_cost = get_transition_cost(array, current, neighbor, cost_config)
            candidate_g = g_score[current] + step_cost
            previous_g = g_score.get(neighbor, inf)
            if not isfinite(previous_g) or candidate_g < previous_g - config.cost_tolerance:
                g_score[neighbor] = candidate_g
                parents[neighbor] = current
                depths[neighbor] = depths[current] + 1
                f_score[neighbor] = candidate_g + h_value(neighbor)
                if neighbor in closed_states:
                    closed_states.remove(neighbor)
                if neighbor not in open_states:
                    open_states.add(neighbor)
                    sampler_update_started = perf_counter()
                    sampler.on_add(neighbor)
                    sampling_seconds += perf_counter() - sampler_update_started
                if neighbor not in discovered:
                    discovered.add(neighbor)
                max_open_size = max(max_open_size, len(open_states))

        if config.max_expansions is not None and expanded_nodes >= config.max_expansions:
            stop_reason = "max_expansions"
            break

    search_seconds = perf_counter() - started_at
    return SeeAStarResult(
        found=found_path is not None,
        path=found_path,
        found_ground_truth_cost=float(found_cost),
        expanded_nodes=expanded_nodes,
        generated_nodes=len(discovered),
        reexpansions=reexpansions,
        max_open_size=max_open_size,
        final_open_size=len(open_states),
        mean_open_size=float(open_size_total / iterations) if iterations else 0.0,
        closed_size=len(closed_states),
        sampling_time_seconds=float(sampling_seconds),
        search_time_seconds=float(search_seconds),
        iterations=iterations,
        eligible_open_sizes=tuple(eligible_open_sizes),
        candidate_sizes=tuple(candidate_sizes),
        expanded_order=tuple(expanded_order),
        stop_reason=stop_reason,
        sampling_strategy=config.sampling_strategy,
    )
