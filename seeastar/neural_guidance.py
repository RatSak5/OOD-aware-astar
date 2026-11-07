"""Uniform-sampling SeeA* whose accumulated g-values use Neural A* guidance."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf, isfinite
from time import perf_counter

import numpy as np

from algorithms.movement import (
    Coord,
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    get_transition_cost,
    is_legal_transition,
    validate_map,
)
from seeastar.sampling import UniformSampler, lowest_f_candidate


NEURAL_GUIDANCE_TIE_BREAK_EPSILON = 0.001


@dataclass(frozen=True, slots=True)
class GuidanceSeeAStarConfig:
    """Fixed-K settings for guidance-based uniform SeeA*."""

    k: int = 16
    seed: int = 0
    max_expansions: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("K must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.max_expansions is not None and self.max_expansions < 1:
            raise ValueError("max_expansions must be positive or None.")


@dataclass(frozen=True, slots=True)
class GuidanceSelectionTrace:
    """Read-only record of a draw, selected candidate, and actual expansion."""

    eligible_open: tuple[Coord, ...]
    sampled_candidates: tuple[Coord, ...]
    sampled_f_values: tuple[tuple[Coord, float], ...]
    selected: Coord
    expanded: Coord | None


@dataclass(frozen=True, slots=True)
class GuidanceUpdateTrace:
    """Record of an OPEN-state guidance-cost update decision."""

    source: Coord
    destination: Coord
    previous_g: float | None
    candidate_g: float | None
    previous_parent: Coord | None
    updated: bool
    state: str


@dataclass(frozen=True, slots=True)
class GuidanceSeeAStarResult:
    """Search outputs with Neural-guidance and ground-truth costs separate."""

    found: bool
    path: tuple[Coord, ...] | None
    found_guidance_cost: float
    found_ground_truth_cost: float
    guidance_map: np.ndarray
    accumulated_guidance_cost_map: np.ndarray
    heuristic_map: np.ndarray
    priority_map: np.ndarray
    discovered_mask: np.ndarray
    closed_mask: np.ndarray
    parent_map: dict[Coord, Coord]
    expanded_nodes: int
    generated_nodes: int
    reexpansions: int
    max_open_size: int
    max_closed_size: int
    final_open_size: int
    mean_open_size: float
    closed_size: int
    sampling_time_seconds: float
    search_time_seconds: float
    iterations: int
    eligible_open_sizes: tuple[int, ...]
    candidate_sizes: tuple[int, ...]
    selected_order: tuple[Coord, ...]
    expanded_order: tuple[Coord, ...]
    stop_reason: str
    sampling_strategy: str
    configured_k: int
    selection_trace: tuple[GuidanceSelectionTrace, ...] = ()
    update_trace: tuple[GuidanceUpdateTrace, ...] = ()


def neural_guidance_heuristic(node: Coord, goal: Coord) -> float:
    """Return the specified Chebyshev plus 0.001 Euclidean heuristic."""

    dr = abs(int(node[0]) - int(goal[0]))
    dc = abs(int(node[1]) - int(goal[1]))
    return float(max(dr, dc) + NEURAL_GUIDANCE_TIE_BREAK_EPSILON * hypot(dr, dc))


def guidance_heuristic_map(shape: tuple[int, int], goal: Coord) -> np.ndarray:
    """Calculate h(v)=d_Chebyshev(v,goal)+0.001*d_Euclidean(v,goal)."""

    rows, columns = np.indices(shape)
    row_distance = np.abs(rows - int(goal[0]))
    column_distance = np.abs(columns - int(goal[1]))
    return (
        np.maximum(row_distance, column_distance).astype(np.float64)
        + NEURAL_GUIDANCE_TIE_BREAK_EPSILON
        * np.hypot(row_distance, column_distance)
    )


def _validate_endpoint(grid: np.ndarray, coord: Coord, label: str) -> Coord:
    if len(coord) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in coord
    ):
        raise ValueError(f"{label} must be a pair of integer coordinates.")
    normalized = (int(coord[0]), int(coord[1]))
    if not (0 <= normalized[0] < grid.shape[0] and 0 <= normalized[1] < grid.shape[1]):
        raise ValueError(f"{label} {coord!r} is outside the grid.")
    if grid[normalized] != 0:
        raise ValueError(f"{label} cell must be traversable.")
    return normalized


def _reconstruct_path(
    parents: dict[Coord, Coord], start: Coord, goal: Coord, *, free_cell_count: int
) -> tuple[Coord, ...]:
    reverse_path = [goal]
    current = goal
    while current != start:
        parent = parents.get(current)
        if parent is None:
            raise RuntimeError("Guidance SeeA* could not reconstruct the selected goal.")
        reverse_path.append(parent)
        if len(reverse_path) > free_cell_count + 1:
            raise RuntimeError("Guidance SeeA* parent pointers contain a cycle.")
        current = parent
    reverse_path.reverse()
    return tuple(reverse_path)


def _validate_neural_guidance_map(
    phi: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Validate the guidance-search contract without rescaling ``Phi``.

    This corrected search accepts any finite non-negative guidance values.
    Neural A* inference normally emits sigmoid-bounded values, but that model
    output range is not an extra restriction on this search API.
    """

    try:
        guidance = np.asarray(phi, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Phi must be a numeric two-dimensional array.") from error
    if guidance.ndim != 2 or guidance.size == 0:
        raise ValueError("Phi must be a non-empty two-dimensional array.")
    if guidance.shape != shape:
        raise ValueError(f"Phi shape {guidance.shape} does not match map shape {shape}.")
    if not np.isfinite(guidance).all():
        raise ValueError("Every Phi value must be finite.")
    if np.any(guidance < 0.0):
        raise ValueError("Every Phi value must be non-negative.")
    return guidance


def run_guidance_uniform_seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: np.ndarray,
    config: GuidanceSeeAStarConfig = GuidanceSeeAStarConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    *,
    capture_trace: bool = False,
) -> GuidanceSeeAStarResult:
    """Search with uniform OPEN sampling and destination-cell Neural costs.

    The supplied ``phi`` is validated and copied without normalization or
    rescaling. Search initializes ``g(start)=0`` and every guidance update is
    ``g_candidate(v)=g(u)+phi(v)``. The shared ground-truth transition model is
    used only after path reconstruction to report path cost; it never affects
    the search priorities. CLOSED nodes are never reopened.
    """

    started = perf_counter()
    obstacle_map = validate_map(grid)
    source = _validate_endpoint(obstacle_map, start, "start")
    target = _validate_endpoint(obstacle_map, goal, "goal")
    if not isinstance(cost_config, GroundTruthCostConfig):
        raise TypeError("cost_config must be a GroundTruthCostConfig instance.")
    guidance = _validate_neural_guidance_map(phi, obstacle_map.shape).copy()
    heuristic = guidance_heuristic_map(obstacle_map.shape, target)
    height, width = obstacle_map.shape
    accumulated = np.zeros((height, width), dtype=np.float64)
    priorities = np.zeros((height, width), dtype=np.float64)
    discovered_mask = np.zeros((height, width), dtype=bool)
    closed_mask = np.zeros((height, width), dtype=bool)
    discovered_mask[source] = True
    accumulated[source] = 0.0  # Phi(start) is deliberately not added.
    priorities[source] = heuristic[source]

    parents: dict[Coord, Coord] = {}
    open_nodes: set[Coord] = {source}
    closed_nodes: set[Coord] = set()
    sampler = UniformSampler(config.k, config.seed)
    sampler.on_add(source)

    expanded_order: list[Coord] = []
    selected_order: list[Coord] = []
    eligible_sizes: list[int] = []
    candidate_sizes: list[int] = []
    selection_trace: list[GuidanceSelectionTrace] = []
    update_trace: list[GuidanceUpdateTrace] = []
    expanded_nodes = 0
    max_open_size = 1
    max_closed_size = 0
    iterations = 0
    open_size_total = 0
    sampling_seconds = 0.0
    found_path: tuple[Coord, ...] | None = None
    found_guidance_cost = inf
    stop_reason = "open_exhausted"

    while open_nodes:
        if config.max_expansions is not None and expanded_nodes >= config.max_expansions:
            stop_reason = "max_expansions"
            break
        ordered_open = sorted(open_nodes)
        sampler.assert_open_consistent(ordered_open)
        open_size_total += len(ordered_open)
        iterations += 1
        eligible_sizes.append(len(ordered_open))

        selection_started = perf_counter()
        candidates = sampler.select(ordered_open, {
            node: float(priorities[node]) for node in ordered_open
        }, {})
        if not candidates:
            raise RuntimeError("Uniform sampler returned no candidate for non-empty OPEN.")
        if len(candidates) != len(set(candidates)):
            raise RuntimeError("Uniform sampler returned duplicate candidates.")
        if not set(candidates).issubset(open_nodes):
            raise RuntimeError("Uniform sampler returned a node outside OPEN.")
        candidate_f_values = {node: float(priorities[node]) for node in candidates}
        selected = lowest_f_candidate(candidates, candidate_f_values)
        expanded_node = None if selected == target else selected
        sampling_seconds += perf_counter() - selection_started
        candidate_sizes.append(len(candidates))
        selected_order.append(selected)
        if capture_trace:
            selection_trace.append(
                GuidanceSelectionTrace(
                    eligible_open=tuple(ordered_open),
                    sampled_candidates=tuple(candidates),
                    sampled_f_values=tuple((node, candidate_f_values[node]) for node in candidates),
                    selected=selected,
                    expanded=expanded_node,
                )
            )

        open_nodes.remove(selected)
        sampler.on_remove(selected)
        # Count a valid OPEN removal as expanded, including a selected goal.
        # The goal still terminates before successor generation as before.
        expanded_nodes += 1
        if selected == target:
            found_path = _reconstruct_path(
                parents,
                source,
                target,
                free_cell_count=int(np.count_nonzero(obstacle_map == 0)),
            )
            found_guidance_cost = float(accumulated[target])
            stop_reason = "goal"
            break

        closed_nodes.add(selected)
        closed_mask[selected] = True
        expanded_order.append(selected)
        max_closed_size = max(max_closed_size, len(closed_nodes))

        # Every legal neighbor is already filtered by the common 8-connected
        # generator, including obstacle, boundary, and corner-cutting rules.
        for neighbor in get_neighbors(obstacle_map, selected, cost_config):
            if neighbor in closed_nodes:
                if capture_trace:
                    update_trace.append(
                        GuidanceUpdateTrace(
                            source=selected,
                            destination=neighbor,
                            previous_g=float(accumulated[neighbor]),
                            candidate_g=None,
                            previous_parent=parents.get(neighbor),
                            updated=False,
                            state="closed_excluded",
                        )
                    )
                continue

            candidate_g = float(accumulated[selected] + guidance[neighbor])
            previous_g = float(accumulated[neighbor]) if discovered_mask[neighbor] else None
            previous_parent = parents.get(neighbor)
            is_new = neighbor not in open_nodes and not discovered_mask[neighbor]
            improved = is_new or (neighbor in open_nodes and candidate_g < accumulated[neighbor])
            if capture_trace:
                update_trace.append(
                    GuidanceUpdateTrace(
                        source=selected,
                        destination=neighbor,
                        previous_g=previous_g,
                        candidate_g=candidate_g,
                        previous_parent=previous_parent,
                        updated=bool(improved),
                        state="new" if is_new else "open",
                    )
                )
            if not improved:
                continue

            accumulated[neighbor] = candidate_g
            priorities[neighbor] = candidate_g + heuristic[neighbor]
            discovered_mask[neighbor] = True
            parents[neighbor] = selected
            if is_new:
                open_nodes.add(neighbor)
                sampler.on_add(neighbor)
            max_open_size = max(max_open_size, len(open_nodes))

        if config.max_expansions is not None and expanded_nodes >= config.max_expansions:
            stop_reason = "max_expansions"
            break

    ground_truth_cost = inf
    if found_path is not None:
        measured = 0.0
        for source_cell, destination_cell in zip(found_path, found_path[1:]):
            if not is_legal_transition(
                obstacle_map, source_cell, destination_cell, cost_config
            ):
                raise RuntimeError("Guidance SeeA* reconstructed an illegal path transition.")
            measured += get_transition_cost(
                obstacle_map, source_cell, destination_cell, cost_config
            )
        ground_truth_cost = float(measured)

    elapsed = perf_counter() - started
    return GuidanceSeeAStarResult(
        found=found_path is not None,
        path=found_path,
        found_guidance_cost=float(found_guidance_cost),
        found_ground_truth_cost=float(ground_truth_cost),
        guidance_map=guidance,
        accumulated_guidance_cost_map=accumulated,
        heuristic_map=heuristic,
        priority_map=priorities,
        discovered_mask=discovered_mask,
        closed_mask=closed_mask,
        parent_map=dict(parents),
        expanded_nodes=expanded_nodes,
        generated_nodes=int(np.count_nonzero(discovered_mask)),
        reexpansions=0,
        max_open_size=max_open_size,
        max_closed_size=max_closed_size,
        final_open_size=len(open_nodes),
        mean_open_size=float(open_size_total / iterations) if iterations else 0.0,
        closed_size=len(closed_nodes),
        sampling_time_seconds=float(sampling_seconds),
        search_time_seconds=float(elapsed),
        iterations=iterations,
        eligible_open_sizes=tuple(eligible_sizes),
        candidate_sizes=tuple(candidate_sizes),
        selected_order=tuple(selected_order),
        expanded_order=tuple(expanded_order),
        stop_reason=stop_reason,
        sampling_strategy="uniform",
        configured_k=config.k,
        selection_trace=tuple(selection_trace),
        update_trace=tuple(update_trace),
    )


def run_neural_guidance_seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: np.ndarray,
    *,
    k: int = 16,
    seed: int = 0,
    max_expansions: int | None = None,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    capture_trace: bool = False,
) -> GuidanceSeeAStarResult:
    """Run fixed-K Neural-guidance SeeA* through the shared search core.

    ``phi`` is the Neural A* guidance map, never an H1/H2/H3 heuristic map.
    Guidance accumulation is ``g(start)=0`` and ``g_candidate(v)=g(u)+Phi(v)``.
    The returned ``found_ground_truth_cost`` counts legal transitions, while
    ``found_guidance_cost`` sums destination-cell Phi values along the path.
    """

    return run_guidance_uniform_seeastar(
        grid,
        start,
        goal,
        phi,
        GuidanceSeeAStarConfig(k=k, seed=seed, max_expansions=max_expansions),
        cost_config,
        capture_trace=capture_trace,
    )
