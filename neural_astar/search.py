"""Forward Neural A* search using a learned, per-cell guidance-cost map."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt
from time import perf_counter

import numpy as np

from algorithms.movement import (
    Coord,
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    validate_map,
)
from neural_astar.guidance import accumulate_guidance_cost, validate_guidance_map


@dataclass(frozen=True, slots=True)
class NeuralAStarResult:
    """Search result whose cost fields explicitly remain guidance quantities."""

    found: bool
    path: tuple[Coord, ...] | None
    selected_order: tuple[Coord, ...]
    expanded_mask: np.ndarray
    accumulated_guidance_cost: np.ndarray
    guidance_map: np.ndarray
    final_open_size: int
    closed_size: int
    search_time_seconds: float
    goal_guidance_cost: float
    temperature: float
    tie_break_epsilon: float
    expanded_nodes: int = 0
    generated_nodes: int = 0
    reexpansions: int = 0
    max_open_size: int = 0
    max_closed_size: int = 0


def neural_astar_heuristic(
    node: Coord,
    goal: Coord,
    *,
    tie_break_epsilon: float = 0.001,
) -> float:
    """Return paper's Chebyshev heuristic plus its small Euclidean tie-break.

    The Chebyshev component is a lower bound for the project's unit-cost
    eight-connected graph. The tie-break term reproduces Yonetani et al. and
    is kept specific to Neural A*; it is not a conventional admissible-heuristic
    guarantee for the sum.
    """

    if tie_break_epsilon < 0 or not np.isfinite(tie_break_epsilon):
        raise ValueError("tie_break_epsilon must be finite and non-negative.")
    dr, dc = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    return float(max(dr, dc) + tie_break_epsilon * sqrt(dr * dr + dc * dc))


def normalized_open_weights(
    accumulated_guidance_cost: np.ndarray,
    heuristic: np.ndarray,
    open_mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Evaluate Eq. (3)'s normalized exponential weights stably over OPEN.

    Subtracting the minimum open ``G+H`` before exponentiation is algebraically
    equivalent to the paper's normalized expression and prevents overflow.
    """

    guidance_cost = np.asarray(accumulated_guidance_cost, dtype=np.float64)
    h_values = np.asarray(heuristic, dtype=np.float64)
    opened = np.asarray(open_mask, dtype=bool)
    if guidance_cost.shape != h_values.shape or opened.shape != guidance_cost.shape:
        raise ValueError("G, H, and OPEN must have identical dimensions.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not opened.any():
        raise ValueError("Cannot normalize selector weights for an empty OPEN set.")
    priorities = guidance_cost + h_values
    if not np.isfinite(priorities[opened]).all():
        raise ValueError("OPEN priorities must be finite.")
    minimum = float(np.min(priorities[opened]))
    weights = np.zeros(guidance_cost.shape, dtype=np.float64)
    weights[opened] = np.exp(-(priorities[opened] - minimum) / temperature)
    total = float(weights.sum())
    if total == 0.0 or not np.isfinite(total):
        raise FloatingPointError("Stable OPEN-weight normalization failed.")
    weights /= total
    return weights


def run_neural_astar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: np.ndarray,
    *,
    temperature: float | None = None,
    tie_break_epsilon: float = 0.001,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> NeuralAStarResult:
    """Run the hard forward pass of Neural A* with the supplied guidance map.

    The selected OPEN cell minimizes ``G_phi + H`` (Eq. (3)). Entering a
    neighbor adds that destination cell's ``phi`` (Eqs. (5)–(6)); the shared
    ground-truth transition cost is intentionally not used to accumulate this
    guidance value. Successors are always provided by the project's shared
    eight-connected ``get_neighbors`` function, including its no-corner-cutting
    adaptation.
    """

    started = perf_counter()
    obstacle_map = validate_map(grid)
    guidance = validate_guidance_map(phi, obstacle_map.shape)
    height, width = obstacle_map.shape
    for label, coordinate in (("start", start), ("goal", goal)):
        if len(coordinate) != 2:
            raise ValueError(f"{label} must be a (row, column) coordinate.")
        row, col = coordinate
        if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
            raise ValueError(f"{label} coordinates must be integers.")
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
        if obstacle_map[row, col] != 0:
            raise ValueError(f"{label} cell must be traversable.")
    if not isinstance(cost_config, GroundTruthCostConfig):
        raise TypeError("cost_config must be a GroundTruthCostConfig instance.")
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    tau = sqrt(width) if temperature is None else float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive.")
    if tie_break_epsilon < 0 or not np.isfinite(tie_break_epsilon):
        raise ValueError("tie_break_epsilon must be finite and non-negative.")

    rows, cols = np.indices(obstacle_map.shape)
    row_distance = np.abs(rows - goal[0])
    col_distance = np.abs(cols - goal[1])
    heuristic = np.maximum(row_distance, col_distance).astype(np.float64)
    heuristic += tie_break_epsilon * np.hypot(row_distance, col_distance)

    open_mask = np.zeros(obstacle_map.shape, dtype=bool)
    closed_mask = np.zeros(obstacle_map.shape, dtype=bool)
    accumulated = np.zeros(obstacle_map.shape, dtype=np.float64)
    open_mask[start] = True
    open_count = 1
    max_open_size = 1
    closed_count = 0
    max_closed_size = 0
    parents: dict[Coord, Coord | None] = {start: None}
    selected_order: list[Coord] = []
    found = False
    goal_cost = inf

    # Every passable node is selected at most once because CLOSED nodes are not
    # reopened, matching the paper's non-negative guidance-cost A* procedure.
    while open_mask.any():
        weights = normalized_open_weights(accumulated, heuristic, open_mask, tau)
        selected_flat = int(np.argmax(weights))
        selected = tuple(int(value) for value in np.unravel_index(selected_flat, weights.shape))
        open_mask[selected] = False
        open_count -= 1
        closed_mask[selected] = True
        closed_count += 1
        max_closed_size = max(max_closed_size, closed_count)
        selected_order.append(selected)

        if selected == goal:
            found = True
            goal_cost = float(accumulated[goal])
            break

        selected_cost = float(accumulated[selected])
        for neighbor in get_neighbors(obstacle_map, selected, cost_config):
            if closed_mask[neighbor]:
                continue
            candidate = accumulate_guidance_cost(
                selected_cost, float(guidance[neighbor])
            )
            if not open_mask[neighbor]:
                accumulated[neighbor] = candidate
                open_mask[neighbor] = True
                open_count += 1
                max_open_size = max(max_open_size, open_count)
                parents[neighbor] = selected
            elif candidate < accumulated[neighbor]:
                accumulated[neighbor] = candidate
                parents[neighbor] = selected

    path: tuple[Coord, ...] | None = None
    if found:
        reverse_path = [goal]
        current = goal
        while current != start:
            parent = parents.get(current)
            if parent is None:
                raise RuntimeError("Neural A* parent map could not reconstruct the selected goal.")
            reverse_path.append(parent)
            current = parent
            if len(reverse_path) > int(np.count_nonzero(obstacle_map == 0)):
                raise RuntimeError("Neural A* parent map contains a cycle.")
        path = tuple(reversed(reverse_path))

    elapsed = perf_counter() - started
    return NeuralAStarResult(
        found=found,
        path=path,
        selected_order=tuple(selected_order),
        expanded_mask=closed_mask,
        accumulated_guidance_cost=accumulated,
        guidance_map=guidance.copy(),
        final_open_size=int(np.count_nonzero(open_mask)),
        closed_size=int(np.count_nonzero(closed_mask)),
        search_time_seconds=elapsed,
        goal_guidance_cost=goal_cost,
        temperature=tau,
        tie_break_epsilon=float(tie_break_epsilon),
        expanded_nodes=len(selected_order),
        generated_nodes=len(parents),
        reexpansions=0,
        max_open_size=max_open_size,
        max_closed_size=max_closed_size,
    )
