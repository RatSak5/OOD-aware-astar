"""Passive Neural A* counters with instrumented/uninstrumented equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

import neural_astar.search as neural_search_module
from algorithms.movement import Coord, DEFAULT_COST_CONFIG, GroundTruthCostConfig
from neural_astar.search import NeuralAStarResult, run_neural_astar


@dataclass(frozen=True, slots=True)
class NeuralAStarTrace:
    """Counters collected by read-only wrappers around existing search helpers."""

    generated_nodes: int
    maximum_open_size: int
    open_sizes: tuple[int, ...]


def run_instrumented_neural_astar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: np.ndarray,
    *,
    temperature: float | None = None,
    tie_break_epsilon: float = 0.001,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> tuple[NeuralAStarResult, NeuralAStarTrace]:
    """Run passive tracing, then an ordinary run and verify exact equivalence.

    Instrumentation records OPEN-mask sizes and unique cells first admitted to
    OPEN. The wrappers return the original helper outputs unchanged. A second
    uninstrumented run is compared on selected-node order, final path, guidance
    map, accumulated guidance-cost map, and goal guidance cost. Timings from
    the instrumented run are diagnostic only and must not enter benchmark
    runtime tables; use the uninstrumented result for search timing.
    """

    generated: set[Coord] = {start}
    opened_ever: set[Coord] = {start}
    closed: set[Coord] = set()
    open_sizes: list[int] = []
    original_normalizer = neural_search_module.normalized_open_weights
    original_neighbors = neural_search_module.get_neighbors

    def observe_normalizer(
        accumulated: np.ndarray,
        heuristic: np.ndarray,
        open_mask: np.ndarray,
        temperature_value: float,
    ) -> np.ndarray:
        opened = np.asarray(open_mask, dtype=bool)
        open_sizes.append(int(np.count_nonzero(opened)))
        opened_ever.update(
            (int(row), int(column)) for row, column in np.argwhere(opened)
        )
        weights = original_normalizer(
            accumulated, heuristic, open_mask, temperature_value
        )
        selected_index = int(np.argmax(weights))
        selected = tuple(
            int(value) for value in np.unravel_index(selected_index, weights.shape)
        )
        closed.add(selected)
        return weights

    def observe_neighbors(
        obstacle_map: np.ndarray,
        source: Coord,
        transition_config: GroundTruthCostConfig,
    ) -> list[Coord]:
        neighbors = original_neighbors(obstacle_map, source, transition_config)
        for neighbor in neighbors:
            if neighbor in closed:
                continue
            if neighbor not in opened_ever:
                generated.add(neighbor)
                opened_ever.add(neighbor)
        return neighbors

    with (
        patch.object(
            neural_search_module,
            "normalized_open_weights",
            new=observe_normalizer,
        ),
        patch.object(
            neural_search_module,
            "get_neighbors",
            new=observe_neighbors,
        ),
    ):
        measured = run_neural_astar(
            grid,
            start,
            goal,
            phi,
            temperature=temperature,
            tie_break_epsilon=tie_break_epsilon,
            cost_config=cost_config,
        )

    trace = NeuralAStarTrace(
        generated_nodes=len(generated),
        maximum_open_size=max(open_sizes, default=0),
        open_sizes=tuple(open_sizes),
    )
    reference = run_neural_astar(
        grid,
        start,
        goal,
        phi,
        temperature=temperature,
        tie_break_epsilon=tie_break_epsilon,
        cost_config=cost_config,
    )
    equivalent = (
        measured.selected_order == reference.selected_order
        and measured.path == reference.path
        and measured.goal_guidance_cost == reference.goal_guidance_cost
        and np.array_equal(measured.guidance_map, reference.guidance_map)
        and np.array_equal(
            measured.accumulated_guidance_cost,
            reference.accumulated_guidance_cost,
        )
        and np.array_equal(measured.expanded_mask, reference.expanded_mask)
    )
    if not equivalent:
        raise RuntimeError(
            "Passive Neural A* instrumentation changed or failed to reproduce search output."
        )
    return reference, trace
