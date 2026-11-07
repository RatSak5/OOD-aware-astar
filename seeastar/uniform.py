"""Dedicated SeeA* implementation using only uniform candidate sampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from algorithms.astar import HeuristicFunction, chebyshev_heuristic
from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig, Coord
from seeastar.original import SeeAStarConfig, SeeAStarResult, seeastar


@dataclass(frozen=True, slots=True)
class UniformSeeAStarConfig:
    """Settings for SeeA* Algorithm 2 with a fixed uniform-sampling budget.

    The config intentionally has no clustering or UCT fields. K stays fixed
    throughout one search; OOD-adaptive K is a separate proposed extension.
    """

    k: int = 50
    max_expansions: int | None = None
    seed: int = 0
    cost_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("K must be at least one.")
        if self.max_expansions is not None and self.max_expansions < 1:
            raise ValueError("max_expansions must be positive or None.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isfinite(self.cost_tolerance) or self.cost_tolerance < 0.0:
            raise ValueError("cost_tolerance must be finite and non-negative.")


def uniform_seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    heuristic: HeuristicFunction = chebyshev_heuristic,
    config: UniformSeeAStarConfig = UniformSeeAStarConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> SeeAStarResult:
    """Run SeeA* using only Algorithm 2's uniform K-subset sampler.

    If ``|OPEN| <= K``, every OPEN state is a candidate. Otherwise, K distinct
    states are sampled uniformly without replacement from OPEN; the candidate
    with the lowest ``f=g+h`` is expanded. This wrapper shares the graph-search
    update rules with the original generic SeeA* implementation while exposing
    no way to select clustering or UCT sampling.
    """

    fixed_config = SeeAStarConfig(
        k=config.k,
        sampling_strategy="uniform",
        max_expansions=config.max_expansions,
        seed=config.seed,
        cost_tolerance=config.cost_tolerance,
    )
    return seeastar(grid, start, goal, heuristic, fixed_config, cost_config)

