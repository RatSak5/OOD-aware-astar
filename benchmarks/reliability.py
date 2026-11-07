"""Per-map learned-heuristic error and OOD-score association analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from algorithms.movement import get_neighbors, get_transition_cost
from data.dataset import DatasetSample
from benchmarks.statistics import bootstrap_mean_interval


HEURISTIC_ERROR_METRICS = (
    "mae",
    "rmse",
    "overestimation_rate",
    "mean_overestimation_magnitude",
    "maximum_overestimation_magnitude",
    "consistency_violation_rate",
    "mean_consistency_violation_magnitude",
    "maximum_consistency_violation_magnitude",
)


def heuristic_error_metrics(
    sample: DatasetSample,
    prediction_map: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Measure raw heuristic error and constraints on reachable valid states.

    Consistency uses every valid directed legal edge. For the undirected grid,
    both directions are visited separately and use the shared unit transition
    cost. The prediction map is not clipped or corrected.
    """

    prediction = np.asarray(prediction_map, dtype=np.float64)
    grid = sample.grid_map.grid
    target = np.asarray(sample.ground_truth.distance_map, dtype=np.float64)
    reachable = np.asarray(sample.ground_truth.reachable_mask, dtype=bool) & (grid == 0)
    if prediction.shape != grid.shape or target.shape != grid.shape or reachable.shape != grid.shape:
        raise ValueError("Prediction, target, reachability, and map shapes must agree.")
    if not np.isfinite(prediction[reachable]).all():
        raise ValueError("Raw heuristic predictions must be finite on reachable cells.")
    residual = prediction[reachable] - target[reachable]
    overestimation = np.maximum(residual, 0.0)

    violations: list[float] = []
    for row, column in np.argwhere(reachable):
        source = (int(row), int(column))
        for destination in get_neighbors(grid, source, sample.cost_config):
            if not reachable[destination]:
                continue
            cost = get_transition_cost(grid, source, destination, sample.cost_config)
            violations.append(
                max(0.0, float(prediction[source] - cost - prediction[destination]))
            )
    violation_values = np.asarray(violations, dtype=np.float64)
    if not len(residual):
        raise ValueError("A heuristic evaluation map must contain reachable traversable cells.")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "overestimation_rate": float(np.mean(residual > tolerance)),
        "mean_overestimation_magnitude": float(overestimation.mean()),
        "maximum_overestimation_magnitude": float(overestimation.max(initial=0.0)),
        "consistency_violation_rate": (
            float(np.mean(violation_values > tolerance)) if len(violation_values) else 0.0
        ),
        "mean_consistency_violation_magnitude": (
            float(violation_values.mean()) if len(violation_values) else 0.0
        ),
        "maximum_consistency_violation_magnitude": (
            float(violation_values.max(initial=0.0)) if len(violation_values) else 0.0
        ),
        "valid_state_count": float(np.count_nonzero(reachable)),
        "valid_directed_edge_count": float(len(violation_values)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values from 1 with average ranks for ties, without SciPy."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = ((begin + 1) + end) / 2.0
        begin = end
    return ranks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Spearman rho using average ranks and Pearson correlation."""

    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Spearman inputs must be same-shaped one-dimensional arrays.")
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 2:
        return float("nan")
    first_ranks, second_ranks = _average_ranks(first), _average_ranks(second)
    if np.std(first_ranks) == 0.0 or np.std(second_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _spearman_ci(
    scores: np.ndarray,
    errors: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if len(scores) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(scores), size=len(scores))
        rho = spearman_rank_correlation(scores[indices], errors[indices])
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def analyze_score_heuristic_reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str = "smooth_ood_score",
    group_field: str = "distribution",
    method_field: str = "heuristic_method",
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Correlate one map-level OOD score with raw heuristic error metrics.

    Each input row must represent a unique map, not a search-seed repetition.
    Correlation confidence intervals resample maps as the statistical unit.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    methods = sorted({str(row[method_field]) for row in rows})
    output: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        selected = [row for row in rows if str(row[method_field]) == method]
        score_values = np.asarray([float(row[score_field]) for row in selected], dtype=np.float64)
        groups = sorted({str(row[group_field]) for row in selected})
        group_results: dict[str, Any] = {}
        for group_index, group in enumerate(["all", *groups]):
            subset = selected if group == "all" else [
                row for row in selected if str(row[group_field]) == group
            ]
            scores = np.asarray([float(row[score_field]) for row in subset], dtype=np.float64)
            correlations: dict[str, Any] = {}
            for metric_index, metric in enumerate(HEURISTIC_ERROR_METRICS):
                error_values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
                rho = spearman_rank_correlation(scores, error_values)
                low, high = _spearman_ci(
                    scores,
                    error_values,
                    repetitions=repetitions,
                    seed=seed + method_index * 100_000 + group_index * 1_000 + metric_index,
                )
                correlations[metric] = {
                    "spearman_rho": rho,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_maps": len(subset),
                }
            group_results[group] = correlations
        output[method] = group_results
    return {
        "statistical_unit": "map",
        "score_field": score_field,
        "correlations": output,
    }


def evaluate_heuristic_reliability(
    samples: Sequence[DatasetSample],
    prediction_maps: Sequence[np.ndarray],
    scores: Sequence[Mapping[str, float | str]],
    *,
    methods: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Calculate per-map error rows and pooled/distribution-specific Spearman."""

    if not (len(samples) == len(prediction_maps) == len(scores)):
        raise ValueError("Samples, prediction maps, and OOD score records must align by map.")
    per_map: list[dict[str, Any]] = []
    for sample_index, (sample, sample_scores) in enumerate(zip(samples, scores)):
        base = {
            "sample_index": sample_index,
            "map_id": f"{sample.grid_map.distribution}:{sample.grid_map.seed}",
            "distribution": sample.grid_map.distribution,
            "map_seed": int(sample.grid_map.seed),
            "mahalanobis_distance": float(sample_scores["mahalanobis_distance"]),
            "smooth_ood_score": float(sample_scores["smooth_ood_score"]),
        }
        for method_index, method in enumerate(methods):
            metric_values = heuristic_error_metrics(
                sample, prediction_maps[method_index][sample_index]
            )
            per_map.append(
                {
                    **base,
                    "heuristic_method": method,
                    **metric_values,
                }
            )
    return {
        "per_map": per_map,
        "correlations": analyze_score_heuristic_reliability(
            per_map,
            repetitions=repetitions,
            seed=seed,
        ),
    }
