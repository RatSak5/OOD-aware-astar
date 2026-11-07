"""Dependency-light, map-level statistical procedures for benchmark results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def average_seeds_by_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    method: str,
    map_key: str = "map_id",
) -> dict[str, float]:
    """Average repeated runs within map before treating maps as independent."""

    values: dict[str, list[float]] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        value = row.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.setdefault(str(row[map_key]), []).append(numeric)
    return {key: float(np.mean(items)) for key, items in values.items()}


def paired_map_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    method_a: str,
    method_b: str,
    map_key: str = "map_id",
) -> tuple[list[str], np.ndarray]:
    """Return matched per-map mean differences ``A - B`` after seed averaging."""

    a = average_seeds_by_map(rows, metric=metric, method=method_a, map_key=map_key)
    b = average_seeds_by_map(rows, metric=metric, method=method_b, map_key=map_key)
    common = sorted(set(a) & set(b))
    differences = np.asarray([a[key] - b[key] for key in common], dtype=np.float64)
    return common, differences


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval over independent map-level observations."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    if repetitions < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("repetitions must be positive and confidence in (0, 1).")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def paired_sign_flip_test(
    differences: Sequence[float] | np.ndarray,
    *,
    repetitions: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-flip test of a zero mean map-level difference.

    The map is the independent unit. Repeated sampling seeds must already have
    been averaged per map. The plus-one Monte Carlo correction prevents a zero
    p-value when a finite number of sign patterns is sampled.
    """

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    batch_size = min(2_000, repetitions)
    completed = 0
    while completed < repetitions:
        batch = min(batch_size, repetitions - completed)
        signs = rng.integers(0, 2, size=(batch, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += batch
    return float((exceedances + 1) / (repetitions + 1))


def paired_rank_biserial(differences: Sequence[float] | np.ndarray) -> float:
    """Return signed rank-biserial effect size for paired map differences."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0.0)]
    if not len(values):
        return 0.0
    absolute = np.abs(values)
    order = np.argsort(absolute, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_abs = absolute[order]
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and sorted_abs[end] == sorted_abs[begin]:
            end += 1
        average_rank = ((begin + 1) + end) / 2.0
        ranks[order[begin:end]] = average_rank
        begin = end
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted.tolist()
    sorted_indices = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(sorted_indices)
    for rank, original_index in enumerate(sorted_indices):
        candidate = (count - rank) * float(values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def paired_comparison(
    differences: Sequence[float] | np.ndarray,
    *,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    """Summarize paired effect, confidence interval, and sign-flip inference."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = bootstrap_mean_interval(
        values, repetitions=bootstrap_repetitions, seed=seed
    )
    return {
        "n_maps": int(len(values)),
        "mean_difference": float(values.mean()) if len(values) else float("nan"),
        "median_difference": float(np.median(values)) if len(values) else float("nan"),
        "mean_difference_ci95_low": low,
        "mean_difference_ci95_high": high,
        "paired_rank_biserial": paired_rank_biserial(values),
        "sign_flip_p_value": paired_sign_flip_test(
            values, repetitions=permutation_repetitions, seed=seed + 1
        ),
    }
