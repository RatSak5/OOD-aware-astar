"""Map-level OOD metrics and separate heuristic-error association analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data.dataset import DatasetSample, map_fingerprint
from heuristics.inference import predict_heuristic_map
from ood.mahalanobis import MahalanobisDetector, _embed_samples

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        description = kwargs.get("desc", "Evaluating")
        total = len(iterable)

        def _iterate() -> Any:
            for index, item in enumerate(iterable, start=1):
                yield item
                print(f"\r{description}: {index}/{total}", end="", flush=True)
            if total:
                print()

        return _iterate()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including exact ties."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Calculate AUROC using the Mann-Whitney rank statistic (OOD positive)."""

    id_values = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_values = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if not len(id_values) or not len(ood_values):
        return float("nan")
    combined = np.concatenate((id_values, ood_values))
    ranks = _rankdata(combined)
    n_id, n_ood = len(id_values), len(ood_values)
    return float((ranks[n_id:].sum() - n_ood * (n_ood + 1) / 2) / (n_id * n_ood))


def aupr_trapezoid(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Calculate trapezoidal precision-recall area with OOD as positive."""

    id_values = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_values = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if not len(id_values) or not len(ood_values):
        return float("nan")
    scores = np.concatenate((id_values, ood_values))
    labels = np.concatenate((np.zeros(len(id_values), dtype=np.int8), np.ones(len(ood_values), dtype=np.int8)))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    recall = [0.0]
    precision = [1.0]
    index = 0
    positive_count = len(ood_values)
    while index < len(order):
        end = index + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group = sorted_labels[index:end]
        true_positive += int(group.sum())
        false_positive += len(group) - int(group.sum())
        recall.append(true_positive / positive_count)
        precision.append(true_positive / max(true_positive + false_positive, 1))
        index = end
    return float(np.trapz(np.asarray(precision), np.asarray(recall)))


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _correlation(first: np.ndarray, second: np.ndarray, method: str) -> float:
    if method == "pearson":
        return _pearson(first, second)
    if method == "spearman":
        return _pearson(_rankdata(first), _rankdata(second))
    raise ValueError("Correlation method must be pearson or spearman.")


def _bootstrap_ci(
    metric: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    *,
    seed: int,
    repetitions: int,
    paired: bool = False,
) -> tuple[float, float]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive.")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    if paired:
        count = len(arrays[0])
        if count < 2 or any(len(array) != count for array in arrays):
            return float("nan"), float("nan")
        for _ in range(repetitions):
            indices = rng.integers(0, count, size=count)
            estimate = metric(*(array[indices] for array in arrays))
            if np.isfinite(estimate):
                estimates.append(float(estimate))
    else:
        if any(len(array) == 0 for array in arrays):
            return float("nan"), float("nan")
        for _ in range(repetitions):
            sampled = tuple(
                array[rng.integers(0, len(array), size=len(array))] for array in arrays
            )
            estimate = metric(*sampled)
            if np.isfinite(estimate):
                estimates.append(float(estimate))
    if not estimates:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(estimates), (0.025, 0.975))
    return float(low), float(high)


def _validate_evaluation_splits(
    detector: MahalanobisDetector,
    id_samples: Sequence[DatasetSample],
    ood_samples: Sequence[DatasetSample],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if detector.d95 is None:
        raise ValueError("Calibrate D95 before running evaluation.")
    if not id_samples or not ood_samples:
        raise ValueError("Both validation ID and OOD evaluation samples are required.")
    id_fingerprints: list[str] = []
    ood_fingerprints: list[str] = []
    for label, samples, allowed, dest in (
        ("validation ID", id_samples, {"D1", "D2", "D3"}, id_fingerprints),
        ("OOD test", ood_samples, {"D4", "D5", "D6"}, ood_fingerprints),
    ):
        local: set[str] = set()
        for index, sample in enumerate(samples):
            if sample.grid_map.distribution not in allowed:
                raise ValueError(f"{label}[{index}] has an impermissible distribution.")
            fingerprint = map_fingerprint(sample.grid_map.grid)
            if fingerprint in local:
                raise ValueError(f"Duplicate map in {label} at index {index}.")
            local.add(fingerprint)
            dest.append(fingerprint)
        if {sample.grid_map.distribution for sample in samples} != allowed:
            raise ValueError(f"{label} must include every distribution {sorted(allowed)}.")
    if set(id_fingerprints) & set(detector.training_fingerprints):
        raise ValueError("Validation ID maps overlap the detector training maps.")
    if set(ood_fingerprints) & (
        set(detector.training_fingerprints) | set(detector.calibration_fingerprints)
    ):
        raise ValueError("OOD test maps overlap training or calibration maps.")
    if detector.calibration_fingerprints and set(id_fingerprints) != set(
        detector.calibration_fingerprints
    ):
        raise ValueError("ID evaluation must use the held-out maps used to calibrate D95.")
    if set(id_fingerprints) & set(ood_fingerprints):
        raise ValueError("ID validation and OOD test maps overlap.")
    return tuple(id_fingerprints), tuple(ood_fingerprints)


def _score_summary(
    distribution: str,
    role: str,
    distances: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    distance_ci = _bootstrap_ci(
        lambda values: float(np.mean(values)),
        (distances,),
        seed=seed,
        repetitions=bootstrap_repetitions,
    )
    score_ci = _bootstrap_ci(
        lambda values: float(np.mean(values)),
        (scores,),
        seed=seed + 1,
        repetitions=bootstrap_repetitions,
    )
    return {
        "distribution": distribution,
        "split_role": role,
        "num_maps": int(len(scores)),
        "distance_mean": float(np.mean(distances)),
        "distance_mean_ci95_low": distance_ci[0],
        "distance_mean_ci95_high": distance_ci[1],
        "distance_median": float(np.median(distances)),
        "distance_std": float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
        "distance_p05": float(np.quantile(distances, 0.05)),
        "distance_p95": float(np.quantile(distances, 0.95)),
        "score_mean": float(np.mean(scores)),
        "score_mean_ci95_low": score_ci[0],
        "score_mean_ci95_high": score_ci[1],
        "score_median": float(np.median(scores)),
        "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "score_p05": float(np.quantile(scores, 0.05)),
        "score_p95": float(np.quantile(scores, 0.95)),
    }


def _binary_row(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    comparison: str,
    id_distances: np.ndarray,
    ood_distances: np.ndarray,
    d95: float,
    threshold: float,
    seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    fpr = float(np.mean(id_scores >= threshold))
    fnr = float(np.mean(ood_scores < threshold))
    gap = float(np.mean(ood_scores) - np.mean(id_scores))
    distance_gap = float(np.mean(ood_distances) - np.mean(id_distances))
    auc = auroc(id_scores, ood_scores)
    aupr = aupr_trapezoid(id_scores, ood_scores)
    auc_ci = _bootstrap_ci(
        auroc, (id_scores, ood_scores), seed=seed, repetitions=bootstrap_repetitions
    )
    aupr_ci = _bootstrap_ci(
        aupr_trapezoid,
        (id_scores, ood_scores),
        seed=seed + 1,
        repetitions=bootstrap_repetitions,
    )
    fpr_ci = _bootstrap_ci(
        lambda id_s, ood_s: float(np.mean(id_s >= threshold)),
        (id_scores, ood_scores), seed=seed + 2, repetitions=bootstrap_repetitions,
    )
    fnr_ci = _bootstrap_ci(
        lambda id_s, ood_s: float(np.mean(ood_s < threshold)),
        (id_scores, ood_scores), seed=seed + 3, repetitions=bootstrap_repetitions,
    )
    gap_ci = _bootstrap_ci(
        lambda id_s, ood_s: float(np.mean(ood_s) - np.mean(id_s)),
        (id_scores, ood_scores), seed=seed + 4, repetitions=bootstrap_repetitions,
    )
    distance_gap_ci = _bootstrap_ci(
        lambda id_d, ood_d: float(np.mean(ood_d) - np.mean(id_d)),
        (id_distances, ood_distances),
        seed=seed + 5,
        repetitions=bootstrap_repetitions,
    )
    return {
        "comparison": comparison,
        "positive_class": "OOD",
        "operating_score_threshold": threshold,
        "equivalent_distance_threshold": (
            "infinite" if threshold >= 1.0 else float(-d95 * np.log1p(-threshold))
        ),
        "num_id_maps": int(len(id_scores)),
        "num_ood_maps": int(len(ood_scores)),
        "auroc": auc,
        "auroc_ci95_low": auc_ci[0],
        "auroc_ci95_high": auc_ci[1],
        "aupr_trapezoid": aupr,
        "aupr_ci95_low": aupr_ci[0],
        "aupr_ci95_high": aupr_ci[1],
        "fpr": fpr,
        "fpr_ci95_low": fpr_ci[0],
        "fpr_ci95_high": fpr_ci[1],
        "fnr": fnr,
        "fnr_ci95_low": fnr_ci[0],
        "fnr_ci95_high": fnr_ci[1],
        "mean_score_gap_ood_minus_id": gap,
        "score_gap_ci95_low": gap_ci[0],
        "score_gap_ci95_high": gap_ci[1],
        "mean_distance_gap_ood_minus_id": distance_gap,
        "distance_gap_ci95_low": distance_gap_ci[0],
        "distance_gap_ci95_high": distance_gap_ci[1],
    }


def _correlation_rows(
    records: Sequence[dict[str, Any]],
    *,
    method: str,
    seed: int,
    bootstrap_repetitions: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[record["distribution"]].append(record)
    output: list[dict[str, Any]] = []
    for distribution, group in sorted(groups.items()):
        errors = np.asarray([row["heuristic_mae"] for row in group], dtype=np.float64)
        for x_name in ("distance", "score"):
            values = np.asarray([row[f"ood_{x_name}"] for row in group], dtype=np.float64)
            for correlation_method in ("pearson", "spearman"):
                function = lambda x, y, _method=correlation_method: _correlation(x, y, _method)
                value = function(values, errors)
                low, high = _bootstrap_ci(
                    function,
                    (values, errors),
                    seed=seed + len(output),
                    repetitions=bootstrap_repetitions,
                    paired=True,
                )
                output.append(
                    {
                        "heuristic_method": method,
                        "distribution": distribution,
                        "ood_measure": x_name,
                        "correlation": correlation_method,
                        "num_maps": int(len(group)),
                        "value": value,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return output


def evaluate_ood_detector(
    detector: MahalanobisDetector,
    extractor: Any,
    validation_id_samples: Sequence[DatasetSample],
    ood_test_samples: Sequence[DatasetSample],
    *,
    heuristic_models: Mapping[str, tuple[Any, str]] | None = None,
    device: str = "cpu",
    score_threshold: float | None = None,
    bootstrap_repetitions: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate continuous scores, supplementary binary metrics, and h-error.

    ``heuristic_models`` maps a method label to ``(model, method_code)`` where
    method_code is the Prompt 4 H1/H2/H3 decoder. Heuristic errors are computed
    from raw predictions on reachable free cells and are analyzed separately.
    """

    if not np.isfinite(score_threshold if score_threshold is not None else 0.0):
        raise ValueError("score_threshold must be finite.")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive.")
    _validate_evaluation_splits(detector, validation_id_samples, ood_test_samples)
    threshold = (
        float(-np.expm1(-1.0)) if score_threshold is None else float(score_threshold)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1].")

    id_embeddings = _embed_samples(validation_id_samples, extractor, device=device)
    ood_embeddings = _embed_samples(ood_test_samples, extractor, device=device)
    id_distances = detector.distance(id_embeddings)
    ood_distances = detector.distance(ood_embeddings)
    id_scores = detector.score(id_embeddings)
    ood_scores = detector.score(ood_embeddings)
    records: list[dict[str, Any]] = []
    for role, samples, distances, scores in (
        ("validation_id", validation_id_samples, id_distances, id_scores),
        ("ood_test", ood_test_samples, ood_distances, ood_scores),
    ):
        for index, (sample, distance, score) in enumerate(zip(samples, distances, scores)):
            records.append(
                {
                    "sample_index": index,
                    "distribution": sample.grid_map.distribution,
                    "split_role": role,
                    "seed": int(sample.grid_map.seed),
                    "mahalanobis_distance": float(distance),
                    "normalized_score": float(score),
                }
            )

    score_summaries: list[dict[str, Any]] = []
    for summary_index, distribution in enumerate(("D1", "D2", "D3", "D4", "D5", "D6")):
        selected = [
            record for record in records if record["distribution"] == distribution
        ]
        if selected:
            score_summaries.append(
                _score_summary(
                    distribution,
                    selected[0]["split_role"],
                    np.asarray([row["mahalanobis_distance"] for row in selected]),
                    np.asarray([row["normalized_score"] for row in selected]),
                    seed=seed + summary_index * 10,
                    bootstrap_repetitions=bootstrap_repetitions,
                )
            )

    id_distribution_scores = np.asarray(
        [row["normalized_score"] for row in records if row["split_role"] == "validation_id"]
    )
    id_distribution_distances = np.asarray(
        [row["mahalanobis_distance"] for row in records if row["split_role"] == "validation_id"]
    )
    binary_rows: list[dict[str, Any]] = [
        _binary_row(
            id_distribution_scores,
            ood_scores,
            comparison="validation D1-D3 vs pooled D4-D6",
            id_distances=id_distribution_distances,
            ood_distances=ood_distances,
            d95=float(detector.d95),
            threshold=threshold,
            seed=seed,
            bootstrap_repetitions=bootstrap_repetitions,
        )
    ]
    for distribution in ("D4", "D5", "D6"):
        current = np.asarray(
            [row["normalized_score"] for row in records if row["distribution"] == distribution]
        )
        current_distances = np.asarray(
            [row["mahalanobis_distance"] for row in records if row["distribution"] == distribution]
        )
        if len(current):
            binary_rows.append(
                _binary_row(
                    id_distribution_scores,
                    current,
                    comparison=f"validation D1-D3 vs {distribution}",
                    id_distances=id_distribution_distances,
                    ood_distances=current_distances,
                    d95=float(detector.d95),
                    threshold=threshold,
                    seed=seed + len(binary_rows) * 10,
                    bootstrap_repetitions=bootstrap_repetitions,
                )
            )

    correlation_rows: list[dict[str, Any]] = []
    heuristic_error_rows: list[dict[str, Any]] = []
    for model_index, (method_label, model_info) in enumerate((heuristic_models or {}).items()):
        heuristic_model, method_code = model_info
        model_records: list[dict[str, Any]] = []
        for index, sample in enumerate(
            tqdm(
                [*validation_id_samples, *ood_test_samples],
                desc=f"Heuristic errors {method_label}",
                leave=False,
            )
        ):
            prediction = predict_heuristic_map(
                heuristic_model,
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                method=method_code,
                device=device,
            )
            valid = sample.ground_truth.reachable_mask & (sample.grid_map.grid == 0)
            target = sample.ground_truth.distance_map
            error = float(np.mean(np.abs(prediction[valid] - target[valid])))
            record = records[index]
            model_record = {
                "distribution": sample.grid_map.distribution,
                "heuristic_mae": error,
                "ood_distance": record["mahalanobis_distance"],
                "ood_score": record["normalized_score"],
            }
            model_records.append(model_record)
            heuristic_error_rows.append(
                {
                    "heuristic_method": method_label,
                    "sample_index": index,
                    **model_record,
                }
            )
        correlation_records = [
            {
                "distribution": row["distribution"],
                "heuristic_mae": row["heuristic_mae"],
                "ood_distance": row["ood_distance"],
                "ood_score": row["ood_score"],
            }
            for row in model_records
        ]
        correlation_rows.extend(
            _correlation_rows(
                correlation_records,
                method=method_label,
                seed=seed + model_index * 100,
                bootstrap_repetitions=bootstrap_repetitions,
            )
        )

    return {
        "d95": float(detector.d95),
        "score_threshold": threshold,
        "score_threshold_definition": (
            "default s >= 1-exp(-1), equivalent to D_M >= D95"
            if score_threshold is None
            else f"user-selected s >= {threshold:.12g}"
        ),
        "scores": records,
        "score_summaries_by_distribution": score_summaries,
        "binary_metrics": binary_rows,
        "heuristic_errors": heuristic_error_rows,
        "heuristic_error_correlations": correlation_rows,
        "metric_definitions": {
            "aupr": "trapezoidal area under precision-recall curve; OOD positive",
            "confidence_intervals": f"95% percentile bootstrap with {bootstrap_repetitions} map-level resamples; ID and OOD separately resampled for binary metrics",
            "heuristic_error": "map-wise mean absolute raw heuristic prediction error on reachable traversable cells",
        },
    }


def save_evaluation_csvs(results: Mapping[str, Any], output_directory: str | Path) -> dict[str, Path]:
    """Write continuous scores, summaries, binary metrics, and correlations to CSV."""

    import csv

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "ood_scores.csv": results.get("scores", []),
        "ood_score_summaries.csv": results.get("score_summaries_by_distribution", []),
        "ood_binary_metrics.csv": results.get("binary_metrics", []),
        "heuristic_error_by_map.csv": results.get("heuristic_errors", []),
        "ood_heuristic_correlations.csv": results.get("heuristic_error_correlations", []),
    }
    written: dict[str, Path] = {}
    for filename, rows in tables.items():
        if not rows:
            continue
        path = root / filename
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        written[filename] = path
    return written
