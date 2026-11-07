"""Conventional and Neural-guidance SeeA* experiment entry points."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from adaptive.ood_adaptive_seeastar import (
    AdaptiveKPolicy,
    compute_adaptive_k,
    extract_map_embedding,
    run_ood_adaptive_neural_guidance_seeastar,
    run_ood_adaptive_seeastar,
)
from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig
from data.dataset import DatasetSample, map_fingerprint
from heuristics.inference import fixed_goal_heuristic, predict_heuristic_map
from ood.feature_extractor import OODFeatureExtractor
from ood.mahalanobis import MahalanobisDetector, normalized_ood_score
from seeastar.neural_guidance import run_neural_guidance_seeastar
from seeastar.uniform import UniformSeeAStarConfig, uniform_seeastar
from visualization.adaptive_seeastar import plot_adaptive_seeastar_results
from visualization.ood import plot_heuristic_error_scatter

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        del kwargs
        return iterable


@dataclass(frozen=True, slots=True)
class AdaptiveSeeAStarExperimentConfig:
    """Predeclared conventional unit-cost adaptive SeeA* K sweep."""

    k_bounds: tuple[tuple[int, int], ...] = ((4, 8), (4, 16), (8, 32))
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.k_bounds:
            raise ValueError("At least one (K_min, K_max) pair is required.")
        for k_min, k_max in self.k_bounds:
            if (
                isinstance(k_min, bool)
                or isinstance(k_max, bool)
                or not isinstance(k_min, int)
                or not isinstance(k_max, int)
                or k_min < 1
                or k_max < k_min
            ):
                raise ValueError("Each K bound must satisfy integer 1 <= K_min <= K_max.")
        if len(set(self.k_bounds)) != len(self.k_bounds):
            raise ValueError("K-bound combinations must be unique.")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer.")


@dataclass(frozen=True, slots=True)
class NeuralGuidanceSeeAStarExperimentConfig:
    """Settings for the corrected, Phi-based fixed/adaptive comparison."""

    fixed_ks: tuple[int, ...] = (4, 5, 10, 16)
    k_min: int = 4
    k_max: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.k_min, bool)
            or isinstance(self.k_max, bool)
            or not isinstance(self.k_min, int)
            or not isinstance(self.k_max, int)
            or not 1 <= self.k_min <= self.k_max
        ):
            raise ValueError("K bounds must be integers satisfying 1 <= Kmin <= Kmax.")
        if not self.fixed_ks or tuple(sorted(set(self.fixed_ks))) != self.fixed_ks:
            raise ValueError("fixed_ks must be non-empty, sorted, and unique.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not self.k_min <= value <= self.k_max
            for value in self.fixed_ks
        ):
            raise ValueError("Every fixed K must be an integer within the configured bounds.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer.")


def _validate_samples(
    samples: Sequence[DatasetSample],
    detector: MahalanobisDetector,
    cost_config: GroundTruthCostConfig,
) -> None:
    expected = {"D1", "D2", "D3", "D4", "D5", "D6"}
    if not samples or {sample.grid_map.distribution for sample in samples} != expected:
        raise ValueError("Evaluation maps must include D1-D6.")
    fingerprints: set[str] = set()
    training = set(detector.training_fingerprints)
    calibration = set(detector.calibration_fingerprints)
    for index, sample in enumerate(samples):
        distribution = sample.grid_map.distribution
        if sample.cost_config != cost_config:
            raise ValueError(f"Evaluation map {index} has a different ground-truth cost model.")
        fingerprint = map_fingerprint(sample.grid_map.grid)
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate evaluation map layout at index {index}.")
        fingerprints.add(fingerprint)
        if fingerprint in training:
            raise ValueError(f"Evaluation map {index} overlaps detector training maps.")
        if distribution in {"D4", "D5", "D6"} and fingerprint in calibration:
            raise ValueError(f"OOD test map {index} overlaps D95 calibration maps.")
    if detector.d95 is None:
        raise ValueError("The Mahalanobis detector must be calibrated before this experiment.")
    if (
        detector.num_training_maps < 2
        or not detector.training_fingerprints
        or detector.num_calibration_maps < 1
        or not detector.calibration_fingerprints
    ):
        raise ValueError("Detector must include train-only fit and held-out ID calibration provenance.")
    if detector.d95 <= detector.config.d95_minimum:
        raise ValueError("The detector's D95 is at or below its configured safety minimum.")


def _ratio(found: bool, found_cost: float, optimal_cost: float) -> float:
    if not found or not math.isfinite(optimal_cost):
        return float("nan")
    if optimal_cost == 0.0:
        return 1.0 if found_cost == 0.0 else float("inf")
    return float(found_cost / optimal_cost)


def _exploration_metrics(search: Any, grid: np.ndarray) -> tuple[int, float, float]:
    """Describe expanded-state coverage and spatial spread without assuming benefit."""

    expanded = sorted(set(search.expanded_order))
    free_count = int(np.count_nonzero(grid == 0))
    coverage = len(expanded) / free_count if free_count else 0.0
    if len(expanded) < 2:
        return len(expanded), float(coverage), 0.0
    positions = np.asarray(expanded, dtype=np.float64)
    centroid = positions.mean(axis=0)
    diagonal = max(float(np.hypot(*grid.shape)), 1.0)
    normalized_spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / diagonal)
    return len(expanded), float(coverage), normalized_spread


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["distribution"]),
            int(row["k_min"]),
            int(row["k_max"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method, distribution, k_min, k_max), group in sorted(grouped.items()):
        finite_ratios = [
            float(row["optimality_ratio"])
            for row in group
            if math.isfinite(float(row["optimality_ratio"]))
        ]
        summaries.append(
            {
                "method": method,
                "distribution": distribution,
                "k_min": k_min,
                "k_max": k_max,
                "num_maps": len(group),
                "success_rate": float(np.mean([bool(row["found"]) for row in group])),
                "mean_selected_k": float(np.mean([row["selected_k"] for row in group])),
                "mean_expanded_nodes": float(np.mean([row["expanded_nodes"] for row in group])),
                "mean_expanded_unique_nodes": float(
                    np.mean([row["expanded_unique_nodes"] for row in group])
                ),
                "mean_expanded_grid_coverage": float(
                    np.mean([row["expanded_grid_coverage"] for row in group])
                ),
                "mean_expanded_spatial_spread": float(
                    np.mean([row["expanded_spatial_spread"] for row in group])
                ),
                "mean_generated_nodes": float(np.mean([row["generated_nodes"] for row in group])),
                "mean_reexpansions": float(np.mean([row["reexpansions"] for row in group])),
                "mean_max_open_size": float(np.mean([row["max_open_size"] for row in group])),
                "mean_sampling_overhead_seconds": float(
                    np.mean([row["sampling_overhead_seconds"] for row in group])
                ),
                "mean_search_runtime_seconds": float(
                    np.mean([row["search_runtime_seconds"] for row in group])
                ),
                "mean_total_end_to_end_seconds": float(
                    np.mean([row["total_end_to_end_seconds"] for row in group])
                ),
                "mean_optimality_ratio": (
                    float(np.mean(finite_ratios)) if finite_ratios else float("nan")
                ),
            }
        )
    return summaries


def _paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["run_id"], row["k_min"], row["k_max"], row["method"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    adaptive_methods = (
        "adaptive_high_ood_smaller_k",
        "adaptive_reverse_high_ood_larger_k",
        "adaptive_clipped_high_ood_smaller_k",
    )
    for row in rows:
        adaptive_method = str(row["method"])
        if adaptive_method not in adaptive_methods:
            continue
        for fixed_method in ("fixed_k_min", "fixed_k_mid", "fixed_k_max"):
            fixed = by_key.get(
                (row["run_id"], row["k_min"], row["k_max"], fixed_method)
            )
            if fixed is None:
                continue
            paired.append(
                {
                    "run_id": row["run_id"],
                    "distribution": row["distribution"],
                    "map_seed": row["map_seed"],
                    "k_min": row["k_min"],
                    "k_max": row["k_max"],
                    "adaptive_method": adaptive_method,
                    "fixed_reference": fixed_method,
                    "adaptive_k": row["selected_k"],
                    "fixed_k": fixed["selected_k"],
                    "delta_expanded_nodes": row["expanded_nodes"] - fixed["expanded_nodes"],
                    "delta_expanded_grid_coverage": (
                        row["expanded_grid_coverage"] - fixed["expanded_grid_coverage"]
                    ),
                    "delta_expanded_spatial_spread": (
                        row["expanded_spatial_spread"] - fixed["expanded_spatial_spread"]
                    ),
                    "delta_sampling_overhead_seconds": (
                        row["sampling_overhead_seconds"] - fixed["sampling_overhead_seconds"]
                    ),
                    "delta_search_runtime_seconds": (
                        row["search_runtime_seconds"] - fixed["search_runtime_seconds"]
                    ),
                    "delta_total_end_to_end_seconds": (
                        row["total_end_to_end_seconds"] - fixed["total_end_to_end_seconds"]
                    ),
                    "delta_optimality_ratio": (
                        row["optimality_ratio"] - fixed["optimality_ratio"]
                        if math.isfinite(row["optimality_ratio"])
                        and math.isfinite(fixed["optimality_ratio"])
                        else float("nan")
                    ),
                }
            )
    return paired


def _paired_summaries(paired_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int, int], list[Mapping[str, Any]]] = {}
    for row in paired_rows:
        key = (
            str(row["adaptive_method"]),
            str(row["fixed_reference"]),
            str(row["distribution"]),
            int(row["k_min"]),
            int(row["k_max"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method, fixed_reference, distribution, k_min, k_max), group in sorted(groups.items()):
        finite_ratios = [
            float(row["delta_optimality_ratio"])
            for row in group
            if math.isfinite(float(row["delta_optimality_ratio"]))
        ]
        summaries.append(
            {
                "adaptive_method": method,
                "fixed_reference": fixed_reference,
                "distribution": distribution,
                "k_min": k_min,
                "k_max": k_max,
                "num_maps": len(group),
                "mean_delta_expanded_nodes_vs_fixed_reference": float(
                    np.mean([row["delta_expanded_nodes"] for row in group])
                ),
                "maps_with_more_expansions_than_fixed_reference": int(
                    sum(row["delta_expanded_nodes"] > 0 for row in group)
                ),
                "mean_delta_total_runtime_vs_fixed_reference_seconds": float(
                    np.mean([row["delta_total_end_to_end_seconds"] for row in group])
                ),
                "maps_with_slower_total_runtime_than_fixed_reference": int(
                    sum(row["delta_total_end_to_end_seconds"] > 0 for row in group)
                ),
                "mean_delta_optimality_ratio_vs_fixed_reference": (
                    float(np.mean(finite_ratios)) if finite_ratios else float("nan")
                ),
                "maps_with_worse_optimality_ratio_than_fixed_reference": int(
                    sum(
                        math.isfinite(row["delta_optimality_ratio"])
                        and row["delta_optimality_ratio"] > 0
                        for row in group
                    )
                ),
                "mean_delta_expanded_grid_coverage_vs_fixed_reference": float(
                    np.mean([row["delta_expanded_grid_coverage"] for row in group])
                ),
                "mean_delta_expanded_spatial_spread_vs_fixed_reference": float(
                    np.mean([row["delta_expanded_spatial_spread"] for row in group])
                ),
            }
        )
    return summaries


def run_adaptive_seeastar_experiment(
    evaluation_samples: Sequence[DatasetSample],
    feature_extractor: OODFeatureExtractor,
    detector: MahalanobisDetector,
    heuristic_model: Any,
    heuristic_method: str,
    output_directory: str | Path,
    *,
    config: AdaptiveSeeAStarExperimentConfig = AdaptiveSeeAStarExperimentConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    heuristic_label: str | None = None,
    feature_checkpoint: str | Path | None = None,
    heuristic_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run the predeclared uniform-only K sweep and save metrics and plots.

    Evaluation samples must contain independent D1-D3 ID maps and D4-D6 OOD
    maps. The detector and D95 are provided already fitted; this experiment
    never uses OOD maps to fit features, covariance, D95, or choose a policy.
    The learned heuristic map is predicted once per map and reused by every
    paired search so all K policies see identical guidance values.
    """

    _validate_samples(evaluation_samples, detector, cost_config)
    if heuristic_method not in {"H1", "H2", "H3"}:
        raise ValueError("heuristic_method must be H1, H2, or H3.")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sampling_trace: list[dict[str, Any]] = []
    heuristic_errors: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(
        tqdm(evaluation_samples, desc="OOD-adaptive uniform SeeA*", leave=False)
    ):
        grid = sample.grid_map.grid
        start, goal = sample.grid_map.start, sample.grid_map.goal
        run_id = f"{sample.grid_map.distribution}:{sample.grid_map.seed}:{sample_index}"

        # OOD score diagnostics are saved for fixed-K rows, but those policies
        # do not pay detector inference overhead in their end-to-end runtimes.
        diagnostic_cnn_started = perf_counter()
        diagnostic_embedding = extract_map_embedding(
            feature_extractor, grid, start, goal, device=config.device
        )
        diagnostic_cnn_seconds = perf_counter() - diagnostic_cnn_started
        diagnostic_ood_started = perf_counter()
        diagnostic_distance = float(detector.distance(diagnostic_embedding)[0])
        diagnostic_score = float(normalized_ood_score(diagnostic_distance, float(detector.d95)))
        diagnostic_ood_seconds = perf_counter() - diagnostic_ood_started

        heuristic_started = perf_counter()
        heuristic_map = predict_heuristic_map(
            heuristic_model,
            grid,
            start,
            goal,
            method=heuristic_method,
            device=config.device,
        )
        heuristic_seconds = perf_counter() - heuristic_started
        valid_cells = sample.ground_truth.reachable_mask & (grid == 0)
        target = sample.ground_truth.distance_map
        h_error = float(np.mean(np.abs(heuristic_map[valid_cells] - target[valid_cells])))
        h_callable = fixed_goal_heuristic(heuristic_map, goal)
        heuristic_errors.append(
            {
                "heuristic_method": heuristic_label or heuristic_method,
                "sample_index": sample_index,
                "run_id": run_id,
                "distribution": sample.grid_map.distribution,
                "heuristic_mae": h_error,
                "ood_distance": diagnostic_distance,
                "ood_score": diagnostic_score,
            }
        )

        seed = (config.seed + int(sample.grid_map.seed % (2**31 - 1))) % (2**31 - 1)
        for bound_index, (k_min, k_max) in enumerate(config.k_bounds):
            configuration_id = f"bounds_{k_min}_{k_max}_{bound_index}"
            fixed_ks = (
                ("fixed_k_min", k_min),
                ("fixed_k_mid", k_min + (k_max - k_min) // 2),
                ("fixed_k_max", k_max),
            )
            runs: list[tuple[str, int, str, str, Any]] = []
            for method, fixed_k in fixed_ks:
                runs.append((method, fixed_k, "fixed", "none", None))
            runs.extend(
                (
                    (
                        "adaptive_high_ood_smaller_k",
                        k_max,
                        "higher_ood_smaller_k",
                        "smooth",
                        AdaptiveKPolicy(k_min, k_max, "higher_ood_smaller_k", "smooth"),
                    ),
                    (
                        "adaptive_reverse_high_ood_larger_k",
                        k_min,
                        "higher_ood_larger_k",
                        "smooth",
                        AdaptiveKPolicy(k_min, k_max, "higher_ood_larger_k", "smooth"),
                    ),
                    (
                        "adaptive_clipped_high_ood_smaller_k",
                        k_max,
                        "higher_ood_smaller_k",
                        "clipped",
                        AdaptiveKPolicy(k_min, k_max, "higher_ood_smaller_k", "clipped"),
                    ),
                )
            )

            for method_index, (method, nominal_k, direction, normalization, policy) in enumerate(runs):
                if policy is None:
                    decision = None
                    search = uniform_seeastar(
                        grid,
                        start,
                        goal,
                        heuristic=h_callable,
                        config=UniformSeeAStarConfig(k=nominal_k, seed=seed),
                        cost_config=cost_config,
                    )
                    selected_k = nominal_k
                    policy_score = diagnostic_score
                    cnn_used = 0.0
                    ood_used = 0.0
                else:
                    adaptive_result = run_ood_adaptive_seeastar(
                        grid,
                        start,
                        goal,
                        feature_extractor,
                        detector,
                        policy=policy,
                        heuristic=h_callable,
                        seed=seed,
                        heuristic_inference_seconds=heuristic_seconds,
                        cost_config=cost_config,
                        device=config.device,
                    )
                    decision = adaptive_result.decision
                    search = adaptive_result.search
                    selected_k = decision.selected_k
                    policy_score = decision.policy_score
                    cnn_used = adaptive_result.cnn_inference_seconds
                    ood_used = adaptive_result.ood_inference_seconds

                unique_expansions, expansion_coverage, expansion_spread = (
                    _exploration_metrics(search, grid)
                )

                row = {
                    "run_id": run_id,
                    "configuration_id": configuration_id,
                    "method": method,
                    "distribution": sample.grid_map.distribution,
                    "map_seed": int(sample.grid_map.seed),
                    "start": json.dumps(start),
                    "goal": json.dumps(goal),
                    "heuristic_method": heuristic_label or heuristic_method,
                    "sampling_strategy": "uniform",
                    "ood_direction": direction,
                    "normalization": normalization,
                    "k_min": k_min,
                    "k_max": k_max,
                    "selected_k": selected_k,
                    "mahalanobis_distance": (
                        decision.mahalanobis_distance if decision is not None else diagnostic_distance
                    ),
                    "smooth_ood_score": (
                        decision.smooth_ood_score if decision is not None else diagnostic_score
                    ),
                    "policy_ood_score": policy_score,
                    "heuristic_mae": h_error,
                    "found": bool(search.found),
                    "found_path": json.dumps(search.path),
                    "found_ground_truth_cost": float(search.found_ground_truth_cost),
                    "optimal_ground_truth_cost": float(sample.ground_truth.optimal_path_cost),
                    "optimality_ratio": _ratio(
                        search.found,
                        search.found_ground_truth_cost,
                        sample.ground_truth.optimal_path_cost,
                    ),
                    "expanded_nodes": search.expanded_nodes,
                    "expanded_unique_nodes": unique_expansions,
                    "expanded_grid_coverage": expansion_coverage,
                    "expanded_spatial_spread": expansion_spread,
                    "generated_nodes": search.generated_nodes,
                    "reexpansions": search.reexpansions,
                    "max_open_size": search.max_open_size,
                    "final_open_size": search.final_open_size,
                    "closed_size": search.closed_size,
                    "sampling_operations": len(search.candidate_sizes),
                    "eligible_open_candidates_total": int(sum(search.eligible_open_sizes)),
                    "eligible_open_candidates_mean": (
                        float(np.mean(search.eligible_open_sizes)) if search.eligible_open_sizes else 0.0
                    ),
                    "eligible_open_candidates_max": (
                        max(search.eligible_open_sizes, default=0)
                    ),
                    "candidates_sampled_total": int(sum(search.candidate_sizes)),
                    "candidates_sampled_mean": (
                        float(np.mean(search.candidate_sizes)) if search.candidate_sizes else 0.0
                    ),
                    "candidates_sampled_max": max(search.candidate_sizes, default=0),
                    "search_runtime_seconds": search.search_time_seconds,
                    "sampling_overhead_seconds": search.sampling_time_seconds,
                    "heuristic_inference_seconds": heuristic_seconds,
                    # Fixed-K methods do not invoke the detector in their runtime.
                    "cnn_inference_seconds": cnn_used,
                    "ood_inference_seconds": ood_used,
                    "score_diagnostic_cnn_seconds": diagnostic_cnn_seconds,
                    "score_diagnostic_ood_seconds": diagnostic_ood_seconds,
                    "total_end_to_end_seconds": (
                        heuristic_seconds + cnn_used + ood_used + search.search_time_seconds
                    ),
                    "stop_reason": search.stop_reason,
                    "candidate_sizes": json.dumps(search.candidate_sizes),
                    "eligible_open_sizes": json.dumps(search.eligible_open_sizes),
                }
                rows.append(row)

                for iteration, (eligible, sampled) in enumerate(
                    zip(search.eligible_open_sizes, search.candidate_sizes)
                ):
                    sampling_trace.append(
                        {
                            "run_id": run_id,
                            "configuration_id": configuration_id,
                            "method": method,
                            "distribution": sample.grid_map.distribution,
                            "map_seed": int(sample.grid_map.seed),
                            "iteration": iteration,
                            "selected_k": selected_k,
                            "eligible_open_candidates": eligible,
                            "candidates_sampled": sampled,
                        }
                    )

    summaries = _summarize(rows)
    paired = _paired_comparisons(rows)
    paired_summaries = _paired_summaries(paired)
    _write_csv(root / "adaptive_seeastar_per_map.csv", rows)
    _write_csv(root / "adaptive_seeastar_sampling_trace.csv", sampling_trace)
    _write_csv(root / "adaptive_seeastar_summary.csv", summaries)
    _write_csv(root / "adaptive_seeastar_paired_comparisons.csv", paired)
    _write_csv(root / "adaptive_seeastar_paired_summary.csv", paired_summaries)
    _write_csv(root / "adaptive_seeastar_heuristic_error.csv", heuristic_errors)
    config_payload = {
        "schema_version": 1,
        "method": "OOD-adaptive SeeA* with uniform sampling",
        "ood_score_formula": "1 - exp(-D_M / D95)",
        "proposed_k_formula": "K_max - floor(s * (K_max - K_min))",
        "direction_ablation": "K_min + floor(s * (K_max - K_min))",
        "normalization_ablation": "min(D_M / D95, 1)",
        "k_bounds": [list(pair) for pair in config.k_bounds],
        "seed": config.seed,
        "device": config.device,
        "d95": float(detector.d95),
        "detector_training_maps": detector.num_training_maps,
        "detector_calibration_maps": detector.num_calibration_maps,
        "detector_training_fingerprints": list(detector.training_fingerprints),
        "detector_calibration_fingerprints": list(detector.calibration_fingerprints),
        "feature_checkpoint": str(feature_checkpoint) if feature_checkpoint else None,
        "heuristic_checkpoint": str(heuristic_checkpoint) if heuristic_checkpoint else None,
        "heuristic_method": heuristic_label or heuristic_method,
        "ground_truth_cost_config": asdict(cost_config),
        "sampling_strategy": "uniform without replacement",
        "k_is_calculated_once_per_map": True,
        "ood_test_used_for_model_or_policy_selection": False,
        "fixed_k_runtime_excludes_unused_ood_inference": True,
        "sample_count": len(evaluation_samples),
        "distribution_counts": {
            distribution: sum(
                sample.grid_map.distribution == distribution
                for sample in evaluation_samples
            )
            for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")
        },
    }
    (root / "adaptive_seeastar_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    plots = plot_adaptive_seeastar_results(rows, paired, root / "plots")
    plots.update(plot_heuristic_error_scatter(heuristic_errors, root / "plots"))
    return {
        "per_map": rows,
        "sampling_trace": sampling_trace,
        "summary": summaries,
        "paired_comparisons": paired,
        "paired_summaries": paired_summaries,
        "heuristic_errors": heuristic_errors,
        "configuration": config_payload,
        "plots": plots,
        "output_directory": root,
    }


def run_neural_guidance_seeastar_experiment(
    evaluation_samples: Sequence[DatasetSample],
    phi_maps: Sequence[np.ndarray],
    mahalanobis_distances: Sequence[float],
    d95: float,
    output_directory: str | Path,
    *,
    config: NeuralGuidanceSeeAStarExperimentConfig = NeuralGuidanceSeeAStarExperimentConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> dict[str, Any]:
    """Compare corrected fixed-K and adaptive-K searches on supplied Phi maps.

    ``phi_maps`` must contain the Neural A* guidance output aligned with each
    sample. The independent detector distances and calibrated ID ``d95`` are
    provided as values; this function performs no model inference, fitting,
    or calibration. Every arm for a map receives the same unmodified Phi map.
    """

    if not evaluation_samples:
        raise ValueError("At least one evaluation sample is required.")
    if len(phi_maps) != len(evaluation_samples) or len(mahalanobis_distances) != len(evaluation_samples):
        raise ValueError("Each evaluation sample requires one Phi map and one Mahalanobis distance.")
    if any(sample.cost_config != cost_config for sample in evaluation_samples):
        raise ValueError("Evaluation samples must use the supplied ground-truth cost model.")

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for sample_index, (sample, phi, distance) in enumerate(
        zip(evaluation_samples, phi_maps, mahalanobis_distances)
    ):
        grid = sample.grid_map.grid
        start, goal = sample.grid_map.start, sample.grid_map.goal
        run_id = f"{sample.grid_map.distribution}:{sample.grid_map.seed}:{sample_index}"
        search_seed = (config.seed + int(sample.grid_map.seed % (2**31 - 1))) % (2**31 - 1)
        arms = [(f"neural_guidance_fixed_k{k}", k, False) for k in config.fixed_ks]
        arms.append(("neural_guidance_adaptive_k", None, True))
        for method, fixed_k, adaptive in arms:
            if adaptive:
                adaptive_result = run_ood_adaptive_neural_guidance_seeastar(
                    grid,
                    start,
                    goal,
                    phi,
                    float(distance),
                    d95,
                    k_min=config.k_min,
                    k_max=config.k_max,
                    seed=search_seed,
                    cost_config=cost_config,
                    capture_trace=True,
                )
                decision = adaptive_result.decision
                search = adaptive_result.search
                selected_k = adaptive_result.selected_k
                score = adaptive_result.ood_score
            else:
                decision = None
                search = run_neural_guidance_seeastar(
                    grid,
                    start,
                    goal,
                    phi,
                    k=int(fixed_k),
                    seed=search_seed,
                    cost_config=cost_config,
                    capture_trace=True,
                )
                selected_k = int(fixed_k)
                score = None

            unique_expansions, expansion_coverage, expansion_spread = _exploration_metrics(search, grid)
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "method_family": "neural_guidance_adaptive" if adaptive else "neural_guidance_fixed",
                    "distribution": sample.grid_map.distribution,
                    "map_seed": int(sample.grid_map.seed),
                    "sampling_seed": search_seed,
                    "phi_source": "Neural A* guidance map",
                    "mahalanobis_distance": float(distance),
                    "d95": float(d95),
                    "ood_score": float(score) if score is not None else None,
                    "k_min": config.k_min,
                    "k_max": config.k_max,
                    "selected_k": selected_k,
                    "found": bool(search.found),
                    "found_path": json.dumps(search.path),
                    "found_ground_truth_cost": search.found_ground_truth_cost if search.found else None,
                    "found_guidance_accumulated_cost": search.found_guidance_cost if search.found else None,
                    "guidance_cost_semantics": "sum Phi(v) over path nodes except start",
                    "optimal_ground_truth_cost": float(sample.ground_truth.optimal_path_cost),
                    "optimality_ratio": (
                        _ratio(True, search.found_ground_truth_cost, sample.ground_truth.optimal_path_cost)
                        if search.found else None
                    ),
                    "expanded_nodes": search.expanded_nodes,
                    "expanded_unique_nodes": unique_expansions,
                    "expanded_grid_coverage": expansion_coverage,
                    "expanded_spatial_spread": expansion_spread,
                    "search_runtime_seconds": search.search_time_seconds,
                    "sampling_overhead_seconds": search.sampling_time_seconds,
                    "stop_reason": search.stop_reason,
                    "adaptive_direction": decision.direction if decision else None,
                    "normalization": decision.normalization if decision else None,
                }
            )
            for iteration, selection in enumerate(search.selection_trace):
                trace_rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "iteration": iteration,
                        "eligible_open_candidates": json.dumps(selection.eligible_open),
                        "sampled_candidates": json.dumps(selection.sampled_candidates),
                        "sampled_f_values": json.dumps(selection.sampled_f_values),
                        "selected_candidate": json.dumps(selection.selected),
                        "expanded_node": json.dumps(selection.expanded),
                    }
                )

    config_payload = {
        "method_family": "Neural-guidance SeeA*",
        "fixed_method_identifier": "neural_guidance_fixed_k{K}",
        "adaptive_method_identifier": "neural_guidance_adaptive_k",
        "guidance_input": "Neural A* Phi",
        "guidance_accumulation": "g(start)=0; g_candidate(v)=g(u)+Phi(v)",
        "heuristic": "ChebyshevDistance + 0.001 * EuclideanDistance",
        "movement": "eight-connected; no diagonal corner-cutting; unit GT cost per legal transition",
        "sampling": "uniform without replacement from eligible OPEN; minimum current f within sampled candidates",
        "ood_score_formula": "s = 1 - exp(-D_M / D95)",
        "adaptive_k_formula": "K = Kmax - floor(s * (Kmax - Kmin))",
        "k_min": config.k_min,
        "k_max": config.k_max,
        "fixed_ks": list(config.fixed_ks),
        "d95": float(d95),
        "seed": config.seed,
        "failed_search_costs": None,
        "no_training_or_calibration_performed": True,
    }
    _write_csv(root / "neural_guidance_seeastar_per_map.csv", rows)
    _write_csv(root / "neural_guidance_seeastar_sampling_trace.csv", trace_rows)
    (root / "neural_guidance_seeastar_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "per_map": rows,
        "sampling_trace": trace_rows,
        "configuration": config_payload,
        "output_directory": root,
    }


# Explicit legacy name while retaining the original callable unchanged.
run_conventional_adaptive_seeastar_experiment = run_adaptive_seeastar_experiment

