"""Configuration-driven search benchmark and pilot orchestration."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from adaptive.ood_adaptive_seeastar import AdaptiveKPolicy, compute_adaptive_k
from algorithms.movement import DEFAULT_COST_CONFIG, get_transition_cost, is_legal_transition
from benchmarks.config import BenchmarkConfig
from benchmarks.data import load_evaluation_splits
from benchmarks.execution import SearchWorker
from benchmarks.inference import (
    load_inference_cache,
    prepare_evaluation_inference_cache,
)
from benchmarks.reliability import analyze_score_heuristic_reliability
from benchmarks.statistics import (
    average_seeds_by_map,
    holm_adjust,
    paired_comparison,
    paired_map_differences,
)
from data.dataset import DatasetSample


def experiment_arms(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Return conventional, corrected Neural-guidance, and original SeeA* arms."""

    arms: list[dict[str, Any]] = []
    primary_min, primary_max = config.primary_k_bounds
    for heuristic in (config.primary_heuristic, *config.secondary_heuristics):
        is_primary_heuristic = heuristic == config.primary_heuristic
        bounds_to_test = config.k_bounds if is_primary_heuristic else (config.primary_k_bounds,)
        for k_min, k_max in bounds_to_test:
            is_primary_bounds = (k_min, k_max) == config.primary_k_bounds
            if is_primary_bounds:
                fixed_values = config.primary_fixed_ks
                include_reverse = True
                include_clipped = True
            else:
                fixed_values = tuple(sorted({k_min, (k_min + k_max) // 2, k_max}))
                include_reverse = False
                include_clipped = False
            for fixed_k in fixed_values:
                arms.append(
                    {
                        "method": f"uniform_{heuristic}_fixed_k{fixed_k}_bounds{k_min}_{k_max}",
                        "method_family": "fixed_k_uniform",
                        "search_implementation": "uniform_fixed",
                        "heuristic": heuristic,
                        "configured_k": fixed_k,
                        "k_min": k_min,
                        "k_max": k_max,
                        "adaptive_direction": None,
                        "ood_normalization": None,
                    }
                )
            arms.append(
                {
                    "method": f"uniform_{heuristic}_adaptive_smooth_smallk_bounds{k_min}_{k_max}",
                    "method_family": "adaptive_uniform",
                    "search_implementation": "uniform_fixed",
                    "heuristic": heuristic,
                    "configured_k": None,
                    "k_min": k_min,
                    "k_max": k_max,
                    "adaptive_direction": "higher_ood_smaller_k",
                    "ood_normalization": "smooth",
                }
            )
            if include_reverse:
                arms.append(
                    {
                        "method": f"uniform_{heuristic}_adaptive_reverse_bounds{k_min}_{k_max}",
                        "method_family": "adaptive_direction_ablation",
                        "search_implementation": "uniform_fixed",
                        "heuristic": heuristic,
                        "configured_k": None,
                        "k_min": k_min,
                        "k_max": k_max,
                        "adaptive_direction": "higher_ood_larger_k",
                        "ood_normalization": "smooth",
                    }
                )
            if include_clipped:
                arms.append(
                    {
                        "method": f"uniform_{heuristic}_adaptive_clipped_bounds{k_min}_{k_max}",
                        "method_family": "adaptive_score_ablation",
                        "search_implementation": "uniform_fixed",
                        "heuristic": heuristic,
                        "configured_k": None,
                        "k_min": k_min,
                        "k_max": k_max,
                        "adaptive_direction": "higher_ood_smaller_k",
                        "ood_normalization": "clipped",
                    }
                )
    # These corrected methods consume cached Neural A* Phi directly. Their
    # identifiers are distinct from conventional unit-cost SeeA* arms above.
    for fixed_k in config.primary_fixed_ks:
        arms.append(
            {
                "method": f"neural_guidance_fixed_k{fixed_k}",
                "method_family": "neural_guidance_fixed",
                "search_implementation": "neural_guidance_fixed",
                "heuristic": None,
                "configured_k": fixed_k,
                "k_min": config.primary_k_bounds[0],
                "k_max": config.primary_k_bounds[1],
                "adaptive_direction": None,
                "ood_normalization": None,
            }
        )
    arms.append(
        {
            "method": f"neural_guidance_adaptive_k_bounds{config.primary_k_bounds[0]}_{config.primary_k_bounds[1]}",
            "method_family": "neural_guidance_adaptive",
            "search_implementation": "neural_guidance_adaptive",
            "heuristic": None,
            "configured_k": None,
            "k_min": config.primary_k_bounds[0],
            "k_max": config.primary_k_bounds[1],
            "adaptive_direction": None,
            "ood_normalization": None,
        }
    )
    for strategy in ("uniform", "cluster", "uct"):
        arms.append(
            {
                "method": f"original_seeastar_{strategy}_H1_k{config.original_seeastar_k}",
                "method_family": "original_seeastar",
                "search_implementation": f"seeastar_original_{strategy}",
                "heuristic": "H1",
                "configured_k": config.original_seeastar_k,
                "k_min": None,
                "k_max": None,
                "adaptive_direction": None,
                "ood_normalization": None,
            }
        )
    return arms


def expected_search_count(config: BenchmarkConfig, *, full: bool) -> dict[str, int]:
    """Count predeclared searches without reducing arms or seeds."""

    maps = config.id_holdout_count + config.ood_test_count if full else 6 * config.pilot_maps_per_distribution
    stochastic_repeats = len(config.sampling_seeds) if full else len(config.pilot_sampling_seeds)
    stochastic_arms = len(experiment_arms(config))
    deterministic_methods = 7  # Dijkstra, three conventional fixed heuristics, Neural A*.
    stochastic = maps * stochastic_arms * stochastic_repeats
    deterministic = maps * deterministic_methods
    return {
        "maps": maps,
        "stochastic_arms_per_map": stochastic_arms,
        "stochastic_repeats_per_map_arm": stochastic_repeats,
        "stochastic_searches": stochastic,
        "deterministic_methods_per_map": deterministic_methods,
        "deterministic_searches": deterministic,
        "total_searches": stochastic + deterministic,
    }


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _path_summary(
    grid: np.ndarray,
    path: Sequence[Sequence[int]] | None,
    cost_config: Any,
) -> tuple[float | None, int | None, int | None, int | None, str]:
    if path is None:
        return None, None, None, None, ""
    coords = tuple((int(value[0]), int(value[1])) for value in path)
    total = 0.0
    orthogonal = 0
    diagonal = 0
    for source, destination in zip(coords, coords[1:]):
        if not is_legal_transition(grid, source, destination, cost_config):
            raise RuntimeError("Search result path contains an illegal transition.")
        total += get_transition_cost(grid, source, destination, cost_config)
        if source[0] == destination[0] or source[1] == destination[1]:
            orthogonal += 1
        else:
            diagonal += 1
    return total, len(coords) - 1, orthogonal, diagonal, _json(coords)


def _result_counters(response: Mapping[str, Any]) -> dict[str, Any]:
    result = response.get("search_result")
    trace = response.get("neural_trace")
    if response.get("status") != "completed" or result is None:
        return {}
    if hasattr(result, "candidate_sizes"):
        eligible = tuple(int(value) for value in result.eligible_open_sizes)
        sampled = tuple(int(value) for value in result.candidate_sizes)
        effective = tuple(min(int(response.get("configured_k") or 0), value) for value in eligible)
        return {
            "expanded_nodes": int(result.expanded_nodes),
            "generated_nodes": int(result.generated_nodes),
            "reexpansions": int(result.reexpansions),
            "max_open_size": int(result.max_open_size),
            "final_open_size": int(result.final_open_size),
            "closed_size": int(result.closed_size),
            "sampling_overhead_seconds": float(result.sampling_time_seconds),
            "sampling_operations": len(sampled),
            "total_eligible_open_candidates": int(sum(eligible)),
            "mean_eligible_open_candidates": float(np.mean(eligible)) if eligible else 0.0,
            "total_sampled_candidates": int(sum(sampled)),
            "mean_effective_k": float(np.mean(effective)) if effective else 0.0,
            "minimum_effective_k": min(effective) if effective else 0,
            "maximum_effective_k": max(effective) if effective else 0,
            "candidate_sizes_json": _json(sampled),
            "eligible_open_sizes_json": _json(eligible),
            "expanded_order_json": _json(result.expanded_order),
            **(
                {
                    "guidance_selection_trace_json": _json(
                        [
                            {
                                "eligible_open": item.eligible_open,
                                "sampled_candidates": item.sampled_candidates,
                                "sampled_f_values": item.sampled_f_values,
                                "selected": item.selected,
                                "expanded": item.expanded,
                            }
                            for item in result.selection_trace
                        ]
                    )
                }
                if getattr(result, "selection_trace", ())
                else {}
            ),
        }
    if hasattr(result, "selected_order"):
        if trace is None:
            return {
                "expanded_nodes": None,
                "generated_nodes": None,
                "reexpansions": 0,
                "max_open_size": None,
                "final_open_size": int(result.final_open_size),
                "closed_size": int(result.closed_size),
                "sampling_overhead_seconds": 0.0,
                "sampling_operations": 0,
                "total_eligible_open_candidates": None,
                "mean_eligible_open_candidates": None,
                "total_sampled_candidates": None,
                "mean_effective_k": None,
                "minimum_effective_k": None,
                "maximum_effective_k": None,
                "candidate_sizes_json": "",
                "eligible_open_sizes_json": "",
                "expanded_order_json": _json(result.selected_order),
                "instrumentation_equivalent": False,
            }
        selected_count = len(result.selected_order)
        expanded_count = selected_count - (1 if result.found else 0)
        return {
            "expanded_nodes": expanded_count,
            "generated_nodes": int(trace.generated_nodes),
            "reexpansions": 0,
            "max_open_size": int(trace.maximum_open_size),
            "final_open_size": int(result.final_open_size),
            "closed_size": int(result.closed_size),
            "sampling_overhead_seconds": 0.0,
            "sampling_operations": 0,
            "total_eligible_open_candidates": int(sum(trace.open_sizes)),
            "mean_eligible_open_candidates": float(np.mean(trace.open_sizes)) if trace.open_sizes else 0.0,
            "total_sampled_candidates": 0,
            "mean_effective_k": None,
            "minimum_effective_k": None,
            "maximum_effective_k": None,
            "candidate_sizes_json": "",
            "eligible_open_sizes_json": _json(trace.open_sizes),
            "expanded_order_json": _json(result.selected_order),
            "instrumentation_equivalent": True,
            "guidance_map_json": _json(result.guidance_map.tolist()),
            "accumulated_guidance_cost_json": _json(result.accumulated_guidance_cost.tolist()),
            "goal_guidance_cost": float(result.goal_guidance_cost),
        }
    return {
        "expanded_nodes": int(result.expanded_nodes),
        "generated_nodes": int(result.generated_nodes),
        "reexpansions": int(result.reexpansions),
        "max_open_size": int(result.max_open_size),
        "final_open_size": int(result.final_open_size),
        "closed_size": int(result.closed_size),
        "sampling_overhead_seconds": 0.0,
        "sampling_operations": 0,
        "total_eligible_open_candidates": None,
        "mean_eligible_open_candidates": None,
        "total_sampled_candidates": None,
        "mean_effective_k": None,
        "minimum_effective_k": None,
        "maximum_effective_k": None,
        "candidate_sizes_json": "",
        "eligible_open_sizes_json": "",
        "expanded_order_json": "",
    }


def _search_row(
    sample: DatasetSample,
    inference_row: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    method: str,
    method_family: str,
    heuristic: str | None,
    seed: int | None,
    configured_k: int | None,
    adaptive_decision: Any | None,
    worker_roundtrip_seconds: float,
    worker_restarted: bool,
    loading_seconds: Mapping[str, float],
    inference_reuse_count: int,
) -> dict[str, Any]:
    distribution = sample.grid_map.distribution
    map_id = f"{distribution}:{sample.grid_map.seed}"
    found = bool(response.get("found", False)) if response.get("status") == "completed" else False
    path = response.get("path") if found else None
    measured_cost, move_count, orthogonal, diagonal, path_json = _path_summary(
        sample.grid_map.grid, path, sample.cost_config
    )
    optimal = float(sample.ground_truth.optimal_path_cost)
    ratio = (measured_cost / optimal) if found and optimal > 0.0 else (1.0 if found and measured_cost == 0.0 else None)
    gap = (measured_cost - optimal) / optimal if found and optimal > 0.0 else (0.0 if found and optimal == 0.0 else None)
    status = str(response.get("status", "error"))
    search_seconds = response.get("search_time_seconds") if status == "completed" else None
    result = response.get("search_result")
    if adaptive_decision is None:
        adaptive_decision = response.get("adaptive_decision")
    if isinstance(adaptive_decision, Mapping):
        decision_field = adaptive_decision.get
    else:
        decision_field = lambda name, default=None: getattr(adaptive_decision, name, default)
    effective_k = response.get("selected_k", configured_k)
    counters = _result_counters({**response, "configured_k": effective_k})
    cnn_seconds = 0.0
    if heuristic in {"H1", "H2", "H3"}:
        cnn_seconds += float(inference_row[f"{heuristic}_cnn_inference_seconds"])
    elif method in {"neural_astar", "neural_guidance_fixed", "neural_guidance_adaptive"}:
        cnn_seconds += float(inference_row["neural_astar_cnn_inference_seconds"])
    ood_feature_seconds = 0.0
    ood_detector_seconds = 0.0
    if method_family.startswith("adaptive") or method_family == "neural_guidance_adaptive":
        ood_feature_seconds = float(inference_row["ood_feature_cnn_inference_seconds"])
        ood_detector_seconds = float(inference_row["ood_detector_inference_seconds"])
    if inference_reuse_count < 1:
        raise ValueError("inference_reuse_count must be positive.")
    inference_charge = (cnn_seconds + ood_feature_seconds + ood_detector_seconds) / inference_reuse_count
    total_pipeline = float(search_seconds) + inference_charge if search_seconds is not None else None
    row: dict[str, Any] = {
        "status": status,
        "error_type": response.get("error_type"),
        "error_message": response.get("error_message"),
        "timed_out": status == "timeout",
        "timeout_seconds": response.get("timeout_seconds"),
        "method": method,
        "method_family": method_family,
        "search_implementation": response.get("search_implementation"),
        "heuristic": heuristic,
        "split": "id_holdout" if distribution in {"D1", "D2", "D3"} else "ood_test",
        "distribution": distribution,
        "map_id": map_id,
        "map_seed": int(sample.grid_map.seed),
        "obstacle_layout_fingerprint": inference_row["map_fingerprint"],
        "start": _json(sample.grid_map.start),
        "goal": _json(sample.grid_map.goal),
        "sampling_seed": seed,
        "configured_k": effective_k,
        "selected_k": effective_k,
        "k_min": decision_field("k_min"),
        "k_max": decision_field("k_max"),
        "adaptive_direction": decision_field("direction"),
        "ood_normalization": decision_field("normalization"),
        "raw_mahalanobis_distance": float(inference_row["mahalanobis_distance"]),
        "smooth_ood_score": float(inference_row["smooth_ood_score"]),
        "adaptive_policy_score": decision_field("policy_score", decision_field("ood_score")),
        "d95": float(inference_row["d95"]),
        "success": found,
        "optimal_ground_truth_cost": optimal,
        "found_ground_truth_cost": measured_cost,
        "found_guidance_accumulated_cost": (
            float(response["found_guidance_cost"])
            if found and response.get("found_guidance_cost") is not None
            else None
        ),
        "optimality_ratio": ratio,
        "optimality_gap": gap,
        "path_move_count": move_count,
        "orthogonal_move_count": orthogonal,
        "diagonal_move_count": diagonal,
        "found_path_json": path_json,
        "search_runtime_seconds": search_seconds,
        "sampling_overhead_seconds": counters.get("sampling_overhead_seconds"),
        "cnn_inference_seconds": cnn_seconds,
        "ood_feature_cnn_inference_seconds": ood_feature_seconds,
        "ood_detector_inference_seconds": ood_detector_seconds,
        "ood_inference_seconds": ood_feature_seconds + ood_detector_seconds,
        "inference_reuse_count": int(inference_reuse_count),
        "amortized_inference_charge_seconds": inference_charge,
        "total_end_to_end_seconds": total_pipeline,
        "worker_roundtrip_seconds": float(worker_roundtrip_seconds),
        "worker_restarted": bool(worker_restarted),
        "model_loading_seconds_separate": _json(dict(loading_seconds)),
        **counters,
    }
    if result is not None and hasattr(result, "mean_open_size"):
        row["mean_open_size"] = float(result.mean_open_size)
    if hasattr(result, "goal_guidance_cost"):
        row["neural_goal_guidance_cost"] = float(result.goal_guidance_cost)
        row["neural_temperature"] = float(result.temperature)
        row["neural_tie_break_epsilon"] = float(result.tie_break_epsilon)
    return row


def _adaptive_decision(arm: Mapping[str, Any], distance: float, d95: float) -> Any | None:
    if arm["adaptive_direction"] is None:
        return None
    policy = AdaptiveKPolicy(
        k_min=int(arm["k_min"]),
        k_max=int(arm["k_max"]),
        direction=str(arm["adaptive_direction"]),
        normalization=str(arm["ood_normalization"]),
    )
    return compute_adaptive_k(distance, d95, policy)


def _task_for_arm(
    sample: DatasetSample,
    maps: Mapping[str, np.ndarray],
    arm: Mapping[str, Any],
    *,
    seed: int,
    decision: Any | None,
    instrument_neural: bool,
    mahalanobis_distance: float | None = None,
    d95: float | None = None,
) -> dict[str, Any]:
    implementation = str(arm["search_implementation"])
    heuristic_name = arm.get("heuristic")
    effective_k = int(decision.selected_k) if decision is not None else arm.get("configured_k")
    task: dict[str, Any] = {
        "method": implementation,
        "grid": sample.grid_map.grid,
        "start": sample.grid_map.start,
        "goal": sample.grid_map.goal,
        "seed": int(seed),
        "k": effective_k,
        "configured_k": effective_k,
        "instrument_neural": instrument_neural,
    }
    if heuristic_name in {"H1", "H2", "H3"}:
        task["heuristic_map"] = maps[str(heuristic_name)]
    if implementation in {"neural_astar", "neural_guidance_fixed", "neural_guidance_adaptive"}:
        task["phi"] = maps["phi"]
    if implementation == "neural_guidance_adaptive":
        if mahalanobis_distance is None or d95 is None:
            raise ValueError("Adaptive Neural-guidance SeeA* requires D_M and calibrated D95.")
        task.update(
            {
                "mahalanobis_distance": float(mahalanobis_distance),
                "d95": float(d95),
                "k_min": int(arm["k_min"]),
                "k_max": int(arm["k_max"]),
            }
        )
    return task


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path.stat().st_size


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _pilot_samples(samples: Sequence[DatasetSample], count_per_distribution: int) -> list[DatasetSample]:
    by_distribution: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        by_distribution.setdefault(sample.grid_map.distribution, []).append(sample)
    selected: list[DatasetSample] = []
    for distribution in ("D1", "D2", "D3", "D4", "D5", "D6"):
        choices = by_distribution.get(distribution, [])
        if len(choices) < count_per_distribution:
            raise ValueError(f"Not enough pilot maps in {distribution}.")
        selected.extend(choices[:count_per_distribution])
    return selected


def _all_search_arms(config: BenchmarkConfig) -> list[dict[str, Any]]:
    arms = [
        {
            "method": method,
            "method_family": "conventional_search",
            "search_implementation": method,
            "heuristic": heuristic,
            "configured_k": None,
            "k_min": None,
            "k_max": None,
            "adaptive_direction": None,
            "ood_normalization": None,
        }
        for method, heuristic in (
            ("dijkstra", None),
            ("astar_zero", None),
            ("astar_chebyshev", None),
            ("astar_H1", "H1"),
            ("astar_H2", "H2"),
            ("astar_H3", "H3"),
            ("neural_astar", None),
        )
    ]
    arms.extend(experiment_arms(config))
    return arms


def run_benchmark_searches(
    config: BenchmarkConfig,
    *,
    pilot: bool,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run a representative pilot or, once authorized, the full experiment.

    This orchestration does not implement search mechanics. It calls the
    existing A*, Dijkstra, Neural A*, SeeA*, and uniform SeeA* APIs in isolated
    workers. ``pilot=False`` exists for the later approved full run and is not
    called by preparation commands.
    """

    root = Path(output_directory or config.output_directory)
    ready_path = root / "model_ready_manifest.json"
    if not ready_path.is_file():
        raise PermissionError("Evaluation is sealed until training and D95 calibration finish.")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not (ready.get("training_complete") and ready.get("d95_calibrated")):
        raise PermissionError("Evaluation is sealed until training and D95 calibration finish.")
    if not pilot and config.timeout_seconds is None:
        raise ValueError("The full benchmark requires a pilot-frozen timeout policy.")
    eval_root = root / "evaluation"
    cache_path = eval_root / "map_inference_cache.npz"
    inference_rows_path = eval_root / "per_map_inference_and_reliability.csv"
    inference_manifest_path = eval_root / "inference_manifest.json"
    if not inference_manifest_path.is_file():
        prepare_evaluation_inference_cache(
            root,
            model_ready_manifest=ready_path,
            device=config.device,
        )
    ids, inference_maps = load_inference_cache(cache_path)
    inference_rows = _read_csv(inference_rows_path)
    if len(ids) != len(inference_rows):
        raise ValueError("Prediction cache and inference metadata row counts do not match.")
    id_samples, ood_samples = load_evaluation_splits(
        root / "data", model_ready_manifest=ready_path
    )
    samples = [*id_samples, *ood_samples]
    if [f"{sample.grid_map.distribution}:{sample.grid_map.seed}" for sample in samples] != ids:
        raise ValueError("The current evaluation split order differs from the prediction cache.")
    if pilot:
        analyze_map_inference(
            inference_rows,
            config=config,
            output_directory=eval_root / "reliability_analysis",
        )
        selected_samples = _pilot_samples(samples, config.pilot_maps_per_distribution)
        seed_values = tuple(config.pilot_sampling_seeds)
        timeout = None
        mode_root = root / "pilot"
        instrument_neural = True
    else:
        selected_samples = samples
        seed_values = tuple(config.sampling_seeds)
        timeout = float(config.timeout_seconds)
        mode_root = root / "full"
        instrument_neural = True
    all_samples_by_id = {f"{sample.grid_map.distribution}:{sample.grid_map.seed}": (i, sample) for i, sample in enumerate(samples)}
    arms = _all_search_arms(config)
    map_count = len(selected_samples)
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    worker_timings: list[float] = []
    stochastic_arm_count = sum(arm["method_family"] != "conventional_search" for arm in arms)
    progress_total = map_count * (stochastic_arm_count * len(seed_values) + 7)
    progress_done = 0
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=progress_total, desc="Timing pilot" if pilot else "Full benchmark")
    except ImportError:
        progress = None
    with SearchWorker() as worker:
        startup_seconds = worker.worker_startup_seconds
        for sample in selected_samples:
            map_id = f"{sample.grid_map.distribution}:{sample.grid_map.seed}"
            cache_index, matched_sample = all_samples_by_id[map_id]
            inference_row = inference_rows[cache_index]
            map_arrays = {name: array[cache_index] for name, array in inference_maps.items()}
            d95 = float(inference_row["d95"])
            distance = float(inference_row["mahalanobis_distance"])
            # The one-per-map K decisions are reused unchanged across all paired seeds.
            decisions = {arm["method"]: _adaptive_decision(arm, distance, d95) for arm in arms}
            for arm in arms:
                if arm["method_family"] == "conventional_search":
                    continue
                decision = decisions[arm["method"]]
                seeds: Iterable[int | None] = seed_values
                for seed in seeds:
                    task = _task_for_arm(
                        matched_sample,
                        map_arrays,
                        arm,
                        seed=0 if seed is None else int(seed),
                        decision=decision,
                        instrument_neural=instrument_neural,
                        mahalanobis_distance=distance,
                        d95=d95,
                    )
                    task["search_implementation"] = arm["search_implementation"]
                    try:
                        execution = worker.run(task, timeout_seconds=timeout)
                        response = dict(execution.response)
                        response["search_implementation"] = arm["search_implementation"]
                        if response.get("status") == "timeout":
                            response["timeout_seconds"] = timeout
                        worker_timings.append(execution.worker_roundtrip_seconds)
                        row = _search_row(
                            matched_sample,
                            inference_row,
                            response,
                            method=str(arm["method"]),
                            method_family=str(arm["method_family"]),
                            heuristic=arm.get("heuristic"),
                            seed=None if seed is None else int(seed),
                            configured_k=(
                                int(response["selected_k"])
                                if response.get("selected_k") is not None
                                else int(decision.selected_k) if decision is not None
                                else arm.get("configured_k")
                            ),
                            adaptive_decision=decision,
                            worker_roundtrip_seconds=execution.worker_roundtrip_seconds,
                            worker_restarted=execution.worker_restarted,
                            loading_seconds={},
                            inference_reuse_count=len(seed_values),
                        )
                    except BaseException as error:
                        row = {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "timed_out": False,
                            "method": arm["method"],
                            "method_family": arm["method_family"],
                            "heuristic": arm.get("heuristic"),
                            "split": "id_holdout" if sample.grid_map.distribution in {"D1", "D2", "D3"} else "ood_test",
                            "distribution": sample.grid_map.distribution,
                            "map_id": map_id,
                            "sampling_seed": seed,
                            "success": False,
                            "configured_k": decision.selected_k if decision is not None else arm.get("configured_k"),
                            "raw_mahalanobis_distance": distance,
                            "smooth_ood_score": float(inference_row["smooth_ood_score"]),
                            "d95": d95,
                        }
                    rows.append(row)
                    progress_done += 1
                    if progress is not None:
                        progress.update(1)
            # Deterministic methods are run once per map. Sampling seeds apply only to sampled arms.
            for arm in arms:
                if arm["method_family"] != "conventional_search":
                    continue
                task = _task_for_arm(
                    matched_sample,
                    map_arrays,
                    arm,
                    seed=0,
                    decision=None,
                    instrument_neural=instrument_neural,
                    mahalanobis_distance=distance,
                    d95=d95,
                )
                try:
                    execution = worker.run(task, timeout_seconds=timeout)
                    response = dict(execution.response)
                    response["search_implementation"] = arm["search_implementation"]
                    if response.get("status") == "timeout":
                        response["timeout_seconds"] = timeout
                    worker_timings.append(execution.worker_roundtrip_seconds)
                    row = _search_row(
                        matched_sample,
                        inference_row,
                        response,
                        method=str(arm["method"]),
                        method_family=str(arm["method_family"]),
                        heuristic=arm.get("heuristic"),
                        seed=None,
                        configured_k=None,
                        adaptive_decision=None,
                        worker_roundtrip_seconds=execution.worker_roundtrip_seconds,
                        worker_restarted=execution.worker_restarted,
                        loading_seconds={},
                        inference_reuse_count=1,
                    )
                except BaseException as error:
                    row = {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "timed_out": False,
                        "method": arm["method"],
                        "method_family": arm["method_family"],
                        "heuristic": arm.get("heuristic"),
                        "split": "id_holdout" if sample.grid_map.distribution in {"D1", "D2", "D3"} else "ood_test",
                        "distribution": sample.grid_map.distribution,
                        "map_id": map_id,
                        "sampling_seed": None,
                        "success": False,
                    }
                rows.append(row)
                progress_done += 1
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()
    elapsed = perf_counter() - started
    mode_root.mkdir(parents=True, exist_ok=True)
    raw_path = mode_root / "search_per_map_seed_raw.csv"
    raw_bytes = _write_csv(raw_path, rows)
    pilot_summary: dict[str, Any] | None = None
    if pilot:
        finite_search = [float(row["search_runtime_seconds"]) for row in rows if row.get("search_runtime_seconds") not in (None, "") and _finite_or_none(row.get("search_runtime_seconds"))]
        if not finite_search:
            raise RuntimeError("Timing pilot produced no completed search timings.")
        p99 = float(np.quantile(finite_search, 0.99))
        frozen_timeout = float(math.ceil(max(60.0, 10.0 * p99) / 10.0) * 10.0)
        config_with_timeout = BenchmarkConfig.load(root / "benchmark_config.json") if (root / "benchmark_config.json").is_file() else config
        from dataclasses import replace
        frozen_config = replace(config_with_timeout, timeout_seconds=frozen_timeout)
        frozen_config.save(root / "benchmark_config.json")
        per_method: dict[str, dict[str, Any]] = {}
        for method in sorted({str(row["method"]) for row in rows}):
            values = np.asarray([
                float(row["search_runtime_seconds"])
                for row in rows
                if row["method"] == method and _finite_or_none(row.get("search_runtime_seconds"))
            ], dtype=np.float64)
            per_method[method] = {
                "completed_runs": int(values.size),
                "median_search_seconds": float(np.median(values)) if values.size else None,
                "p95_search_seconds": float(np.quantile(values, 0.95)) if values.size else None,
                "maximum_search_seconds": float(values.max()) if values.size else None,
                "timeout_count": sum(row["method"] == method and bool(row.get("timed_out")) for row in rows),
                "error_count": sum(row["method"] == method and row.get("status") == "error" for row in rows),
            }
        expected = expected_search_count(config, full=True)
        pilot_per_run = float(np.mean(finite_search))
        estimated_search_seconds = pilot_per_run * expected["total_searches"]
        inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
        per_map_inference = sum(
            float(row["H1_cnn_inference_seconds"])
            + float(row["H2_cnn_inference_seconds"])
            + float(row["H3_cnn_inference_seconds"])
            + float(row["neural_astar_cnn_inference_seconds"])
            + float(row["ood_total_inference_seconds"])
            for row in inference_rows
        )
        estimated_total = estimated_search_seconds + per_map_inference
        full_row_estimate = expected["total_searches"]
        estimated_raw_bytes = int(max(raw_bytes, 1) * full_row_estimate / max(len(rows), 1) * 1.25)
        pilot_summary = {
            "timing_pilot_complete": True,
            "full_benchmark_launched": False,
            "pilot_map_count": len(selected_samples),
            "pilot_sampling_seeds": list(seed_values),
            "search_rows": len(rows),
            "search_runtime_seconds_p50": float(np.quantile(finite_search, 0.50)),
            "search_runtime_seconds_p95": float(np.quantile(finite_search, 0.95)),
            "search_runtime_seconds_p99": p99,
            "search_runtime_seconds_max": float(max(finite_search)),
            "frozen_timeout_seconds_per_search": frozen_timeout,
            "timeout_policy": "max(60 seconds, 10x pilot p99), rounded up to the next 10 seconds",
            "pilot_timeouts": sum(bool(row.get("timed_out")) for row in rows),
            "pilot_errors": sum(row.get("status") == "error" for row in rows),
            "worker_startup_seconds": float(startup_seconds),
            "worker_restart_count": worker.worker_restart_count,
            "worker_roundtrip_p50_seconds": float(np.median(worker_timings)) if worker_timings else None,
            "pilot_wallclock_seconds": float(elapsed),
            "per_method_timing": per_method,
            "expected_full_searches": expected,
            "estimated_search_runtime_hours_mean_extrapolation": estimated_search_seconds / 3600.0,
            "estimated_inference_seconds_all_eval_maps": per_map_inference,
            "estimated_search_plus_cached_inference_hours": estimated_total / 3600.0,
            "estimated_full_raw_csv_bytes_with_25pct_margin": estimated_raw_bytes,
            "pilot_raw_csv_bytes": raw_bytes,
            "full_rows_csv": str(root / "full" / "search_per_map_seed_raw.csv"),
            "storage_estimate_note": "Linear estimate from pilot CSV bytes per search row; excludes raw model maps, plots, and filesystem overhead.",
        }
        (mode_root / "timing_pilot_summary.json").write_text(json.dumps(pilot_summary, indent=2, sort_keys=True), encoding="utf-8")
        return {"rows": rows, "raw_csv": raw_path, "summary": pilot_summary, "config": frozen_config}
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    analysis = analyze_search_results(
        rows,
        inference_rows,
        config=config,
        output_directory=mode_root / "analysis",
    )
    full_manifest = {
        "full_benchmark_complete": True,
        "search_rows": len(rows),
        "wallclock_seconds": float(elapsed),
        "frozen_timeout_seconds": timeout,
        "sampling_seeds": list(seed_values),
        "expected_search_count": expected_search_count(config, full=True),
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in ("completed", "timeout", "error")
        },
        "inference_manifest": inference_manifest,
        "raw_search_csv": str(raw_path),
        "analysis_directory": str(mode_root / "analysis"),
        "model_loading_seconds": inference_manifest.get("model_loading_seconds"),
        "search_algorithms_unchanged": True,
        "adaptive_k_algorithm_unchanged": True,
        "uniform_sampling_preserved_for_fixed_and_adaptive_methods": True,
        "neural_guidance_methods_use_phi": True,
        "neural_guidance_fixed_k_values": list(config.primary_fixed_ks),
        "neural_guidance_adaptive_bounds": list(config.primary_k_bounds),
        "conventional_uniform_adaptive_remains_separate": True,
        "analysis_outputs": analysis["figures"],
    }
    manifest_path = mode_root / "full_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rows": rows,
        "raw_csv": raw_path,
        "wallclock_seconds": elapsed,
        "analysis": analysis,
        "manifest": full_manifest,
        "manifest_path": manifest_path,
    }


def _finite_or_none(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _inference_reliability_rows(inference_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "mae", "rmse", "overestimation_rate", "mean_overestimation_magnitude",
        "maximum_overestimation_magnitude", "consistency_violation_rate",
        "mean_consistency_violation_magnitude", "maximum_consistency_violation_magnitude",
    )
    return [
        {
            "map_id": row["map_id"],
            "distribution": row["distribution"],
            "heuristic_method": method,
            "mahalanobis_distance": float(row["mahalanobis_distance"]),
            "smooth_ood_score": float(row["smooth_ood_score"]),
            **{metric: float(row[f"{method}_{metric}"]) for metric in metrics},
        }
        for row in inference_rows
        for method in ("H1", "H2", "H3")
    ]


def analyze_map_inference(
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    config: BenchmarkConfig,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Publish the direct OOD-score versus raw heuristic reliability analysis."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    per_map = _inference_reliability_rows(inference_rows)
    score_analysis = analyze_score_heuristic_reliability(
        per_map, score_field="smooth_ood_score", repetitions=config.bootstrap_repetitions, seed=config.base_seed
    )
    distance_analysis = analyze_score_heuristic_reliability(
        per_map, score_field="mahalanobis_distance", repetitions=config.bootstrap_repetitions, seed=config.base_seed + 900_000
    )
    ood_metrics = _ood_score_metrics(inference_rows, seed=config.base_seed)
    (root / "heuristic_reliability_correlations_smooth_score.json").write_text(
        json.dumps(score_analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "heuristic_reliability_correlations_raw_distance.json").write_text(
        json.dumps(distance_analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "ood_score_evaluation.json").write_text(
        json.dumps(ood_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(root / "heuristic_reliability_per_map.csv", per_map)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        figure_root = root / "figures"
        figure_root.mkdir(parents=True, exist_ok=True)
        distributions = ("D1", "D2", "D3", "D4", "D5", "D6")
        scores_by_distribution = [
            [float(row["smooth_ood_score"]) for row in inference_rows if row["distribution"] == distribution]
            for distribution in distributions
        ]
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.boxplot(scores_by_distribution, labels=distributions, showfliers=True)
        axis.set_ylabel("Continuous normalized Mahalanobis score (not a probability)")
        axis.set_title("OOD score by map distribution")
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(figure_root / "ood_score_by_distribution.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for axis, metric in zip(axes.ravel(), ("mae", "rmse", "overestimation_rate", "consistency_violation_rate")):
            for distribution in distributions:
                group = [row for row in per_map if row["heuristic_method"] == "H1" and row["distribution"] == distribution]
                axis.scatter(
                    [float(row["smooth_ood_score"]) for row in group],
                    [float(row[metric]) for row in group],
                    s=15,
                    alpha=0.6,
                    label=distribution,
                )
            axis.set(xlabel="Continuous OOD score", ylabel=metric.replace("_", " "), title=f"H1: score versus {metric.replace('_', ' ')}")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.savefig(figure_root / "h1_ood_score_reliability_scatter.png", dpi=180)
        plt.close(fig)
    except ImportError:
        figure_root = root / "figures"
    return {
        "correlations_smooth_score": score_analysis,
        "correlations_raw_distance": distance_analysis,
        "ood_score_metrics": ood_metrics,
        "figure_directory": str(figure_root),
    }


def _map_average_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if not _finite_or_none(value):
            continue
        key = (str(row["map_id"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def analyze_search_results(
    rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    config: BenchmarkConfig,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Aggregate seed replicates by map, paired inference, correlations, and figures."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    method_names = sorted({str(row["method"]) for row in rows})
    metrics = (
        "search_runtime_seconds",
        "expanded_nodes",
        "generated_nodes",
        "reexpansions",
        "optimality_gap",
        "optimality_ratio",
        "total_end_to_end_seconds",
        "sampling_overhead_seconds",
    )
    map_values = {metric: _map_average_rows(rows, metric) for metric in metrics}
    summaries: list[dict[str, Any]] = []
    for method in method_names:
        for group in sorted({str(row["distribution"]) for row in rows} | {"ID" if str(row["distribution"]) in {"D1","D2","D3"} else "OOD" for row in rows}):
            subset = [row for row in rows if row["method"] == method and (
                row["distribution"] == group
                or (group == "ID" and row["distribution"] in {"D1", "D2", "D3"})
                or (group == "OOD" and row["distribution"] in {"D4", "D5", "D6"})
            )]
            if not subset:
                continue
            successes = [bool(row.get("success", False)) for row in subset]
            successful_ratios = [float(row["optimality_ratio"]) for row in subset if _finite_or_none(row.get("optimality_ratio"))]
            successful_gaps = [float(row["optimality_gap"]) for row in subset if _finite_or_none(row.get("optimality_gap"))]
            summaries.append({
                "method": method,
                "distribution": group,
                "runs": len(subset),
                "unique_maps": len({row["map_id"] for row in subset}),
                "success_rate": float(np.mean(successes)),
                "timeout_count": sum(bool(row.get("timed_out")) for row in subset),
                "error_count": sum(row.get("status") == "error" for row in subset),
                "mean_optimality_ratio_successful_only": float(np.mean(successful_ratios)) if successful_ratios else None,
                "mean_optimality_gap_successful_only": float(np.mean(successful_gaps)) if successful_gaps else None,
                **{
                    f"mean_{metric}": float(np.nanmean([
                        float(row[metric]) for row in subset if _finite_or_none(row.get(metric))
                    ])) if any(_finite_or_none(row.get(metric)) for row in subset) else None
                    for metric in metrics
                },
            })
    paired_rows: list[dict[str, Any]] = []
    comparisons: list[tuple[str, str, str, str]] = []
    adaptive_method = f"uniform_{config.primary_heuristic}_adaptive_smooth_smallk_bounds{config.primary_k_bounds[0]}_{config.primary_k_bounds[1]}"
    primary_fixed_methods = [
        f"uniform_{config.primary_heuristic}_fixed_k{k}_bounds{config.primary_k_bounds[0]}_{config.primary_k_bounds[1]}"
        for k in config.primary_fixed_ks
    ]
    comparisons.extend((adaptive_method, method, "primary_fixed_k", method) for method in primary_fixed_methods)
    guidance_adaptive_method = (
        f"neural_guidance_adaptive_k_bounds{config.primary_k_bounds[0]}_{config.primary_k_bounds[1]}"
    )
    guidance_fixed_methods = [
        f"neural_guidance_fixed_k{k}" for k in config.primary_fixed_ks
    ]
    comparisons.extend(
        (guidance_adaptive_method, method, "neural_guidance_fixed_k", method)
        for method in guidance_fixed_methods
    )
    comparisons.extend(
        (
            guidance_adaptive_method,
            f"original_seeastar_{strategy}_H1_k{config.original_seeastar_k}",
            "original_seeastar",
            strategy,
        )
        for strategy in ("uniform", "cluster", "uct")
    )
    comparisons.extend(
        (guidance_adaptive_method, method, "conventional_search", method)
        for method in (
            "astar_zero",
            "astar_chebyshev",
            "astar_H1",
            "astar_H2",
            "astar_H3",
            "neural_astar",
            "dijkstra",
        )
    )
    comparisons.extend((adaptive_method, f"original_seeastar_{strategy}_H1_k{config.original_seeastar_k}", "original_seeastar", strategy) for strategy in ("uniform", "cluster", "uct"))
    comparisons.extend((adaptive_method, method, "conventional_search", method) for method in ("astar_zero", "astar_chebyshev", "astar_H1", "astar_H2", "astar_H3", "neural_astar", "dijkstra"))
    metric_tests = ("search_runtime_seconds", "expanded_nodes", "optimality_gap", "total_end_to_end_seconds", "sampling_overhead_seconds")
    raw_p_values: dict[str, list[float]] = {metric: [] for metric in metric_tests}
    for adaptive, reference, comparison_family, label in comparisons:
        for distribution in ("ID", "OOD", "D1", "D2", "D3", "D4", "D5", "D6"):
            def belongs(row: Mapping[str, Any]) -> bool:
                if distribution == "ID":
                    return row["distribution"] in {"D1", "D2", "D3"}
                if distribution == "OOD":
                    return row["distribution"] in {"D4", "D5", "D6"}
                return row["distribution"] == distribution
            map_ids = sorted({
                str(row["map_id"])
                for row in rows
                if row["method"] == adaptive and belongs(row)
            })
            for metric in metric_tests:
                first, second = map_values[metric], map_values[metric]
                differences: list[float] = []
                for map_id in map_ids:
                    a, b = first.get((map_id, adaptive)), second.get((map_id, reference))
                    # Runtime inference is restricted to map pairs where both completed;
                    # success rates and timeouts are reported separately.
                    if a is None or b is None:
                        continue
                    differences.append(a - b)
                test = paired_comparison(
                    differences,
                    bootstrap_repetitions=config.bootstrap_repetitions,
                    permutation_repetitions=config.permutation_repetitions,
                    seed=config.base_seed + len(paired_rows) + 1,
                )
                key = f"{adaptive}__minus__{reference}__{distribution}__{metric}"
                raw_p_values[metric].append(float(test["sign_flip_p_value"]))
                paired_rows.append({
                    "comparison_family": comparison_family,
                    "comparison_label": label,
                    "adaptive_method": adaptive,
                    "reference_method": reference,
                    "distribution": distribution,
                    "metric": metric,
                    "difference_definition": "adaptive minus reference; negative is lower",
                    "runtime_inference_note": "Search/end-to-end runtime tests use map pairs with finite recorded runtimes; success/timeout rates are reported separately.",
                    **test,
                    "p_value_holm": None,
                    "comparison_id": key,
                })
    for metric in metric_tests:
        adjusted = iter(holm_adjust(raw_p_values[metric]))
        for row in paired_rows:
            if row["metric"] == metric:
                row["p_value_holm"] = next(adjusted)
    per_map_reliability = [
        {
            "map_id": row["map_id"],
            "distribution": row["distribution"],
            "heuristic_method": method,
            "mahalanobis_distance": float(row["mahalanobis_distance"]),
            "smooth_ood_score": float(row["smooth_ood_score"]),
            **{metric: float(row[f"{method}_{metric}"]) for metric in (
                "mae", "rmse", "overestimation_rate", "mean_overestimation_magnitude",
                "maximum_overestimation_magnitude", "consistency_violation_rate",
                "mean_consistency_violation_magnitude", "maximum_consistency_violation_magnitude",
            )},
        }
        for row in inference_rows
        for method in ("H1", "H2", "H3")
    ]
    correlations = analyze_score_heuristic_reliability(
        per_map_reliability,
        repetitions=config.bootstrap_repetitions,
        seed=config.base_seed,
    )
    raw_distance_correlations = analyze_score_heuristic_reliability(
        per_map_reliability,
        score_field="mahalanobis_distance",
        repetitions=config.bootstrap_repetitions,
        seed=config.base_seed + 900_000,
    )
    _write_csv(output / "search_summary_by_method_distribution.csv", summaries)
    _write_csv(output / "paired_map_comparisons.csv", paired_rows)
    _write_csv(output / "heuristic_reliability_per_map.csv", per_map_reliability)
    (output / "heuristic_reliability_correlations.json").write_text(json.dumps(correlations, indent=2, sort_keys=True), encoding="utf-8")
    (output / "heuristic_reliability_raw_distance_correlations.json").write_text(json.dumps(raw_distance_correlations, indent=2, sort_keys=True), encoding="utf-8")
    (output / "statistical_methods.json").write_text(json.dumps({
        "statistical_unit": "map",
        "sampling_repeats": list(config.sampling_seeds),
        "repeats_averaged_within_map_before_inference": True,
        "paired_method": "paired map differences on identical map IDs",
        "confidence_intervals": "percentile bootstrap of paired map mean differences",
        "hypothesis_test": "two-sided Monte Carlo sign-flip test of paired map differences",
        "effect_size": "signed Wilcoxon rank-biserial correlation",
        "multiplicity": "Holm step-down correction within each metric family across all listed comparisons/distributions",
        "correlation_confidence_intervals": "map-level percentile bootstrap for Spearman rho",
        "runtime_failure_handling": "timeouts and errors are retained; success is analyzed separately; runtime pair tests use finite successful records only",
    }, indent=2, sort_keys=True), encoding="utf-8")
    ood_metrics = _ood_score_metrics(inference_rows, seed=config.base_seed)
    (output / "ood_score_evaluation.json").write_text(json.dumps(ood_metrics, indent=2, sort_keys=True), encoding="utf-8")
    figures = _make_benchmark_figures(rows, inference_rows, output / "figures")
    return {
        "summary_rows": summaries,
        "paired_rows": paired_rows,
        "correlations_smooth_score": correlations,
        "correlations_raw_distance": raw_distance_correlations,
        "ood_score_metrics": ood_metrics,
        "figures": figures,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


def _binary_ood_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return rank-based AUROC and average precision for higher=more OOD."""

    positives = scores[labels]
    negatives = scores[~labels]
    if not positives.size or not negatives.size:
        return float("nan"), float("nan")
    ranks = _average_ranks(scores)
    auc = (float(ranks[labels].sum()) - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives))
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels, dtype=np.int64)
    precision = cumulative_true / np.arange(1, len(labels) + 1)
    average_precision = float(precision[sorted_labels].mean())
    return float(auc), average_precision


def _ood_score_metrics(inference_rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    scores = np.asarray([float(row["smooth_ood_score"]) for row in inference_rows], dtype=np.float64)
    labels = np.asarray([str(row["distribution"]) in {"D4", "D5", "D6"} for row in inference_rows], dtype=bool)
    distances = np.asarray([float(row["mahalanobis_distance"]) for row in inference_rows], dtype=np.float64)
    threshold = float(1.0 - np.exp(-1.0))  # score corresponding to D_M = D95; predeclared scale
    auc, aupr = _binary_ood_metrics(scores, labels)
    id_values, ood_values = scores[~labels], scores[labels]
    rng = np.random.default_rng(seed)
    bootstrap_auc: list[float] = []
    bootstrap_ap: list[float] = []
    for _ in range(2_000):
        id_draw = id_values[rng.integers(0, len(id_values), size=len(id_values))]
        ood_draw = ood_values[rng.integers(0, len(ood_values), size=len(ood_values))]
        draw_scores = np.concatenate((id_draw, ood_draw))
        draw_labels = np.concatenate((np.zeros(len(id_draw), dtype=bool), np.ones(len(ood_draw), dtype=bool)))
        draw_auc, draw_ap = _binary_ood_metrics(draw_scores, draw_labels)
        bootstrap_auc.append(draw_auc)
        bootstrap_ap.append(draw_ap)
    return {
        "score_is_probability": False,
        "score_name": "continuous smooth normalized Mahalanobis distance",
        "threshold": threshold,
        "threshold_definition": "s = 1-exp(-1), corresponding to D_M = D95; it is an explicitly selected operating point, not a calibrated probability threshold",
        "d95": float(inference_rows[0]["d95"]) if inference_rows else None,
        "id_map_count": int(np.count_nonzero(~labels)),
        "ood_map_count": int(np.count_nonzero(labels)),
        "id_mean_distance": float(distances[~labels].mean()),
        "ood_mean_distance": float(distances[labels].mean()),
        "id_mean_score": float(id_values.mean()),
        "ood_mean_score": float(ood_values.mean()),
        "auroc": auc,
        "auroc_ci95": [float(np.quantile(bootstrap_auc, 0.025)), float(np.quantile(bootstrap_auc, 0.975))],
        "aupr": aupr,
        "aupr_ci95": [float(np.quantile(bootstrap_ap, 0.025)), float(np.quantile(bootstrap_ap, 0.975))],
        "fpr_at_threshold": float(np.mean(id_values >= threshold)),
        "fnr_at_threshold": float(np.mean(ood_values < threshold)),
        "score_distribution_by_map_type": {
            distribution: {
                "map_count": sum(row["distribution"] == distribution for row in inference_rows),
                "mean_score": float(np.mean([row["smooth_ood_score"] for row in inference_rows if row["distribution"] == distribution])),
                "std_score": float(np.std([row["smooth_ood_score"] for row in inference_rows if row["distribution"] == distribution], ddof=1)),
                "mean_raw_distance": float(np.mean([row["mahalanobis_distance"] for row in inference_rows if row["distribution"] == distribution])),
            }
            for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")
        },
    }


def _make_benchmark_figures(
    rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> list[str]:
    """Create compact distribution-level plots from all raw runs."""

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError:
        return []
    output_directory.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    key_methods = sorted({str(row["method"]) for row in rows if "adaptive_smooth_smallk" in str(row["method"]) or "fixed_k" in str(row["method"]) or str(row["method"]).startswith("neural_guidance_") or str(row["method"]).startswith("astar_") or str(row["method"]) in {"neural_astar", "dijkstra"}})
    for metric, ylabel, filename in (
        ("search_runtime_seconds", "Search runtime (seconds)", "runtime_by_distribution.png"),
        ("expanded_nodes", "Expanded nodes", "expansions_by_distribution.png"),
        ("optimality_gap", "Optimality gap (successful searches)", "optimality_gap_by_distribution.png"),
        ("total_end_to_end_seconds", "Search plus per-map inference (seconds)", "end_to_end_by_distribution.png"),
    ):
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        distributions = ("D1", "D2", "D3", "D4", "D5", "D6")
        width = 0.8 / max(len(key_methods), 1)
        for method_index, method in enumerate(key_methods):
            means = []
            errors = []
            for distribution in distributions:
                values: dict[str, list[float]] = {}
                for row in rows:
                    if row["method"] == method and row["distribution"] == distribution and _finite_or_none(row.get(metric)):
                        values.setdefault(str(row["map_id"]), []).append(float(row[metric]))
                map_means = np.asarray([np.mean(value) for value in values.values()], dtype=np.float64)
                means.append(float(map_means.mean()) if len(map_means) else np.nan)
                errors.append(float(map_means.std(ddof=1) / np.sqrt(len(map_means))) if len(map_means) > 1 else 0.0)
            x = np.arange(len(distributions)) + (method_index - (len(key_methods) - 1) / 2) * width
            ax.bar(x, means, width=width, yerr=errors, capsize=2, label=method)
        ax.set_xticks(np.arange(len(distributions)), distributions)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by map distribution; bars show mean ± map-level SEM")
        ax.grid(axis="y", alpha=0.25)
        if len(key_methods) <= 18:
            ax.legend(fontsize=6, ncol=2)
        path = output_directory / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))
    score = np.asarray([float(row["smooth_ood_score"]) for row in inference_rows])
    dists = [str(row["distribution"]) for row in inference_rows]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    boxdata = [score[np.asarray(dists) == distribution] for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")]
    ax.boxplot(boxdata, labels=("D1", "D2", "D3", "D4", "D5", "D6"), showfliers=True)
    ax.set_ylabel("Continuous OOD score (not a probability)")
    ax.set_title("OOD score distribution by map family")
    ax.grid(axis="y", alpha=0.25)
    path = output_directory / "ood_score_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(str(path))
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for distribution in ("D1", "D2", "D3", "D4", "D5", "D6"):
        selected = [row for row in inference_rows if row["distribution"] == distribution]
        ax.scatter(
            [float(row["smooth_ood_score"]) for row in selected],
            [float(row["H1_mae"]) for row in selected],
            s=18,
            alpha=0.65,
            label=distribution,
        )
    ax.set_xlabel("Continuous OOD score (not a probability)")
    ax.set_ylabel("H1 raw heuristic MAE")
    ax.set_title("Heuristic error versus OOD score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    path = output_directory / "heuristic_error_vs_ood_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(str(path))
    k_rows = [row for row in rows if row.get("method_family") in {"adaptive_uniform", "adaptive_direction_ablation", "adaptive_score_ablation", "neural_guidance_adaptive"}]
    if k_rows:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        for name in sorted({str(row["method"]) for row in k_rows}):
            selected = [row for row in k_rows if row["method"] == name]
            ax.scatter([float(row["smooth_ood_score"]) for row in selected], [int(row["selected_k"]) for row in selected], s=12, alpha=0.4, label=name)
        ax.set_xlabel("Continuous OOD score")
        ax.set_ylabel("Selected configured K")
        ax.set_title("Adaptive K versus OOD score")
        ax.grid(alpha=0.2)
        if len({row["method"] for row in k_rows}) <= 8:
            ax.legend(fontsize=7)
        path = output_directory / "adaptive_k_vs_ood_score.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))
        proposed = [row for row in k_rows if "adaptive_smooth_smallk" in str(row["method"]) or "neural_guidance_adaptive" in str(row["method"])]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        axes = axes.ravel()
        completed_proposed = [row for row in proposed if _finite_or_none(row.get("search_runtime_seconds"))]
        axes[0].scatter(
            [float(row["smooth_ood_score"]) for row in completed_proposed],
            [float(row["search_runtime_seconds"]) for row in completed_proposed],
            s=12,
            alpha=0.45,
        )
        axes[0].set(xlabel="Continuous OOD score", ylabel="Search runtime (seconds)", title="Runtime versus OOD score")
        axes[1].scatter(
            [int(row["selected_k"]) for row in proposed if _finite_or_none(row.get("sampling_overhead_seconds"))],
            [float(row["sampling_overhead_seconds"]) for row in proposed if _finite_or_none(row.get("sampling_overhead_seconds"))],
            s=12,
            alpha=0.45,
        )
        axes[1].set(xlabel="Configured K", ylabel="Sampling overhead (seconds)", title="Sampling overhead versus K")
        axes[2].scatter(
            [int(row["selected_k"]) for row in proposed if _finite_or_none(row.get("search_runtime_seconds"))],
            [float(row["search_runtime_seconds"]) for row in proposed if _finite_or_none(row.get("search_runtime_seconds"))],
            s=12,
            alpha=0.45,
        )
        axes[2].set(xlabel="Configured K", ylabel="Search runtime (seconds)", title="Runtime versus K")
        axes[3].scatter(
            [int(row["selected_k"]) for row in proposed if _finite_or_none(row.get("optimality_ratio"))],
            [float(row["optimality_ratio"]) for row in proposed if _finite_or_none(row.get("optimality_ratio"))],
            s=12,
            alpha=0.45,
        )
        axes[3].set(xlabel="Configured K", ylabel="Ground-truth path-cost ratio", title="Solution quality versus K")
        for axis in axes:
            axis.grid(alpha=0.2)
        path = output_directory / "adaptive_runtime_overhead_quality_ablations.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        distributions = ("D1", "D2", "D3", "D4", "D5", "D6")
        policies = sorted({str(row["method"]) for row in proposed})
        for policy in policies:
            selected = [row for row in proposed if row["method"] == policy]
            values = [[int(row["selected_k"]) for row in selected if row["distribution"] == distribution] for distribution in distributions]
            axes[0].plot(distributions, [np.mean(value) if value else np.nan for value in values], marker="o", label=policy)
        axes[0].set(ylabel="Mean configured K", title="Adaptive K by distribution")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.25)
        method_set = sorted({str(row["method"]) for row in rows if "fixed_k" in str(row["method"]) or "adaptive_smooth_smallk" in str(row["method"]) or str(row["method"]).startswith("neural_guidance_")})
        rates = []
        for method in method_set:
            values = [bool(row.get("success", False)) for row in rows if row["method"] == method]
            rates.append(float(np.mean(values)) if values else np.nan)
        axes[1].bar(np.arange(len(method_set)), rates)
        axes[1].set_xticks(np.arange(len(method_set)), method_set, rotation=75, ha="right", fontsize=7)
        axes[1].set(ylabel="Search success rate", title="Fixed-K and adaptive success")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(axis="y", alpha=0.25)
        path = output_directory / "adaptive_k_and_success_by_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))
    return saved
