"""Same-map ID/OOD pilot for Neural-guidance SeeA* on 64x64 maps.

The pilot never opens either holdout split. Its mandatory arms are Dijkstra,
conventional A* with Chebyshev and Euclidean heuristics, four fixed-K Neural-
guidance SeeA* arms, and the approved OOD-adaptive-K arm. Original Neural A* is
included as an available optional comparison. H1/H2/H3 and Original SeeA* are
reported as unavailable unless separately configured compatible inputs exist;
they are never trained or required by this runner.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, get_transition_cost
from benchmarks.execution import SearchWorker
from benchmarks.statistics import paired_comparison, paired_map_differences
from data.rectangular_dataset import RUN_ID
from experiments.rectangle_pipeline import SAMPLING_SEEDS, load_samples, resolve_device
from ood.mahalanobis import load_detector

try:
    import torch
except ImportError:  # pragma: no cover - Kaggle includes PyTorch
    torch = None


PRIMARY_FIXED_K = (4, 5, 10, 16)
MANDATORY_DETERMINISTIC = (
    ("dijkstra", "Dijkstra", "ground_truth_reference"),
    ("astar_chebyshev", "A* / Chebyshev", "admissible_consistent"),
    ("astar_euclidean", "A* / Euclidean", "potentially_non_admissible"),
    ("neural_astar", "Original Neural A*", "learned_guidance_baseline"),
)
RAW_FIELDS = (
    "run_id", "run_variant", "split", "density_group", "actual_density_percent",
    "map_id", "layout_hash", "phi_sha256", "start", "goal", "sampling_seed",
    "algorithm", "method", "method_family", "heuristic", "status", "success",
    "valid_path", "invalid_path", "timed_out", "unreachable_result",
    "numerical_failure", "exception", "memory_failure", "termination_reason",
    "failure_reason", "expanded_nodes", "generated_nodes", "reexpansions",
    "max_open_size", "max_closed_size", "final_open_size", "final_closed_size",
    "goal_counted_as_expanded", "sampling_operations", "configured_k",
    "effective_k_min", "effective_k_mean", "effective_k_max", "selected_candidate",
    "actual_expanded_candidate", "selection_expansion_consistent",
    "search_time_ms", "search_time_seconds", "worker_roundtrip_seconds",
    "end_to_end_runtime_seconds", "ground_truth_path_cost", "dijkstra_optimal_cost",
    "absolute_optimality_gap", "path_cost_ratio", "number_of_moves",
    "orthogonal_moves", "diagonal_moves", "geometric_path_length",
    "guidance_accumulated_cost", "raw_mahalanobis_distance", "continuous_ood_score",
    "d95", "model_loading_seconds", "neural_inference_seconds",
    "ood_feature_extraction_seconds", "mahalanobis_calculation_seconds",
    "worker_restarted",
)

PATH_METRICS = (
    "ground_truth_path_cost", "absolute_optimality_gap", "path_cost_ratio",
    "number_of_moves", "orthogonal_moves", "diagonal_moves",
    "geometric_path_length", "guidance_accumulated_cost",
)
EFFORT_METRICS = (
    "expanded_nodes", "generated_nodes", "reexpansions", "max_open_size",
    "max_closed_size", "sampling_operations", "effective_k_mean",
    "search_time_seconds", "end_to_end_runtime_seconds",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _layout_hash(grid: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _phi_hash(phi: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(phi).tobytes()).hexdigest()


def _load_cache(
    path: Path,
    expected_ids: list[str],
    *,
    expected_checkpoint_hash: str,
    d95: float,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    ids = arrays.get("map_ids")
    if ids is None or ids.astype(str).tolist() != expected_ids:
        raise ValueError(f"Inference cache {path.name} has a different map order.")
    count = len(expected_ids)
    expected_shapes = {
        "phi_maps": (count, 64, 64),
        "mahalanobis_distances": (count,),
        "continuous_ood_scores": (count,),
        "neural_inference_seconds": (count,),
        "ood_feature_seconds": (count,),
        "mahalanobis_seconds": (count,),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Inference cache {path.name} has invalid {name} shape.")
    if arrays["phi_maps"].dtype.kind not in "f" or not np.isfinite(arrays["phi_maps"]).all():
        raise ValueError("Cached Neural A* Phi must be finite floating point.")
    if np.any(arrays["phi_maps"] < 0.0):
        raise ValueError("Cached Neural A* Phi must be non-negative.")
    distance = arrays["mahalanobis_distances"].astype(np.float64, copy=False)
    score = arrays["continuous_ood_scores"].astype(np.float64, copy=False)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Cached Mahalanobis distances must be finite and non-negative.")
    if not np.isfinite(score).all() or np.any((score < 0) | (score >= 1)):
        raise ValueError("Cached OOD scores must lie in [0, 1).")
    expected_score = -np.expm1(-distance / float(d95))
    if not np.allclose(score, expected_score, rtol=0.0, atol=1e-12):
        raise ValueError("Cached OOD scores do not match the finalized D95 formula.")
    cached_hash = str(arrays.get("neural_checkpoint_sha256", np.asarray("")).item())
    if cached_hash != expected_checkpoint_hash:
        raise ValueError("Cached Phi was generated by a different Neural A* checkpoint.")
    return arrays


def _task_methods() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        {
            "method": method,
            "method_name": method,
            "algorithm": label,
            "method_family": "deterministic_search",
            "heuristic": heuristic,
            "seed": None,
        }
        for method, label, heuristic in MANDATORY_DETERMINISTIC
    ]
    for k in PRIMARY_FIXED_K:
        for seed in SAMPLING_SEEDS:
            tasks.append({
                "method": "neural_guidance_fixed",
                "method_name": f"neural_guidance_fixed_k{k}",
                "algorithm": f"Neural-guidance SeeA* / fixed K={k}",
                "method_family": "neural_guidance_fixed",
                "heuristic": "Chebyshev + 0.001 Euclidean; accumulated Phi g",
                "seed": seed,
                "k": k,
            })
    for seed in SAMPLING_SEEDS:
        tasks.append({
            "method": "neural_guidance_adaptive",
            "method_name": "neural_guidance_adaptive_k4_16",
            "algorithm": "Neural-guidance SeeA* / adaptive K=4..16",
            "method_family": "neural_guidance_adaptive",
            "heuristic": "Chebyshev + 0.001 Euclidean; accumulated Phi g",
            "seed": seed,
            "k_min": 4,
            "k_max": 16,
        })
    return tasks


def _search_task_payload(
    arm: dict[str, Any],
    *,
    sample: Any,
    phi: np.ndarray,
    distance: float,
    d95: float,
    capture_trace: bool,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "method": str(arm["method"]),
        "grid": sample.grid_map.grid,
        "start": sample.grid_map.start,
        "goal": sample.grid_map.goal,
        "instrument_neural": False,
        "capture_guidance_trace": capture_trace,
    }
    if arm["method"] in {
        "neural_astar", "neural_guidance_fixed", "neural_guidance_adaptive"
    }:
        task["phi"] = phi
    if arm.get("seed") is not None:
        task["seed"] = int(arm["seed"])
    if arm["method"] == "neural_guidance_fixed":
        task["k"] = int(arm["k"])
    elif arm["method"] == "neural_guidance_adaptive":
        task.update({
            "mahalanobis_distance": float(distance),
            "d95": float(d95),
            "k_min": int(arm["k_min"]),
            "k_max": int(arm["k_max"]),
        })
    return task


def _path_metrics(
    sample: Any,
    path: Iterable[tuple[int, int]] | None,
    phi: np.ndarray,
    optimal_cost: float,
    *,
    use_guidance: bool,
) -> dict[str, Any]:
    if path is None:
        return {"valid_path": None, "invalid_path": False, **{key: None for key in PATH_METRICS}}
    cells = [tuple(map(int, point)) for point in path]
    if not cells or cells[0] != sample.grid_map.start or cells[-1] != sample.grid_map.goal:
        return {"valid_path": False, "invalid_path": True, **{key: None for key in PATH_METRICS}}
    orthogonal = diagonal = 0
    cost = 0.0
    for source, destination in zip(cells, cells[1:]):
        step = get_transition_cost(
            sample.grid_map.grid, source, destination, sample.cost_config
        )
        if not math.isfinite(float(step)):
            return {"valid_path": False, "invalid_path": True, **{key: None for key in PATH_METRICS}}
        dr, dc = abs(destination[0] - source[0]), abs(destination[1] - source[1])
        if max(dr, dc) != 1:
            return {"valid_path": False, "invalid_path": True, **{key: None for key in PATH_METRICS}}
        if dr == 1 and dc == 1:
            diagonal += 1
        else:
            orthogonal += 1
        cost += float(step)
    moves = len(cells) - 1
    if not math.isclose(cost, float(moves), rel_tol=0.0, abs_tol=1e-12):
        return {"valid_path": False, "invalid_path": True, **{key: None for key in PATH_METRICS}}
    gap = cost - float(optimal_cost)
    if gap < -1e-10:
        return {"valid_path": False, "invalid_path": True, **{key: None for key in PATH_METRICS}}
    guidance_cost = (
        float(sum(float(phi[cell]) for cell in cells[1:])) if use_guidance else None
    )
    return {
        "valid_path": True,
        "invalid_path": False,
        "ground_truth_path_cost": float(cost),
        "absolute_optimality_gap": float(max(gap, 0.0)),
        "path_cost_ratio": float(cost / optimal_cost) if optimal_cost > 0 else 1.0,
        "number_of_moves": int(moves),
        "orthogonal_moves": int(orthogonal),
        "diagonal_moves": int(diagonal),
        "geometric_path_length": float(orthogonal + math.sqrt(2.0) * diagonal),
        "guidance_accumulated_cost": guidance_cost,
    }


def _response_fields(response: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    result = response.get("search_result")
    status = str(response.get("status", "error"))
    success = status == "completed" and bool(response.get("found", False))
    sizes = tuple(int(v) for v in getattr(result, "candidate_sizes", ())) if result is not None else ()
    selected_k = response.get("selected_k", arm.get("k"))
    if selected_k is None and response.get("adaptive_decision"):
        selected_k = response["adaptive_decision"].get("selected_k")
    def field(name: str) -> Any:
        return getattr(result, name, None) if result is not None else None
    if sizes:
        effective_min, effective_mean, effective_max = min(sizes), float(np.mean(sizes)), max(sizes)
    else:
        effective_min = response.get("effective_k_min")
        effective_mean = response.get("effective_k_mean")
        effective_max = response.get("effective_k_max")
    memory_failure = "memory" in str(response.get("error_type", "")).lower() or "outofmemory" in str(response.get("error_type", "")).lower()
    termination = str(response.get("termination_reason", "exception" if status == "error" else "unknown"))
    failure_reason = None
    if not success:
        failure_reason = (
            "timeout" if status == "timeout" else
            "exception" if status == "error" else
            "unreachable" if termination in {"open_exhausted", "frontier_exhausted"} else
            termination
        )
    return {
        "status": status,
        "success": success,
        "timed_out": status == "timeout",
        "unreachable_result": (not success and termination in {"open_exhausted", "frontier_exhausted"}),
        "numerical_failure": any(word in str(response.get("error_type", "")).lower() for word in ("floating", "overflow", "invalid")),
        "exception": response.get("error_type"),
        "memory_failure": memory_failure,
        "termination_reason": termination,
        "failure_reason": failure_reason,
        "expanded_nodes": field("expanded_nodes"),
        "generated_nodes": field("generated_nodes"),
        "reexpansions": field("reexpansions"),
        "max_open_size": field("max_open_size"),
        "max_closed_size": field("max_closed_size"),
        "final_open_size": field("final_open_size"),
        "final_closed_size": field("closed_size"),
        "sampling_operations": int(response.get("sampling_operations", len(sizes))),
        "configured_k": selected_k,
        "effective_k_min": effective_min,
        "effective_k_mean": effective_mean,
        "effective_k_max": effective_max,
        "selected_candidate": response.get("selected_candidate"),
        "actual_expanded_candidate": response.get("actual_expanded_candidate"),
        "selection_expansion_consistent": response.get("selection_expansion_consistent"),
        "path": response.get("path"),
        "found_ground_truth_cost": _parse_number(response.get("found_ground_truth_cost")),
        "found_guidance_accumulated_cost": _parse_number(response.get("found_guidance_cost")),
        "search_time_seconds": _parse_number(response.get("search_time_seconds")),
        "worker_restarted": False,
    }


def _sampled_trace_record(
    split: str, map_id: str, arm: dict[str, Any], seed: int | None, response: dict[str, Any]
) -> dict[str, Any] | None:
    result = response.get("search_result")
    trace = getattr(result, "selection_trace", ()) if result is not None else ()
    if not trace:
        return None
    return {
        "split": split,
        "map_id": map_id,
        "algorithm": arm["algorithm"],
        "sampling_seed": seed,
        "steps": [
            {
                "eligible_open": item.eligible_open,
                "sampled_candidates": item.sampled_candidates,
                "sampled_f_values": item.sampled_f_values,
                "selected_candidate": item.selected,
                "actual_expanded_candidate": item.expanded,
            }
            for item in trace
        ],
    }


def _warmup_tasks() -> list[dict[str, Any]]:
    grid = np.zeros((64, 64), dtype=np.uint8)
    phi = np.full((64, 64), 0.5, dtype=np.float64)
    base = {"grid": grid, "start": (0, 0), "goal": (63, 63), "instrument_neural": False}
    tasks = [
        {**base, "method": "dijkstra"},
        {**base, "method": "astar_chebyshev"},
        {**base, "method": "astar_euclidean"},
        {**base, "method": "neural_astar", "phi": phi},
        {**base, "method": "neural_guidance_fixed", "phi": phi, "k": 4, "seed": 11},
        {**base, "method": "neural_guidance_adaptive", "phi": phi,
         "mahalanobis_distance": 0.0, "d95": 1.0, "k_min": 4, "k_max": 16, "seed": 11},
    ]
    return tasks


def _read_raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"Existing {path.name} has a different schema; use a new run directory.")
        return list(reader)


def _write_raw_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    serializable = dict(row)
    for key in ("start", "goal", "selected_candidate", "actual_expanded_candidate"):
        value = serializable.get(key)
        if isinstance(value, tuple):
            serializable[key] = json.dumps(value)
    writer.writerow(serializable)


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _per_map_and_aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["density_group"], row["map_id"], row["method"])].append(row)
    per_map: list[dict[str, Any]] = []
    for (_, _, _, _), trials in sorted(grouped.items()):
        first = trials[0]
        successful = [row for row in trials if str(row["success"]).lower() == "true"]
        summary: dict[str, Any] = {
            "split": first["split"],
            "density_group": first["density_group"],
            "map_id": first["map_id"],
            "layout_hash": first["layout_hash"],
            "start": first["start"],
            "goal": first["goal"],
            "actual_density_percent": first["actual_density_percent"],
            "algorithm": first["algorithm"],
            "method": first["method"],
            "method_family": first["method_family"],
            "heuristic": first["heuristic"],
            "run_count": len(trials),
            "success_count": len(successful),
            "success_rate": len(successful) / len(trials),
            "timeout_count": sum(str(row["timed_out"]).lower() == "true" for row in trials),
            "failure_count": len(trials) - len(successful),
            "dijkstra_optimal_cost": first["dijkstra_optimal_cost"],
            "continuous_ood_score": first["continuous_ood_score"],
            "raw_mahalanobis_distance": first["raw_mahalanobis_distance"],
        }
        for field in EFFORT_METRICS:
            values = [value for row in trials if (value := _number(row, field)) is not None]
            summary[f"mean_{field}"] = _mean(values)
            summary[f"median_{field}"] = _median(values)
            summary[f"sd_{field}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        for field in PATH_METRICS:
            values = [value for row in successful if (value := _number(row, field)) is not None]
            summary[f"mean_{field}"] = _mean(values)
            summary[f"median_{field}"] = _median(values)
        summary["configured_k"] = first["configured_k"]
        summary["effective_k_min"] = min((v for row in trials if (v := _number(row, "effective_k_min")) is not None), default=None)
        summary["effective_k_mean"] = _mean([v for row in trials if (v := _number(row, "effective_k_mean")) is not None])
        summary["effective_k_max"] = max((v for row in trials if (v := _number(row, "effective_k_max")) is not None), default=None)
        per_map.append(summary)

    aggregate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in per_map:
        aggregate_groups[(item["split"], item["density_group"], item["method"])].append(item)
    aggregates: list[dict[str, Any]] = []
    for (split, distribution, method), maps in sorted(aggregate_groups.items()):
        first_raw = next(row for row in rows if row["split"] == split and row["density_group"] == distribution and row["method"] == method)
        record: dict[str, Any] = {
            "algorithm": first_raw["algorithm"],
            "method": method,
            "method_family": first_raw["method_family"],
            "heuristic": first_raw["heuristic"],
            "split": split,
            "density_group": distribution,
            "number_of_maps": len(maps),
            "number_of_runs": sum(int(item["run_count"]) for item in maps),
            "success_rate": float(np.mean([item["success_rate"] for item in maps])),
            "timeout_count": sum(int(item["timeout_count"]) for item in maps),
            "failure_count": sum(int(item["failure_count"]) for item in maps),
            "configured_k": first_raw["configured_k"],
        }
        for field in EFFORT_METRICS:
            map_values = [float(item[f"mean_{field}"]) for item in maps if item[f"mean_{field}"] is not None]
            record[f"mean_{field}"] = _mean(map_values)
            record[f"median_{field}"] = _median(map_values)
            record[f"sd_{field}"] = float(np.std(map_values, ddof=1)) if len(map_values) > 1 else 0.0 if map_values else None
        for field in PATH_METRICS:
            map_values = [float(item[f"mean_{field}"]) for item in maps if item[f"mean_{field}"] is not None]
            record[f"mean_{field}"] = _mean(map_values)
            record[f"median_{field}"] = _median(map_values)
        record["mean_dijkstra_optimal_cost"] = _mean([float(item["dijkstra_optimal_cost"]) for item in maps])
        record["effective_k_min"] = min((float(item["effective_k_min"]) for item in maps if item["effective_k_min"] not in (None, "")), default=None)
        record["mean_effective_k"] = _mean([float(item["effective_k_mean"]) for item in maps if item["effective_k_mean"] not in (None, "")])
        record["effective_k_max"] = max((float(item["effective_k_max"]) for item in maps if item["effective_k_max"] not in (None, "")), default=None)
        aggregates.append(record)
    return per_map, aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _pilot_statistics(per_map: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    parsed = [
        {**item, "method": item["method"], "map_id": item["map_id"]}
        for item in per_map
    ]
    comparisons: list[dict[str, Any]] = []
    by_metric_family: dict[tuple[str, str], list[int]] = defaultdict(list)
    method_pairs = [
        ("astar_chebyshev", f"neural_guidance_fixed_k{k}") for k in PRIMARY_FIXED_K
    ] + [
        ("astar_chebyshev", "neural_guidance_adaptive_k4_16")
    ] + [
        (f"neural_guidance_fixed_k{k}", "neural_guidance_adaptive_k4_16")
        for k in PRIMARY_FIXED_K
    ]
    metric_fields = (
        "success_rate", "mean_expanded_nodes", "mean_generated_nodes",
        "mean_search_time_seconds", "mean_ground_truth_path_cost",
        "mean_absolute_optimality_gap",
    )
    group_keys = sorted({(row["split"], row["density_group"]) for row in parsed})
    for split, distribution in group_keys:
        group = [row for row in parsed if row["split"] == split and row["density_group"] == distribution]
        for method_a, method_b in method_pairs:
            for metric in metric_fields:
                _, diff = paired_map_differences(
                    group,
                    metric=metric,
                    method_a=method_a,
                    method_b=method_b,
                )
                if len(diff) < 2:
                    continue
                result = paired_comparison(
                    diff,
                    bootstrap_repetitions=2_000,
                    permutation_repetitions=10_000,
                    seed=seed + len(comparisons),
                )
                comparisons.append({
                    "split": split,
                    "density_group": distribution,
                    "comparison": f"{method_a} minus {method_b}",
                    "metric": metric,
                    **result,
                })
                by_metric_family[(split, metric)].append(len(comparisons) - 1)
    for (_, _), indices in by_metric_family.items():
        adjusted = __import__("benchmarks.statistics", fromlist=["holm_adjust"]).holm_adjust(
            [comparisons[index]["sign_flip_p_value"] for index in indices]
        )
        for index, value in zip(indices, adjusted):
            comparisons[index]["holm_adjusted_p_value"] = float(value)

    correlations: list[dict[str, Any]] = []
    for split, distribution in group_keys:
        for method in [f"neural_guidance_fixed_k{k}" for k in PRIMARY_FIXED_K]:
            group = [row for row in parsed if row["split"] == split and row["density_group"] == distribution and row["method"] == method]
            for metric in ("mean_expanded_nodes", "mean_generated_nodes", "mean_search_time_seconds", "success_rate", "mean_absolute_optimality_gap", "mean_path_cost_ratio"):
                pairs = [(_number(row, "continuous_ood_score"), _number(row, metric)) for row in group]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                x = np.asarray([p[0] for p in pairs], dtype=np.float64)
                y = np.asarray([p[1] for p in pairs], dtype=np.float64)
                rho = _spearman(x, y)
                ci = None
                if rho is not None:
                    rng = np.random.default_rng(seed + len(correlations))
                    values = []
                    for _ in range(1_000):
                        take = rng.integers(0, len(x), size=len(x))
                        item = _spearman(x[take], y[take])
                        if item is not None:
                            values.append(item)
                    if values:
                        ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                correlations.append({
                    "split": split,
                    "density_group": distribution,
                    "method": method,
                    "metric": metric,
                    "map_count": len(pairs),
                    "spearman_rho": rho,
                    "bootstrap_ci95": ci,
                })
    return {
        "independent_unit": "obstacle-map layout; stochastic seeds averaged within each map",
        "comparisons": comparisons,
        "ood_score_correlations_fixed_k_only": correlations,
        "interpretation": "pilot evidence is descriptive and not a superiority claim",
    }


def _parse_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    bool_fields = (
        "success", "valid_path", "invalid_path", "timed_out", "unreachable_result",
        "numerical_failure", "memory_failure", "selection_expansion_consistent", "worker_restarted",
    )
    for field in bool_fields:
        if parsed.get(field) not in (None, ""):
            parsed[field] = str(parsed[field]).lower() == "true"
    for field in RAW_FIELDS:
        if field in {"success", "valid_path", "invalid_path", "timed_out", "unreachable_result", "numerical_failure", "memory_failure", "selection_expansion_consistent", "worker_restarted"}:
            continue
        value = parsed.get(field)
        if field in {"sampling_seed", "expanded_nodes", "generated_nodes", "reexpansions", "max_open_size", "max_closed_size", "final_open_size", "final_closed_size", "sampling_operations", "configured_k", "effective_k_min", "effective_k_max"}:
            numeric = _parse_number(value)
            parsed[field] = int(numeric) if numeric is not None else None
        elif field not in {"run_id", "run_variant", "split", "density_group", "map_id", "layout_hash", "phi_sha256", "start", "goal", "algorithm", "method", "method_family", "heuristic", "status", "exception", "termination_reason", "failure_reason", "selected_candidate", "actual_expanded_candidate"}:
            parsed[field] = _parse_number(value)
    return parsed


def run_pilot(
    data_directory: str | Path,
    result_root: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run mandatory same-map ID/OOD pilot arms and build result artifacts."""

    if device != "cuda":
        raise RuntimeError("Phase B pilot expects the verified CUDA runtime; CPU fallback is disabled.")
    resolve_device("cuda")
    data_root = Path(data_directory)
    root = Path(result_root)
    pilot_root = root / "pilot"
    complete_path = pilot_root / "pilot_complete.json"
    if complete_path.exists() or (root / "phase_b_results.zip").exists():
        raise FileExistsError(f"Completed pilot outputs already exist under {root}; choose a new run variant.")
    ready = json.loads((root / "models" / "model_ready.json").read_text(encoding="utf-8"))
    integration = json.loads((root / "integration_tests" / "inference_to_search.json").read_text(encoding="utf-8"))
    cache_manifest = json.loads((root / "guidance_maps" / "pilot_inference_manifest.json").read_text(encoding="utf-8"))
    if not (ready.get("training_complete") and ready.get("d95_calibrated")):
        raise PermissionError("The pilot requires finalized training and D95 calibration.")
    if integration.get("status") != "passed" or integration.get("ood_pilot_loaded"):
        raise PermissionError("The all-ID integration gate must pass before OOD-pilot loading.")
    if cache_manifest.get("cache_splits") != ["id_pilot", "ood_pilot"]:
        raise PermissionError("Pilot caches must be generated only after the OOD release gate.")
    if not cache_manifest.get("models_and_detector_finalized_before_ood_pilot"):
        raise PermissionError("Cache provenance does not confirm finalized models before OOD release.")

    split_samples: dict[str, list[Any]] = {}
    caches: dict[str, dict[str, np.ndarray]] = {}
    for split in ("id_pilot", "ood_pilot"):
        samples = load_samples(data_root, split)
        split_samples[split] = samples
        expected_ids = [f"{split}:{sample.grid_map.distribution}:{sample.grid_map.seed}" for sample in samples]
        caches[split] = _load_cache(
            root / "guidance_maps" / f"{split}_inference.npz",
            expected_ids,
            expected_checkpoint_hash=str(ready["model_checkpoint_sha256"]["NeuralAStar"]),
            d95=float(ready["d95"]),
        )
    if len(split_samples["id_pilot"]) != 200 or len(split_samples["ood_pilot"]) != 200:
        raise ValueError("The ID and OOD pilot must each contain exactly 200 maps.")
    detector = load_detector(root / "detectors" / "mahalanobis_detector.npz")
    d95 = float(detector.d95)
    if not math.isfinite(d95) or d95 <= 0 or d95 != float(ready["d95"]):
        raise ValueError("Detector and finalized model-ready D95 disagree or are invalid.")
    checkpoint_load_seconds = float(cache_manifest.get("model_loading_seconds", 0.0))

    integration_times = [
        float(case[key])
        for case in integration.get("cases", [])
        for key in ("fixed_k_search_seconds", "adaptive_search_seconds")
        if case.get(key) is not None
    ]
    timeout_path = pilot_root / "timeout_policy.json"
    if timeout_path.exists():
        timeout_policy = json.loads(timeout_path.read_text(encoding="utf-8"))
    else:
        timeout_policy = {
            "per_search_timeout_seconds": max(30.0, 10.0 * max(integration_times, default=0.0)),
            "derived_from": "all-ID integration runtime before any pilot search",
            "timeout_is_failure": True,
            "automatic_retry_of_failed_or_timed_out_searches": False,
            "frozen_before_pilot_searches": True,
        }
        _atomic_json(timeout_path, timeout_policy)
    timeout_seconds = float(timeout_policy["per_search_timeout_seconds"])
    if timeout_seconds <= 0:
        raise ValueError("Pilot timeout policy must be positive.")

    raw_path = pilot_root / "raw" / "search_results.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_raw_rows(raw_path)
    completed = {
        (str(row["split"]), str(row["map_id"]), str(row["method"]), str(row["sampling_seed"]))
        for row in existing_rows
    }
    arms = _task_methods()
    expected_count = sum(len(items) for items in split_samples.values()) * len(arms)
    trace_path = pilot_root / "raw" / "sampling_traces.jsonl"
    captured_trace_groups: set[tuple[str, str, str]] = set()
    traces: list[dict[str, Any]] = []
    started = perf_counter()

    with SearchWorker() as worker:
        for task in _warmup_tasks():
            warmup = worker.run(task, timeout_seconds=timeout_seconds)
            if warmup.response.get("status") != "completed":
                raise RuntimeError(f"Search warm-up failed for {task['method']}: {warmup.response}")
        for split in ("id_pilot", "ood_pilot"):
            samples = split_samples[split]
            cache = caches[split]
            phis = cache["phi_maps"]
            distances = cache["mahalanobis_distances"]
            scores = cache["continuous_ood_scores"]
            neural_times = cache["neural_inference_seconds"]
            feature_times = cache["ood_feature_seconds"]
            mahalanobis_times = cache["mahalanobis_seconds"]
            for index, sample in enumerate(samples):
                group = str(sample.grid_map.distribution)
                map_id = f"{split}:{group}:{sample.grid_map.seed}"
                phi = phis[index].astype(np.float64, copy=False)
                distance = float(distances[index])
                score = float(scores[index])
                phi_digest = _phi_hash(phi)
                optimal = float(sample.ground_truth.optimal_path_cost)
                layout = _layout_hash(sample.grid_map.grid)
                for arm in arms:
                    seed_value = arm.get("seed")
                    seed_key = "" if seed_value is None else str(seed_value)
                    key = (split, map_id, str(arm["method_name"]), seed_key)
                    if key in completed:
                        continue
                    trace_key = (
                        split,
                        group,
                        "fixed" if arm["method"] == "neural_guidance_fixed" else "adaptive",
                    )
                    capture_trace = (
                        arm["method"] in {"neural_guidance_fixed", "neural_guidance_adaptive"}
                        and seed_value == SAMPLING_SEEDS[0]
                        and trace_key not in captured_trace_groups
                    )
                    task = _search_task_payload(
                        arm, sample=sample, phi=phi, distance=distance,
                        d95=d95, capture_trace=capture_trace,
                    )
                    roundtrip = None
                    worker_restarted = False
                    try:
                        execution = worker.run(task, timeout_seconds=timeout_seconds)
                        response = dict(execution.response)
                        roundtrip = float(execution.worker_roundtrip_seconds)
                        worker_restarted = bool(execution.worker_restarted)
                    except BaseException as error:
                        response = {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    fields = _response_fields(response, arm)
                    fields["worker_restarted"] = worker_restarted
                    fields["valid_path"] = None
                    fields["invalid_path"] = False
                    path_metrics = _path_metrics(
                        sample,
                        fields.pop("path"),
                        phi,
                        optimal,
                        use_guidance=arm["method"] in {
                            "neural_guidance_fixed", "neural_guidance_adaptive", "neural_astar"
                        },
                    )
                    fields.update(path_metrics)
                    if fields["success"] and not fields["valid_path"]:
                        fields["success"] = False
                        fields["invalid_path"] = True
                        fields["failure_reason"] = "invalid_path"
                        fields["status"] = "invalid_path"
                    measured_path_cost = fields.get("ground_truth_path_cost")
                    if arm["method"] == "dijkstra" and fields["success"]:
                        if measured_path_cost is None or not math.isclose(
                            measured_path_cost, optimal, rel_tol=0.0, abs_tol=1e-12
                        ):
                            raise RuntimeError(f"Dijkstra reference disagrees with labels for {map_id}.")
                    runtime = fields.get("search_time_seconds")
                    inference_seconds = float(neural_times[index])
                    feature_seconds = float(feature_times[index])
                    distance_seconds = float(mahalanobis_times[index])
                    end_to_end = (runtime or 0.0)
                    if arm["method"] in {"neural_astar", "neural_guidance_fixed"}:
                        end_to_end += inference_seconds
                    elif arm["method"] == "neural_guidance_adaptive":
                        end_to_end += inference_seconds + feature_seconds + distance_seconds
                    row = {
                        "run_id": RUN_ID,
                        "run_variant": root.name,
                        "split": split,
                        "density_group": group,
                        "actual_density_percent": float(np.count_nonzero(sample.grid_map.grid) * 100.0 / sample.grid_map.grid.size),
                        "map_id": map_id,
                        "layout_hash": layout,
                        "phi_sha256": phi_digest,
                        "start": json.dumps(sample.grid_map.start),
                        "goal": json.dumps(sample.grid_map.goal),
                        "sampling_seed": seed_key,
                        "algorithm": arm["algorithm"],
                        "method": arm["method_name"],
                        "method_family": arm["method_family"],
                        "heuristic": arm["heuristic"],
                        **fields,
                        "goal_counted_as_expanded": True,
                        "search_time_ms": (1000.0 * runtime) if runtime is not None else None,
                        "worker_roundtrip_seconds": roundtrip,
                        "end_to_end_runtime_seconds": float(end_to_end) if runtime is not None else None,
                        "dijkstra_optimal_cost": optimal,
                        "raw_mahalanobis_distance": distance,
                        "continuous_ood_score": score,
                        "d95": d95,
                        "model_loading_seconds": checkpoint_load_seconds,
                        "neural_inference_seconds": inference_seconds,
                        "ood_feature_extraction_seconds": feature_seconds,
                        "mahalanobis_calculation_seconds": distance_seconds,
                    }
                    with raw_path.open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, extrasaction="ignore")
                        if stream.tell() == 0:
                            writer.writeheader()
                        _write_raw_row(writer, row)
                        stream.flush()
                    completed.add(key)
                    trace = _sampled_trace_record(split, map_id, arm, seed_value, response)
                    if trace is not None:
                        traces.append(trace)
                        captured_trace_groups.add(trace_key)
                    _atomic_json(pilot_root / "progress.json", {
                        "run_id": RUN_ID,
                        "run_variant": root.name,
                        "completed_rows": len(completed),
                        "expected_rows": expected_count,
                        "last_completed_map": map_id,
                        "current_method": arm["method"],
                        "status": "running",
                    })
    if traces:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as stream:
            for trace in traces:
                stream.write(json.dumps(trace, default=lambda value: list(value) if isinstance(value, tuple) else str(value)) + "\n")
    elapsed = perf_counter() - started
    raw_rows = _read_raw_rows(raw_path)
    if len(raw_rows) != expected_count:
        raise RuntimeError(
            f"Pilot is incomplete: {len(raw_rows)} of {expected_count} rows. "
            "Resume this run to continue; failed and timed-out rows are retained and not retried."
        )
    parsed_rows = [_parse_raw_row(row) for row in raw_rows]
    per_map, aggregates = _per_map_and_aggregate(parsed_rows)
    per_map_path = pilot_root / "per_map" / "per_map_comparison.csv"
    aggregate_path = pilot_root / "aggregate" / "algorithm_comparison.csv"
    _write_csv(per_map_path, per_map)
    _write_csv(aggregate_path, aggregates)
    analysis = _pilot_statistics(per_map, seed=20261106)
    _atomic_json(pilot_root / "aggregate" / "paired_statistics.json", analysis)
    pilot_summary = {
        "run_id": RUN_ID,
        "run_variant": root.name,
        "pilot_complete": True,
        "pilot_map_counts": {"id_pilot": 200, "ood_pilot": 200},
        "sampling_seeds": list(SAMPLING_SEEDS),
        "mandatory_algorithms": [
            "Dijkstra", "A* / Chebyshev", "A* / Euclidean",
            "Neural-guidance SeeA* / fixed K=4", "Neural-guidance SeeA* / fixed K=5",
            "Neural-guidance SeeA* / fixed K=10", "Neural-guidance SeeA* / fixed K=16",
            "Neural-guidance SeeA* / adaptive K=4..16",
        ],
        "optional_baselines": {
            "Original Neural A*": "included; actual implementation uses cached Neural A* Phi",
            "Original SeeA*": "unavailable; compatible learned heuristic input is not configured",
            "H1": "unavailable/disabled; no compatible configured 64x64 checkpoint",
            "H2": "unavailable/disabled; no compatible configured 64x64 checkpoint",
            "H3": "unavailable/disabled; no compatible configured 64x64 checkpoint",
        },
        "raw_rows": len(parsed_rows),
        "search_timing_excludes_loading_inference_serialization_and_plotting": True,
        "inference_cost_accounting": (
            "Phi and detector inference are cached once per map. Raw rows record shared costs; "
            "per-search end-to-end time includes only the inference required by that algorithm, "
            "and aggregate graph-search timing remains search-only."
        ),
        "goal_included_in_expanded_nodes": True,
        "expanded_node_definition": "valid non-stale removal from OPEN, including the goal node",
        "generated_node_definition": "unique traversable nodes first discovered and inserted into OPEN, including the start",
        "search_wall_time_seconds": float(elapsed),
        "timeout_policy": timeout_policy,
        "ood_pilot_loaded_after_training_calibration_and_all_id_integration": True,
        "id_holdout_loaded": False,
        "sealed_ood_test_loaded": False,
    }
    _atomic_json(pilot_root / "aggregate" / "pilot_summary.json", pilot_summary)
    from benchmarks.phase_b_report import finalize_phase_b_results

    package = finalize_phase_b_results(root, pilot_summary=pilot_summary)
    complete = {
        **pilot_summary,
        "results_zip": str(package),
        "raw_results": str(raw_path),
        "per_map_comparison": str(per_map_path),
        "algorithm_comparison": str(aggregate_path),
        "report": str(root / "reports" / "phase_b_experimental_report.md"),
        "sealed_ood_test_loaded": False,
        "id_holdout_loaded": False,
    }
    _atomic_json(complete_path, complete)
    return complete
