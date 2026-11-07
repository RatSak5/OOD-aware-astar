"""Matched-map comparisons for Dijkstra, A*, and published SeeA* variants."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from algorithms.astar import (
    AStarResult,
    HeuristicFunction,
    astar,
    chebyshev_heuristic,
    zero_heuristic,
)
from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig
from data.dataset import DatasetSample, assert_no_map_leakage
from seeastar.original import SeeAStarConfig, SeeAStarResult, seeastar
from seeastar.uniform import UniformSeeAStarConfig, uniform_seeastar

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        description = kwargs.get("desc", "Benchmarking")
        total = len(iterable)

        def _iterate() -> Any:
            for index, item in enumerate(iterable, start=1):
                yield item
                print(f"\r{description}: {index}/{total}", end="", flush=True)
            if total:
                print()

        return _iterate()


def _optimality_ratio(found: bool, found_cost: float, optimal_cost: float) -> float:
    if not found or not math.isfinite(optimal_cost):
        return float("nan")
    if optimal_cost == 0.0:
        return 1.0 if found_cost == 0.0 else float("inf")
    return float(found_cost / optimal_cost)


def _path_json(path: tuple[tuple[int, int], ...] | None) -> str:
    return json.dumps(path)


def _row(
    method: str,
    sample: DatasetSample,
    result: AStarResult | SeeAStarResult,
    optimal_cost: float,
    *,
    strategy: str,
) -> dict[str, Any]:
    sampling_seconds = (
        result.sampling_time_seconds if isinstance(result, SeeAStarResult) else 0.0
    )
    return {
        "method": method,
        "distribution": sample.grid_map.distribution,
        "map_seed": int(sample.grid_map.seed),
        "start": json.dumps(sample.grid_map.start),
        "goal": json.dumps(sample.grid_map.goal),
        "sampling_strategy": strategy,
        "found": bool(result.found),
        "found_path": _path_json(result.path),
        "optimal_path_cost": float(optimal_cost),
        "found_ground_truth_cost": float(result.found_ground_truth_cost),
        "optimality_ratio": _optimality_ratio(
            result.found, result.found_ground_truth_cost, optimal_cost
        ),
        "path_move_count": (
            len(result.path) - 1 if result.path is not None else float("nan")
        ),
        "expanded_nodes": int(result.expanded_nodes),
        "generated_nodes": int(result.generated_nodes),
        "reexpansions": int(result.reexpansions),
        "mean_open_size": float(result.mean_open_size),
        "max_open_size": int(result.max_open_size),
        "final_open_size": int(result.final_open_size),
        "sampling_operations": len(result.candidate_sizes)
        if isinstance(result, SeeAStarResult)
        else 0,
        "total_eligible_open_candidates": int(sum(result.eligible_open_sizes))
        if isinstance(result, SeeAStarResult)
        else 0,
        "mean_eligible_open_candidates": float(np.mean(result.eligible_open_sizes))
        if isinstance(result, SeeAStarResult) and result.eligible_open_sizes
        else 0.0,
        "total_sampled_candidates": int(sum(result.candidate_sizes))
        if isinstance(result, SeeAStarResult)
        else 0,
        "closed_size": int(result.closed_size),
        "sampling_overhead_seconds": float(sampling_seconds),
        "search_time_seconds": float(result.search_time_seconds),
        "total_runtime_seconds": float(result.search_time_seconds),
        "stop_reason": (
            result.stop_reason
            if isinstance(result, SeeAStarResult)
            else ("goal" if result.found else "open_exhausted")
        ),
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["distribution"]), []).append(row)
        groups.setdefault((row["method"], "all"), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method, distribution), group in sorted(groups.items()):
        found_rows = [row for row in group if row["found"]]
        ratios = np.asarray(
            [
                row["optimality_ratio"]
                for row in found_rows
                if math.isfinite(row["optimality_ratio"])
            ],
            dtype=np.float64,
        )
        summaries.append(
            {
                "method": method,
                "distribution": distribution,
                "num_maps": len(group),
                "success_rate": float(np.mean([row["found"] for row in group])),
                "mean_optimality_ratio": (
                    float(np.mean(ratios)) if len(ratios) else float("nan")
                ),
                "mean_path_move_count": (
                    float(np.mean([row["path_move_count"] for row in found_rows]))
                    if found_rows
                    else float("nan")
                ),
                "mean_expanded_nodes": float(
                    np.mean([row["expanded_nodes"] for row in group])
                ),
                "mean_generated_nodes": float(
                    np.mean([row["generated_nodes"] for row in group])
                ),
                "mean_open_size": float(
                    np.mean([row["mean_open_size"] for row in group])
                ),
                "mean_max_open_size": float(
                    np.mean([row["max_open_size"] for row in group])
                ),
                "mean_closed_size": float(
                    np.mean([row["closed_size"] for row in group])
                ),
                "mean_sampling_overhead_seconds": float(
                    np.mean([row["sampling_overhead_seconds"] for row in group])
                ),
                "mean_total_runtime_seconds": float(
                    np.mean([row["total_runtime_seconds"] for row in group])
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_search_benchmarks(
    samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    heuristic: HeuristicFunction = chebyshev_heuristic,
    seeastar_config: SeeAStarConfig = SeeAStarConfig(),
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> dict[str, Any]:
    """Compare target-directed Dijkstra, conventional A*, and all paper variants.

    Every method receives the same map, endpoints, ground-truth cost convention,
    and (except Dijkstra's zero heuristic) heuristic callable. The generic
    SeeA* method has no single canonical sampler in the paper, so this runner
    reports its original Uniform, Cluster, and UCT strategies separately.
    Search time includes each SeeA* sampler's full candidate-generation and
    minimum-f ranking work; that portion is also reported as sampling overhead.
    """

    if not samples:
        raise ValueError("Benchmark samples cannot be empty.")
    if not callable(heuristic):
        raise TypeError("heuristic must be callable.")
    if {sample.grid_map.distribution for sample in samples} != {
        "D1", "D2", "D3", "D4", "D5", "D6"
    }:
        raise ValueError("Benchmark input must cover each distribution D1-D6.")
    for index, sample in enumerate(samples):
        if sample.cost_config != cost_config:
            raise ValueError(f"sample {index} uses a different ground-truth cost config.")
    assert_no_map_leakage({"benchmark": samples})
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    strategies = ("uniform", "cluster", "uct")
    for sample_index, sample in enumerate(
        tqdm(samples, desc="SeeA* matched-map benchmark")
    ):
        grid = sample.grid_map.grid
        start, goal = sample.grid_map.start, sample.grid_map.goal
        # Zero-heuristic A* is target-directed Dijkstra on the shared graph.
        dijkstra_result = astar(grid, start, goal, zero_heuristic, cost_config)
        optimum = dijkstra_result.found_ground_truth_cost
        rows.append(_row("Dijkstra", sample, dijkstra_result, optimum, strategy="none"))

        astar_result = astar(grid, start, goal, heuristic, cost_config)
        rows.append(
            _row("Conventional A*", sample, astar_result, optimum, strategy="none")
        )

        for strategy_index, strategy in enumerate(strategies):
            run_seed = (
                seeastar_config.seed
                + int(sample.grid_map.seed % (2**31 - 1))
                + sample_index * 1_000_003
                + strategy_index * 10_000_019
            ) % (2**31 - 1)
            if strategy == "uniform":
                # Keep the research variant on its dedicated API, which cannot
                # select the cluster or UCT samplers.
                uniform_config = UniformSeeAStarConfig(
                    k=seeastar_config.k,
                    max_expansions=seeastar_config.max_expansions,
                    seed=run_seed,
                    cost_tolerance=seeastar_config.cost_tolerance,
                )
                result = uniform_seeastar(
                    grid, start, goal, heuristic, uniform_config, cost_config
                )
            else:
                config = SeeAStarConfig(
                    k=seeastar_config.k,
                    sampling_strategy=strategy,
                    max_expansions=seeastar_config.max_expansions,
                    num_clusters=seeastar_config.num_clusters,
                    eta=seeastar_config.eta,
                    cb=seeastar_config.cb,
                    seed=run_seed,
                    cost_tolerance=seeastar_config.cost_tolerance,
                )
                result = seeastar(grid, start, goal, heuristic, config, cost_config)
            method_name = {
                "uniform": "SeeA* (Uniform)",
                "cluster": "SeeA* (Cluster)",
                "uct": "SeeA* (UCT)",
            }[strategy]
            rows.append(_row(method_name, sample, result, optimum, strategy=strategy))

    summaries = _summarize(rows)
    _write_csv(root / "search_benchmark_per_map.csv", rows)
    _write_csv(root / "search_benchmark_summary.csv", summaries)
    config_payload = {
        "heuristic_name": getattr(
            heuristic, "__name__", heuristic.__class__.__name__
        ),
        "ground_truth_cost_config": asdict(cost_config),
        "seeastar_config": {
            "K": seeastar_config.k,
            "cluster_count": seeastar_config.num_clusters,
            "eta": seeastar_config.eta,
            "cb": seeastar_config.cb,
            "max_expansions": seeastar_config.max_expansions,
            "seed": seeastar_config.seed,
            "K_policy": "fixed per search; no OOD adaptation",
            "sampling_strategies": list(strategies),
        },
        "map_count": len(samples),
        "distribution_counts": {
            distribution: sum(
                sample.grid_map.distribution == distribution for sample in samples
            )
            for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")
        },
    }
    (root / "search_benchmark_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "per_map": rows,
        "summary": summaries,
        "output_directory": root,
        "heuristic_name": config_payload["heuristic_name"],
        "cost_model": cost_config.model,
        "seeastar_config": config_payload["seeastar_config"],
    }
