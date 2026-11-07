"""Isolated CPU search worker used for frozen per-search timeout handling."""

from __future__ import annotations

from dataclasses import asdict
from math import inf
from time import perf_counter
from typing import Any

import numpy as np

from algorithms.astar import (
    astar,
    chebyshev_heuristic,
    euclidean_heuristic,
    zero_heuristic,
)
from algorithms.dijkstra import dijkstra
from algorithms.movement import DEFAULT_COST_CONFIG, get_transition_cost, is_legal_transition
from adaptive.ood_adaptive_seeastar import run_ood_adaptive_neural_guidance_seeastar
from benchmarks.neural_trace import run_instrumented_neural_astar
from heuristics.inference import fixed_goal_heuristic
from neural_astar.search import run_neural_astar
from seeastar.original import SeeAStarConfig, seeastar
from seeastar.neural_guidance import run_neural_guidance_seeastar
from seeastar.uniform import UniformSeeAStarConfig, uniform_seeastar


def _dijkstra_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], result: Any):
    if not np.isfinite(result.distances[goal]):
        return None
    reverse = [goal]
    current = goal
    while current != start:
        previous = result.predecessors[current]
        if previous[0] < 0:
            raise RuntimeError("Dijkstra predecessor map cannot reconstruct the goal path.")
        current = (int(previous[0]), int(previous[1]))
        reverse.append(current)
        if len(reverse) > grid.size + 1:
            raise RuntimeError("Dijkstra predecessor map contains a cycle.")
    return tuple(reversed(reverse))


def run_search_task(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one existing algorithm without changing its search policy."""

    method = str(task["method"])
    grid = np.asarray(task["grid"], dtype=np.uint8)
    start = tuple(int(value) for value in task["start"])
    goal = tuple(int(value) for value in task["goal"])
    heuristic_map = task.get("heuristic_map")
    if heuristic_map is not None:
        heuristic_map = np.asarray(heuristic_map, dtype=np.float64)
    if method == "dijkstra":
        started = perf_counter()
        result = dijkstra(grid, start, DEFAULT_COST_CONFIG)
        elapsed = perf_counter() - started
        path = _dijkstra_path(grid, start, goal, result)
        return {
            "status": "completed",
            "search_result": result,
            "path": path,
            "found": path is not None,
            "found_ground_truth_cost": float(result.distances[goal]),
            "search_time_seconds": float(elapsed),
            "termination_reason": "frontier_exhausted",
            "dijkstra_distances": result.distances,
            "dijkstra_reachable_count": int(np.count_nonzero(result.reachable_mask)),
            "neural_trace": None,
        }

    if method == "astar_zero":
        heuristic = zero_heuristic
    elif method == "astar_chebyshev":
        heuristic = chebyshev_heuristic
    elif method == "astar_euclidean":
        heuristic = euclidean_heuristic
    elif method.startswith("astar_") and method[6:] in {"H1", "H2", "H3"}:
        if heuristic_map is None:
            raise ValueError(f"{method} requires a precomputed raw heuristic map.")
        heuristic = fixed_goal_heuristic(heuristic_map, goal)
    elif method == "neural_astar":
        if task.get("phi") is None:
            raise ValueError("Neural A* requires a precomputed guidance map.")
        if task.get("instrument_neural", True):
            result, trace = run_instrumented_neural_astar(
                grid,
                start,
                goal,
                np.asarray(task["phi"], dtype=np.float64),
            )
        else:
            result = run_neural_astar(
                grid,
                start,
                goal,
                np.asarray(task["phi"], dtype=np.float64),
            )
            trace = None
        return {
            "status": "completed",
            "search_result": result,
            "path": result.path,
            "found": bool(result.found),
            "found_ground_truth_cost": _path_ground_truth_cost(
                grid, result.path, DEFAULT_COST_CONFIG
            ),
            "found_guidance_cost": float(result.goal_guidance_cost) if result.found else None,
            "search_time_seconds": float(result.search_time_seconds),
            "termination_reason": "goal" if result.found else "open_exhausted",
            "dijkstra_distances": None,
            "dijkstra_reachable_count": None,
            "neural_trace": trace,
        }
    elif method.startswith("seeastar_original_"):
        if heuristic_map is None:
            raise ValueError("Original learned-guidance SeeA* requires a heuristic map.")
        strategy = method.removeprefix("seeastar_original_")
        result = seeastar(
            grid,
            start,
            goal,
            fixed_goal_heuristic(heuristic_map, goal),
            SeeAStarConfig(
                k=int(task["k"]),
                sampling_strategy=strategy,
                seed=int(task["seed"]),
            ),
            DEFAULT_COST_CONFIG,
        )
        return _search_response(result)
    elif method == "uniform_fixed":
        if heuristic_map is None:
            raise ValueError("Fixed-K uniform SeeA* requires a heuristic map.")
        result = uniform_seeastar(
            grid,
            start,
            goal,
            fixed_goal_heuristic(heuristic_map, goal),
            UniformSeeAStarConfig(k=int(task["k"]), seed=int(task["seed"])),
            DEFAULT_COST_CONFIG,
        )
        return _search_response(result)
    elif method == "neural_guidance_fixed":
        if task.get("phi") is None:
            raise ValueError("Fixed-K Neural-guidance SeeA* requires the Neural A* Phi map.")
        result = run_neural_guidance_seeastar(
            grid,
            start,
            goal,
            np.asarray(task["phi"], dtype=np.float64),
            k=int(task["k"]),
            seed=int(task["seed"]),
            max_expansions=task.get("max_expansions"),
            cost_config=DEFAULT_COST_CONFIG,
            capture_trace=bool(task.get("capture_guidance_trace", False)),
        )
        return _guidance_search_response(result, selected_k=result.configured_k)
    elif method == "neural_guidance_adaptive":
        if task.get("phi") is None:
            raise ValueError("Adaptive Neural-guidance SeeA* requires the Neural A* Phi map.")
        required = ("mahalanobis_distance", "d95", "k_min", "k_max", "seed")
        missing = [name for name in required if task.get(name) is None]
        if missing:
            raise ValueError(f"Adaptive Neural-guidance SeeA* is missing {missing!r}.")
        adaptive = run_ood_adaptive_neural_guidance_seeastar(
            grid,
            start,
            goal,
            np.asarray(task["phi"], dtype=np.float64),
            float(task["mahalanobis_distance"]),
            float(task["d95"]),
            k_min=int(task["k_min"]),
            k_max=int(task["k_max"]),
            seed=int(task["seed"]),
            max_expansions=task.get("max_expansions"),
            cost_config=DEFAULT_COST_CONFIG,
            capture_trace=bool(task.get("capture_guidance_trace", False)),
        )
        return _guidance_search_response(
            adaptive.search,
            selected_k=adaptive.selected_k,
            adaptive_decision=asdict(adaptive.decision),
        )
    else:
        raise ValueError(f"Unsupported benchmark search method {method!r}.")

    result = astar(grid, start, goal, heuristic, DEFAULT_COST_CONFIG)
    return _search_response(result)


def _search_response(result: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "search_result": result,
        "path": result.path,
        "found": bool(result.found),
        "found_ground_truth_cost": (
            float(result.found_ground_truth_cost) if result.found else None
        ),
        "search_time_seconds": float(result.search_time_seconds),
        "termination_reason": "goal" if result.found else "open_exhausted",
        "dijkstra_distances": None,
        "dijkstra_reachable_count": None,
        "neural_trace": None,
    }


def _guidance_search_response(
    result: Any,
    *,
    selected_k: int,
    adaptive_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep guidance accumulation distinct from measured transition cost."""

    selected = tuple(getattr(result, "selected_order", ()))
    expanded = tuple(getattr(result, "expanded_order", ()))
    expected_expanded = selected[:-1] if result.found and selected else selected
    selected_candidate = selected[-1] if selected else None
    expanded_candidate = expanded[-1] if expanded else None
    sampling_sizes = tuple(getattr(result, "candidate_sizes", ()))
    selection_consistent = expected_expanded == expanded
    if not selection_consistent:
        raise AssertionError("Selected sampling candidates do not match actual expansions.")

    return {
        "status": "completed",
        "search_result": result,
        "path": result.path,
        "found": bool(result.found),
        "selected_k": int(selected_k),
        "adaptive_decision": adaptive_decision,
        "found_ground_truth_cost": (
            float(result.found_ground_truth_cost) if result.found else None
        ),
        "found_guidance_cost": (
            float(result.found_guidance_cost) if result.found else None
        ),
        "search_time_seconds": float(result.search_time_seconds),
        "termination_reason": getattr(result, "stop_reason", "goal" if result.found else "open_exhausted"),
        "sampling_operations": len(sampling_sizes),
        "effective_k_min": min(sampling_sizes) if sampling_sizes else None,
        "effective_k_mean": float(np.mean(sampling_sizes)) if sampling_sizes else None,
        "effective_k_max": max(sampling_sizes) if sampling_sizes else None,
        "selected_candidate": selected_candidate,
        "actual_expanded_candidate": expanded_candidate,
        "selection_expansion_consistent": selection_consistent,
        "dijkstra_distances": None,
        "dijkstra_reachable_count": None,
        "neural_trace": None,
    }


def _path_ground_truth_cost(
    grid: np.ndarray,
    path: tuple[tuple[int, int], ...] | None,
    cost_config: Any,
) -> float | None:
    """Measure physical search paths under the shared ground-truth cost model."""

    if path is None:
        return None
    total = 0.0
    for source, destination in zip(path, path[1:]):
        if not is_legal_transition(grid, source, destination, cost_config):
            raise RuntimeError("Search returned a path containing an illegal transition.")
        total += get_transition_cost(grid, source, destination, cost_config)
    return float(total)


def worker_main(connection: Any) -> None:
    """Run a persistent pipe worker; per-task exceptions become raw records."""

    connection.send({"status": "ready"})
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            try:
                response = run_search_task(task)
            except BaseException as error:  # retain failures instead of dropping rows
                response = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            connection.send(response)
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()
