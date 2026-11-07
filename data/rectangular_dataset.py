"""Reproducible 64x64 datasets with shared rectangular obstacle generation.

The module is deliberately separate from :mod:`data.map_generator`, whose
Prompt 8 distributions use different obstacle processes.  All six density
groups here use the same rectangle, endpoint, connectivity, and label rules.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig, get_neighbors
from data.ground_truth import compute_ground_truth


RUN_ID = "64x64_rectangles_v1"
BASE_SEED = 20261106
MAP_SHAPE = (64, 64)
MIN_CHEBYSHEV_SEPARATION = 24
RECTANGLE_SIDE_MIN = 2
RECTANGLE_SIDE_MAX = 14
GENERATOR_VERSION = "rectangles-v1"
CHANNEL_ORDER = ("obstacle_occupancy", "start_one_hot", "goal_one_hot")

DENSITY_INTERVALS_PERCENT: dict[str, tuple[int, int]] = {
    "D1": (10, 16),
    "D2": (17, 23),
    "D3": (24, 30),
    "D4": (3, 8),
    "D5": (33, 37),
    "D6": (38, 42),
}

SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "training": {"D1": 300, "D2": 300, "D3": 300},
    "validation_calibration": {"D1": 60, "D2": 60, "D3": 60},
    "id_pilot": {"D1": 20, "D2": 20, "D3": 20},
    "id_holdout": {"D1": 60, "D2": 60, "D3": 60},
    "ood_pilot": {"D4": 20, "D5": 20, "D6": 20},
    "ood_test_sealed": {"D4": 60, "D5": 60, "D6": 60},
}


@dataclass(frozen=True, slots=True)
class RectangleGenerationConfig:
    """Parameters shared by every distribution, split, and map."""

    run_id: str = RUN_ID
    base_seed: int = BASE_SEED
    height: int = MAP_SHAPE[0]
    width: int = MAP_SHAPE[1]
    rectangle_width_min: int = RECTANGLE_SIDE_MIN
    rectangle_width_max: int = RECTANGLE_SIDE_MAX
    rectangle_height_min: int = RECTANGLE_SIDE_MIN
    rectangle_height_max: int = RECTANGLE_SIDE_MAX
    minimum_chebyshev_separation: int = MIN_CHEBYSHEV_SEPARATION
    max_candidate_attempts_per_map: int = 1000
    maximum_rectangles_per_candidate: int = 1000
    movement: GroundTruthCostConfig = DEFAULT_COST_CONFIG

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError(f"run_id must be {RUN_ID!r}.")
        if (self.height, self.width) != MAP_SHAPE:
            raise ValueError("This dataset is fixed to 64x64 maps.")
        if (self.rectangle_width_min, self.rectangle_width_max) != (2, 14):
            raise ValueError("Rectangle width must be sampled uniformly from 2 through 14.")
        if (self.rectangle_height_min, self.rectangle_height_max) != (2, 14):
            raise ValueError("Rectangle height must be sampled uniformly from 2 through 14.")
        if self.minimum_chebyshev_separation != MIN_CHEBYSHEV_SEPARATION:
            raise ValueError("The minimum Chebyshev separation must remain 24.")
        if self.max_candidate_attempts_per_map < 1 or self.maximum_rectangles_per_candidate < 1:
            raise ValueError("Generation attempt limits must be positive.")


@dataclass(frozen=True, slots=True)
class RectangleCandidate:
    grid: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    rectangles: tuple[tuple[int, int, int, int], ...]
    actual_density: float
    overlap_placements: int
    start_goal_separation: int
    map_seed: int
    accepted_attempt_index: int
    layout_hash: str


class CandidateRejected(RuntimeError):
    """A candidate layout failed a named acceptance check."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _layout_hash(grid: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _seed_value(parts: Iterable[int]) -> int:
    sequence = np.random.SeedSequence([int(part) for part in parts])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def split_group_seed(base_seed: int, split_name: str, distribution: str) -> int:
    """Return a stable seed unique to a split and density group."""

    split_names = tuple(SPLIT_COUNTS)
    if split_name not in split_names or distribution not in DENSITY_INTERVALS_PERCENT:
        raise ValueError("Unknown split or density distribution.")
    return _seed_value((base_seed, split_names.index(split_name) + 1,
                        tuple(DENSITY_INTERVALS_PERCENT).index(distribution) + 1))


def sample_seed_for(group_seed: int, sample_index: int) -> int:
    """Return the reproducible per-map seed recorded in the manifest."""

    return _seed_value((group_seed, sample_index + 1))


def _candidate_rng(map_seed: int, attempt_index: int) -> np.random.Generator:
    return np.random.default_rng(_seed_value((map_seed, attempt_index + 1)))


def _density_cell_bounds(distribution: str, cell_count: int) -> tuple[int, int]:
    low_percent, high_percent = DENSITY_INTERVALS_PERCENT[distribution]
    lower = math.ceil(low_percent * cell_count / 100.0)
    upper = math.floor(high_percent * cell_count / 100.0)
    return lower, upper


def _rectangular_layout(
    distribution: str,
    rng: np.random.Generator,
    config: RectangleGenerationConfig,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...], int]:
    """Draw one rectangle union, stopping at its group's lower density bound."""

    height, width = config.height, config.width
    lower_cells, upper_cells = _density_cell_bounds(distribution, height * width)
    grid = np.zeros((height, width), dtype=np.uint8)
    rectangles: list[tuple[int, int, int, int]] = []
    total_rectangle_area = 0
    for _ in range(config.maximum_rectangles_per_candidate):
        rect_width = int(rng.integers(config.rectangle_width_min, config.rectangle_width_max + 1))
        rect_height = int(rng.integers(config.rectangle_height_min, config.rectangle_height_max + 1))
        top = int(rng.integers(0, height - rect_height + 1))
        left = int(rng.integers(0, width - rect_width + 1))
        bottom, right = top + rect_height, left + rect_width
        rectangles.append((top, left, rect_height, rect_width))
        total_rectangle_area += rect_height * rect_width
        grid[top:bottom, left:right] = 1
        occupied = int(np.count_nonzero(grid))
        if occupied > upper_cells:
            raise CandidateRejected("density_target")
        if occupied >= lower_cells:
            overlap_placements = total_rectangle_area - occupied
            return grid, tuple(rectangles), overlap_placements
    raise CandidateRejected("density_target")


def _free_components(
    grid: np.ndarray,
    movement: GroundTruthCostConfig,
) -> list[list[tuple[int, int]]]:
    visited = np.zeros(grid.shape, dtype=np.bool_)
    components: list[list[tuple[int, int]]] = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] != 0 or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque(((row, col),))
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while queue:
                cell = queue.popleft()
                component.append(cell)
                for neighbor in get_neighbors(grid, cell, movement):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)
    return components


def _select_endpoints(
    grid: np.ndarray,
    rng: np.random.Generator,
    config: RectangleGenerationConfig,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Select endpoints in one legal component with separation at least 24."""

    components = _free_components(grid, config.movement)
    connected_components = [component for component in components if len(component) >= 2]
    if not connected_components:
        raise CandidateRejected("connectivity")

    eligible: list[tuple[list[tuple[int, int]], str, int, int]] = []
    required = config.minimum_chebyshev_separation
    for component in connected_components:
        rows = [cell[0] for cell in component]
        columns = [cell[1] for cell in component]
        row_range = max(rows) - min(rows)
        column_range = max(columns) - min(columns)
        axes = []
        if row_range >= required:
            axes.append(("row", row_range))
        if column_range >= required:
            axes.append(("column", column_range))
        if axes:
            axis, span = axes[int(rng.integers(0, len(axes)))]
            eligible.append((component, axis, span, len(component)))
    if not eligible:
        raise CandidateRejected("endpoint_selection")

    weights = np.asarray([item[3] for item in eligible], dtype=np.float64)
    selected_index = int(rng.choice(len(eligible), p=weights / weights.sum()))
    component, axis, _, _ = eligible[selected_index]
    if axis == "row":
        minimum = min(cell[0] for cell in component)
        maximum = max(cell[0] for cell in component)
        first_edge = [cell for cell in component if cell[0] == minimum]
        second_edge = [cell for cell in component if cell[0] == maximum]
    else:
        minimum = min(cell[1] for cell in component)
        maximum = max(cell[1] for cell in component)
        first_edge = [cell for cell in component if cell[1] == minimum]
        second_edge = [cell for cell in component if cell[1] == maximum]
    start = first_edge[int(rng.integers(0, len(first_edge)))]
    goal = second_edge[int(rng.integers(0, len(second_edge)))]
    if int(rng.integers(0, 2)):
        start, goal = goal, start
    separation = max(abs(start[0] - goal[0]), abs(start[1] - goal[1]))
    if separation < required or start == goal:
        raise CandidateRejected("endpoint_selection")
    return start, goal, separation


def generate_candidate(
    distribution: str,
    map_seed: int,
    attempt_index: int,
    config: RectangleGenerationConfig = RectangleGenerationConfig(),
) -> RectangleCandidate:
    """Generate one deterministic accepted map candidate or raise a reason."""

    if distribution not in DENSITY_INTERVALS_PERCENT:
        raise ValueError(f"Unknown density distribution {distribution!r}.")
    rng = _candidate_rng(map_seed, attempt_index)
    grid, rectangles, overlap_placements = _rectangular_layout(distribution, rng, config)
    start, goal, separation = _select_endpoints(grid, rng, config)
    density = float(np.count_nonzero(grid) / grid.size)
    low, high = DENSITY_INTERVALS_PERCENT[distribution]
    if not low / 100.0 <= density <= high / 100.0:
        raise CandidateRejected("density_target")
    return RectangleCandidate(
        grid=grid,
        start=start,
        goal=goal,
        rectangles=rectangles,
        actual_density=density,
        overlap_placements=overlap_placements,
        start_goal_separation=separation,
        map_seed=map_seed,
        accepted_attempt_index=attempt_index,
        layout_hash=_layout_hash(grid),
    )


def canonical_inputs(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Return the three stored logical channels in their fixed channel order."""

    inputs = np.zeros((3, *grid.shape), dtype=np.uint8)
    inputs[0] = np.asarray(grid, dtype=np.uint8)
    inputs[1, start[0], start[1]] = 1
    inputs[2, goal[0], goal[1]] = 1
    return inputs


def _new_split_arrays() -> dict[str, list[Any]]:
    return {key: [] for key in (
        "maps", "inputs", "starts", "goals", "distribution_labels", "map_ids",
        "seeds", "accepted_attempt_indices", "actual_densities", "rectangle_counts",
        "rectangles_json", "overlap_placements", "start_goal_separations",
        "shortest_path_distances", "distance_maps", "reachable_masks",
        "optimal_path_coordinates", "layout_hashes",
    )}


def _append_candidate(
    arrays: dict[str, list[Any]],
    candidate: RectangleCandidate,
    split_name: str,
    distribution: str,
    sample_index: int,
    shortest_distances: np.ndarray,
    reachable: np.ndarray,
    optimal_path: tuple[tuple[int, int], ...],
) -> None:
    map_id = f"{split_name}_{distribution}_{sample_index:04d}"
    arrays["maps"].append(candidate.grid)
    arrays["inputs"].append(canonical_inputs(candidate.grid, candidate.start, candidate.goal))
    arrays["starts"].append(candidate.start)
    arrays["goals"].append(candidate.goal)
    arrays["distribution_labels"].append(distribution)
    arrays["map_ids"].append(map_id)
    arrays["seeds"].append(candidate.map_seed)
    arrays["accepted_attempt_indices"].append(candidate.accepted_attempt_index)
    arrays["actual_densities"].append(candidate.actual_density)
    arrays["rectangle_counts"].append(len(candidate.rectangles))
    arrays["rectangles_json"].append(json.dumps(candidate.rectangles, separators=(",", ":")))
    arrays["overlap_placements"].append(candidate.overlap_placements)
    arrays["start_goal_separations"].append(candidate.start_goal_separation)
    arrays["shortest_path_distances"].append(float(shortest_distances[candidate.start]))
    arrays["distance_maps"].append(shortest_distances.astype(np.float64, copy=False))
    arrays["reachable_masks"].append(reachable.astype(np.bool_, copy=False))
    arrays["optimal_path_coordinates"].append(np.asarray(optimal_path, dtype=np.int32))
    arrays["layout_hashes"].append(candidate.layout_hash)


def _packed_arrays(arrays: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    paths = arrays["optimal_path_coordinates"]
    path_offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    for index, path in enumerate(paths):
        path_offsets[index + 1] = path_offsets[index] + len(path)
    flat_paths = np.concatenate(paths, axis=0) if path_offsets[-1] else np.empty((0, 2), dtype=np.int32)
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "maps": np.stack(arrays["maps"]).astype(np.uint8),
        "inputs": np.stack(arrays["inputs"]).astype(np.uint8),
        "starts": np.asarray(arrays["starts"], dtype=np.int32),
        "goals": np.asarray(arrays["goals"], dtype=np.int32),
        "distribution_labels": np.asarray(arrays["distribution_labels"], dtype=np.str_),
        "map_ids": np.asarray(arrays["map_ids"], dtype=np.str_),
        "generation_seeds": np.asarray(arrays["seeds"], dtype=np.uint64),
        "accepted_attempt_indices": np.asarray(arrays["accepted_attempt_indices"], dtype=np.int32),
        "actual_densities": np.asarray(arrays["actual_densities"], dtype=np.float64),
        "rectangle_counts": np.asarray(arrays["rectangle_counts"], dtype=np.int16),
        "rectangles_json": np.asarray(arrays["rectangles_json"], dtype=np.str_),
        "overlap_placements": np.asarray(arrays["overlap_placements"], dtype=np.int32),
        "start_goal_separations": np.asarray(arrays["start_goal_separations"], dtype=np.int16),
        "shortest_path_distances": np.asarray(arrays["shortest_path_distances"], dtype=np.float64),
        "distance_maps": np.stack(arrays["distance_maps"]).astype(np.float64),
        "reachable_masks": np.stack(arrays["reachable_masks"]).astype(np.bool_),
        "optimal_path_coordinates": flat_paths,
        "path_offsets": path_offsets,
        "layout_hashes": np.asarray(arrays["layout_hashes"], dtype=np.str_),
        "channel_order": np.asarray(CHANNEL_ORDER, dtype=np.str_),
        "movement_model": np.asarray("eight_connected_no_corner_cutting_unit_cost", dtype=np.str_),
    }


def _validate_map_record(
    candidate: RectangleCandidate,
    distribution: str,
    ground_truth: Any,
) -> None:
    if candidate.grid.shape != MAP_SHAPE:
        raise RuntimeError("Generated map is not 64x64.")
    channels = canonical_inputs(candidate.grid, candidate.start, candidate.goal)
    if channels.shape != (3, 64, 64):
        raise RuntimeError("Logical dataset input channels have an invalid shape.")
    if int(np.sum(channels[1])) != 1 or int(np.sum(channels[2])) != 1:
        raise RuntimeError("Each endpoint channel must contain exactly one cell.")
    if candidate.start == candidate.goal or candidate.grid[candidate.start] or candidate.grid[candidate.goal]:
        raise RuntimeError("Endpoints must be distinct free cells.")
    low, high = DENSITY_INTERVALS_PERCENT[distribution]
    if not low / 100.0 <= candidate.actual_density <= high / 100.0:
        raise RuntimeError("Accepted map falls outside its assigned density interval.")
    if candidate.start_goal_separation < MIN_CHEBYSHEV_SEPARATION:
        raise RuntimeError("Accepted endpoints violate the minimum Chebyshev separation.")
    if ground_truth.optimal_path is None or not ground_truth.reachable_mask[candidate.start]:
        raise RuntimeError("Accepted map lacks a valid start-to-goal path.")
    if ground_truth.distance_map.dtype != np.float64:
        raise RuntimeError("Dijkstra labels must use float64 costs.")
    if not np.array_equal(np.isfinite(ground_truth.distance_map), ground_truth.reachable_mask):
        raise RuntimeError("Reachability mask does not match finite Dijkstra labels.")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_for_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    densities = np.asarray([record["actual_density_percent"] for record in records], dtype=np.float64)
    separations = np.asarray([record["start_goal_chebyshev"] for record in records], dtype=np.float64)
    path_distances = np.asarray([record["shortest_path_distance"] for record in records], dtype=np.float64)
    rectangle_counts = np.asarray([record["rectangle_count"] for record in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "actual_density_percent": {
            "min": float(densities.min()), "mean": float(densities.mean()),
            "median": float(np.median(densities)), "max": float(densities.max()),
        },
        "start_goal_chebyshev": {
            "min": float(separations.min()), "mean": float(separations.mean()),
            "median": float(np.median(separations)), "max": float(separations.max()),
        },
        "shortest_path_distance_unit_moves": {
            "min": float(path_distances.min()), "mean": float(path_distances.mean()),
            "median": float(np.median(path_distances)), "max": float(path_distances.max()),
        },
        "rectangle_count": {
            "min": int(rectangle_counts.min()), "mean": float(rectangle_counts.mean()),
            "median": float(np.median(rectangle_counts)), "max": int(rectangle_counts.max()),
        },
        "density_histogram_1_percent_bins": {
            str(index): int(np.count_nonzero((densities >= index) & (densities < index + 1)))
            for index in range(0, 51)
        },
    }


def generate_dataset(
    output_directory: str | Path,
    *,
    config: RectangleGenerationConfig = RectangleGenerationConfig(),
    split_counts: dict[str, dict[str, int]] = SPLIT_COUNTS,
) -> dict[str, Any]:
    """Generate, validate, serialize, and seal the complete new dataset.

    The sealed OOD archive is validated in memory and written last. After that
    point, this function only hashes and makes the file read-only; it never
    reloads the archive.
    """

    output_root = Path(output_directory)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cost_config = config.movement
    all_layout_hashes: dict[str, str] = {}
    manifest_records: list[dict[str, Any]] = []
    generation_stats: dict[str, dict[str, Any]] = {}
    archives: dict[str, Path] = {}

    try:
        for split_name, distribution_counts in split_counts.items():
            arrays = _new_split_arrays()
            split_records: list[dict[str, Any]] = []
            for distribution, requested_count in distribution_counts.items():
                group_seed = split_group_seed(config.base_seed, split_name, distribution)
                stats = {
                    "requested_maps": int(requested_count),
                    "attempted_maps": 0,
                    "accepted_maps": 0,
                    "rejected_maps": 0,
                    "density_target_failures": 0,
                    "connectivity_failures": 0,
                    "endpoint_selection_failures": 0,
                    "duplicate_layout_failures": 0,
                    "acceptance_rate": 0.0,
                    "group_seed": group_seed,
                    "generator_config": {
                        "rectangle_width_uniform_inclusive": [2, 14],
                        "rectangle_height_uniform_inclusive": [2, 14],
                        "rectangle_position": "uniform among valid top-left coordinates",
                        "overlap": "allowed; obstacle union counted once for density",
                        "movement": asdict(cost_config),
                        "density_interval_percent": list(DENSITY_INTERVALS_PERCENT[distribution]),
                    },
                }
                generation_stats[f"{split_name}/{distribution}"] = stats
                sample_index = 0
                while sample_index < requested_count:
                    map_seed = sample_seed_for(group_seed, sample_index)
                    accepted: RectangleCandidate | None = None
                    for attempt_index in range(config.max_candidate_attempts_per_map):
                        stats["attempted_maps"] += 1
                        try:
                            candidate = generate_candidate(
                                distribution, map_seed, attempt_index, config
                            )
                        except CandidateRejected as rejection:
                            stats["rejected_maps"] += 1
                            if rejection.reason == "density_target":
                                stats["density_target_failures"] += 1
                            elif rejection.reason == "connectivity":
                                stats["connectivity_failures"] += 1
                            elif rejection.reason == "endpoint_selection":
                                stats["endpoint_selection_failures"] += 1
                            else:
                                raise
                            continue

                        if candidate.layout_hash in all_layout_hashes:
                            stats["rejected_maps"] += 1
                            stats["duplicate_layout_failures"] += 1
                            continue
                        truth = compute_ground_truth(
                            candidate.grid, candidate.start, candidate.goal, cost_config
                        )
                        _validate_map_record(candidate, distribution, truth)
                        if truth.optimal_path is None:
                            raise RuntimeError("Connectivity validation and Dijkstra disagree.")
                        _append_candidate(
                            arrays, candidate, split_name, distribution, sample_index,
                            truth.distance_map, truth.reachable_mask, truth.optimal_path,
                        )
                        all_layout_hashes[candidate.layout_hash] = f"{split_name}/{distribution}/{sample_index}"
                        record = {
                            "map_id": f"{split_name}_{distribution}_{sample_index:04d}",
                            "split": split_name,
                            "distribution": distribution,
                            "generation_seed": map_seed,
                            "accepted_generation_attempt": attempt_index,
                            "actual_obstacle_density": candidate.actual_density,
                            "actual_density_percent": candidate.actual_density * 100.0,
                            "rectangle_count": len(candidate.rectangles),
                            "rectangles_top_left_height_width": [list(rectangle) for rectangle in candidate.rectangles],
                            "overlap_placements": candidate.overlap_placements,
                            "start": list(candidate.start),
                            "goal": list(candidate.goal),
                            "start_goal_chebyshev": candidate.start_goal_separation,
                            "shortest_path_distance": float(truth.optimal_path_cost),
                            "reachable": True,
                            "layout_hash": candidate.layout_hash,
                            "endpoint_hash": hashlib.blake2b(
                                f"{candidate.layout_hash}:{candidate.start}:{candidate.goal}".encode(),
                                digest_size=20,
                            ).hexdigest(),
                            "map_shape": [64, 64],
                            "channel_order": list(CHANNEL_ORDER),
                            "generator_version": GENERATOR_VERSION,
                        }
                        manifest_records.append(record)
                        split_records.append(record)
                        stats["accepted_maps"] += 1
                        sample_index += 1
                        accepted = candidate
                        break
                    if accepted is None:
                        stats["acceptance_rate"] = (
                            stats["accepted_maps"] / stats["attempted_maps"]
                            if stats["attempted_maps"] else 0.0
                        )
                        _atomic_json(output_root / "partial_generation_stats.json", generation_stats)
                        raise RuntimeError(
                            f"Could not produce {split_name}/{distribution} map {sample_index}; "
                            f"attempts={stats['attempted_maps']}, accepted={stats['accepted_maps']}, "
                            f"rejections={stats['rejected_maps']}. No generation-rule changes were made."
                        )
                stats["acceptance_rate"] = stats["accepted_maps"] / stats["attempted_maps"]

            packed = _packed_arrays(arrays)
            _validate_split_arrays(packed, split_records)
            archive_path = output_root / f"{split_name}.npz"
            _atomic_npz(archive_path, packed)
            archives[split_name] = archive_path

        manifest = {
            "run_id": RUN_ID,
            "base_seed": config.base_seed,
            "generator_version": GENERATOR_VERSION,
            "map_shape": list(MAP_SHAPE),
            "channel_order": list(CHANNEL_ORDER),
            "density_intervals_percent": {
                key: list(value) for key, value in DENSITY_INTERVALS_PERCENT.items()
            },
            "movement": asdict(cost_config),
            "generation_config": {
                **asdict(config),
                "movement": asdict(cost_config),
                "rectangle_size_distribution": "width and height independently uniform integer 2..14",
                "rectangle_placement": "uniform valid top-left coordinate; rectangles may overlap",
                "density": "union occupied cells / 4096, checked against the assigned interval",
            },
            "splits": {name: dict(counts) for name, counts in split_counts.items()},
            "total_maps": len(manifest_records),
            "maps": manifest_records,
        }
        _atomic_json(output_root / "manifest.json", manifest)

        by_split_distribution: dict[str, dict[str, Any]] = {}
        for split_name, distribution_counts in split_counts.items():
            for distribution in distribution_counts:
                selected = [
                    record for record in manifest_records
                    if record["split"] == split_name and record["distribution"] == distribution
                ]
                by_split_distribution[f"{split_name}/{distribution}"] = _summary_for_distribution(selected)
        report = {
            "run_id": RUN_ID,
            "map_count": len(manifest_records),
            "unique_layout_hashes": len(all_layout_hashes),
            "unique_endpoint_hashes": len({record["endpoint_hash"] for record in manifest_records}),
            "generation_stats": generation_stats,
            "split_distribution_summaries": by_split_distribution,
            "all_density_groups": {
                distribution: _summary_for_distribution(
                    [record for record in manifest_records if record["distribution"] == distribution]
                )
                for distribution in DENSITY_INTERVALS_PERCENT
            },
            "archive_paths": {name: path.name for name, path in archives.items()},
            "validation": {
                "all_maps_64x64": True,
                "all_channel_tensors_3x64x64": True,
                "all_density_intervals_pass": True,
                "all_start_goal_paths_reachable": True,
                "all_minimum_chebyshev_separations_pass": True,
                "all_distance_maps_float64_with_infinity_masks": True,
                "no_duplicate_obstacle_layouts_across_any_split": True,
                "no_duplicate_endpoint_instances": report_endpoint_unique(manifest_records),
                "generator_settings_shared_across_density_groups": True,
                "ood_pilot_and_sealed_test_layouts_disjoint": True,
                "ood_test_opened_only_during_generation_and_validation": True,
            },
        }
        _atomic_json(output_root / "generation_report.json", report)

        sealed_path = archives["ood_test_sealed"]
        seal = {
            "run_id": RUN_ID,
            "archive": sealed_path.name,
            "sha256": _sha256(sealed_path),
            "map_count": int(split_counts["ood_test_sealed"]["D4"]
                              + split_counts["ood_test_sealed"]["D5"]
                              + split_counts["ood_test_sealed"]["D6"]),
            "sealed_after_generation_and_in_memory_validation": True,
            "read_only": True,
            "seal_created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        sealed_path.chmod(0o444)
        _atomic_json(output_root / "ood_test_seal.json", seal)
        return report
    except Exception:
        _atomic_json(output_root / "partial_generation_stats.json", generation_stats)
        raise


def report_endpoint_unique(records: list[dict[str, Any]]) -> bool:
    return len({record["endpoint_hash"] for record in records}) == len(records)


def _validate_split_arrays(packed: dict[str, np.ndarray], records: list[dict[str, Any]]) -> None:
    maps = packed["maps"]
    inputs = packed["inputs"]
    if maps.ndim != 3 or maps.shape[1:] != MAP_SHAPE or inputs.shape != (len(maps), 3, 64, 64):
        raise RuntimeError("Split archive has invalid map or input-channel dimensions.")
    if not np.array_equal(inputs[:, 0], maps):
        raise RuntimeError("Stored obstacle channel differs from the canonical map array.")
    if not np.array_equal(inputs[:, 1].sum(axis=(1, 2)), np.ones(len(maps), dtype=np.uint64)):
        raise RuntimeError("Stored start channels must be one-hot.")
    if not np.array_equal(inputs[:, 2].sum(axis=(1, 2)), np.ones(len(maps), dtype=np.uint64)):
        raise RuntimeError("Stored goal channels must be one-hot.")
    for index, record in enumerate(records):
        if _layout_hash(maps[index]) != record["layout_hash"]:
            raise RuntimeError("Canonical obstacle-layout hash does not match the archive row.")
        if not np.isfinite(packed["distance_maps"][index][packed["reachable_masks"][index]]).all():
            raise RuntimeError("Reachable ground-truth distances must be finite.")
        if np.isfinite(packed["distance_maps"][index][~packed["reachable_masks"][index]]).any():
            raise RuntimeError("Unreachable distances must be infinity and masked explicitly.")
    if len({record["layout_hash"] for record in records}) != len(records):
        raise RuntimeError("Duplicate obstacle layouts occur inside a split.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="data/64x64_rectangles_v1",
        help="New dataset directory; must be absent or empty.",
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--max-candidate-attempts", type=int, default=1000)
    parser.add_argument("--maximum-rectangles", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RectangleGenerationConfig(
        base_seed=args.base_seed,
        max_candidate_attempts_per_map=args.max_candidate_attempts,
        maximum_rectangles_per_candidate=args.maximum_rectangles,
    )
    result = generate_dataset(args.output_directory, config=config)
    print(json.dumps({
        "run_id": result["run_id"],
        "map_count": result["map_count"],
        "unique_layout_hashes": result["unique_layout_hashes"],
        "output_directory": str(Path(args.output_directory).resolve()),
        "generation_stats": result["generation_stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
