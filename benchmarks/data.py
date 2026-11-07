"""Dataset creation, split sealing, and structural distribution diagnostics."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig
from data.dataset import (
    DatasetSample,
    assert_no_map_leakage,
    generate_standard_splits,
    load_dataset_npz,
    make_dataset_sample,
    map_fingerprint,
    save_dataset_npz,
)
from data.ground_truth import GroundTruth
from data.map_generator import MapGenerationConfig, GridMap


SPLIT_FILENAMES = {
    "train": "train.npz",
    "validation": "validation_calibration.npz",
    "id_holdout": "id_holdout.npz",
    "ood_test": "ood_test_sealed.npz",
}


def load_dataset_samples_npz(path: str | Path) -> list[DatasetSample]:
    """Reconstruct typed dataset records from the repository's safe NPZ format."""

    arrays = load_dataset_npz(path)
    maps = np.asarray(arrays["maps"], dtype=np.uint8)
    starts = np.asarray(arrays["starts"], dtype=np.int64)
    goals = np.asarray(arrays["goals"], dtype=np.int64)
    labels = np.asarray(arrays["distribution_labels"])
    structures = np.asarray(arrays["structures"])
    seeds = np.asarray(arrays["seeds"], dtype=np.uint64)
    distances = np.asarray(arrays["distance_maps"], dtype=np.float64)
    reachability = np.asarray(arrays["reachability_masks"], dtype=bool)
    path_coordinates = np.asarray(arrays["optimal_path_coordinates"], dtype=np.int32)
    path_offsets = np.asarray(arrays["path_offsets"], dtype=np.int64)
    path_costs = np.asarray(arrays["optimal_path_costs"], dtype=np.float64)
    generation_configs = arrays["generation_configs"]
    cost_config = GroundTruthCostConfig(**dict(arrays["cost_model_config"]))

    count = len(maps)
    if not (
        len(starts)
        == len(goals)
        == len(labels)
        == len(structures)
        == len(seeds)
        == len(distances)
        == len(reachability)
        == len(path_costs)
        == len(generation_configs)
        == count
    ):
        raise ValueError("Dataset NPZ arrays have inconsistent sample counts.")
    if path_offsets.shape != (count + 1,) or path_offsets[0] != 0:
        raise ValueError("Dataset NPZ path offsets are malformed.")
    if path_offsets[-1] != len(path_coordinates) or np.any(np.diff(path_offsets) < 0):
        raise ValueError("Dataset NPZ path offsets do not match the path coordinates.")

    samples: list[DatasetSample] = []
    for index in range(count):
        start = tuple(int(value) for value in starts[index])
        goal = tuple(int(value) for value in goals[index])
        low, high = int(path_offsets[index]), int(path_offsets[index + 1])
        path_slice = path_coordinates[low:high]
        path = (
            tuple(tuple(int(value) for value in coordinate) for coordinate in path_slice)
            if len(path_slice)
            else None
        )
        grid_map = GridMap(
            grid=maps[index],
            start=start,
            goal=goal,
            distribution=str(labels[index]),
            seed=int(seeds[index]),
            structure=str(structures[index]),
        )
        ground_truth = GroundTruth(
            start=start,
            goal=goal,
            distance_map=distances[index].astype(np.float64, copy=True),
            optimal_path=path,
            optimal_path_cost=float(path_costs[index]),
            reachable_mask=reachability[index].copy(),
        )
        samples.append(
            DatasetSample(
                grid_map=grid_map,
                ground_truth=ground_truth,
                generation_config=MapGenerationConfig(**dict(generation_configs[index])),
                cost_config=cost_config,
            )
        )
    return samples


def _holdout_seed(base_seed: int, sample_index: int, retry: int) -> int:
    """Derive an independent stream for the extra D1-D3 ID holdout split."""

    sequence = np.random.SeedSequence([base_seed, 3, sample_index, retry])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _generate_id_holdout(
    count: int,
    template: MapGenerationConfig,
    *,
    base_seed: int,
    already_seen: set[str],
    cost_config: GroundTruthCostConfig,
    retry_limit: int = 10_000,
) -> list[DatasetSample]:
    if count % 3:
        raise ValueError("ID holdout count must divide evenly across D1-D3.")
    per_distribution = count // 3
    result: list[DatasetSample] = []
    for distribution_index, distribution in enumerate(("D1", "D2", "D3")):
        for within_distribution in range(per_distribution):
            sample_index = distribution_index * per_distribution + within_distribution
            for retry in range(retry_limit):
                seed = _holdout_seed(base_seed, sample_index, retry)
                sample = make_dataset_sample(
                    replace(template, distribution=distribution, seed=seed),
                    cost_config,
                )
                fingerprint = map_fingerprint(sample.grid_map.grid)
                if fingerprint in already_seen:
                    continue
                already_seen.add(fingerprint)
                result.append(sample)
                break
            else:
                raise RuntimeError(
                    f"Could not generate a unique ID holdout map for {distribution} "
                    f"index {within_distribution}."
                )
    return result


def _map_structure_metrics(sample: DatasetSample) -> dict[str, float]:
    """Calculate map-level density, obstacle clustering, and corridor metrics."""

    obstacles = np.asarray(sample.grid_map.grid, dtype=np.uint8)
    height, width = obstacles.shape
    density = float(obstacles.mean())
    edge_count = max(0, height * (width - 1)) + max(0, (height - 1) * width)
    adjacent_obstacle_pairs = 0
    if width > 1:
        adjacent_obstacle_pairs += int(np.count_nonzero(obstacles[:, :-1] & obstacles[:, 1:]))
    if height > 1:
        adjacent_obstacle_pairs += int(np.count_nonzero(obstacles[:-1, :] & obstacles[1:, :]))
    pair_rate = adjacent_obstacle_pairs / edge_count if edge_count else 0.0

    blocked = obstacles == 1
    blocked_neighbor_incidence = 0
    possible_neighbor_incidence = 0
    for center, neighbor in (
        (blocked[1:, :], blocked[:-1, :]),
        (blocked[:-1, :], blocked[1:, :]),
        (blocked[:, 1:], blocked[:, :-1]),
        (blocked[:, :-1], blocked[:, 1:]),
    ):
        possible_neighbor_incidence += int(center.sum())
        blocked_neighbor_incidence += int(np.count_nonzero(center & neighbor))
    obstacle_neighbor_fraction = (
        blocked_neighbor_incidence / possible_neighbor_incidence
        if possible_neighbor_incidence
        else 0.0
    )

    padded = np.pad(obstacles.astype(np.float64), 1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    local_density_std = float(windows.mean(axis=(-1, -2)).std())
    free_cells_mask = obstacles == 0
    all_free_row_fraction = float(np.all(free_cells_mask, axis=1).mean())
    all_free_column_fraction = float(np.all(free_cells_mask, axis=0).mean())
    full_span_corridor_fraction = (all_free_row_fraction + all_free_column_fraction) / 2.0

    free = obstacles == 0
    padded_free = np.pad(free, 1, constant_values=False)
    center = padded_free[1:-1, 1:-1]
    degree_map = np.zeros(obstacles.shape, dtype=np.int16)
    degree_map += padded_free[:-2, 1:-1]
    degree_map += padded_free[2:, 1:-1]
    degree_map += padded_free[1:-1, :-2]
    degree_map += padded_free[1:-1, 2:]
    degree_map += padded_free[:-2, :-2] & padded_free[:-2, 1:-1] & padded_free[1:-1, :-2]
    degree_map += padded_free[:-2, 2:] & padded_free[:-2, 1:-1] & padded_free[1:-1, 2:]
    degree_map += padded_free[2:, :-2] & padded_free[2:, 1:-1] & padded_free[1:-1, :-2]
    degree_map += padded_free[2:, 2:] & padded_free[2:, 1:-1] & padded_free[1:-1, 2:]
    legal_degrees = degree_map[center]
    free_degree_mean = float(legal_degrees.mean()) if legal_degrees.size else 0.0
    corridor_cell_fraction = (
        float(np.mean(legal_degrees <= 3)) if legal_degrees.size else 0.0
    )
    return {
        "obstacle_density": density,
        "obstacle_pair_rate_4_neighbor": float(pair_rate),
        "obstacle_neighbor_fraction": float(obstacle_neighbor_fraction),
        "local_density_std_3x3": local_density_std,
        "full_span_corridor_fraction": float(full_span_corridor_fraction),
        "free_legal_degree_mean": free_degree_mean,
        "free_cells_degree_le_3_fraction": corridor_cell_fraction,
        "free_cell_fraction": float(1.0 - density),
        "optimal_path_cost": float(sample.ground_truth.optimal_path_cost),
    }


def _mean_ci(values: np.ndarray, seed: int, repetitions: int = 2_000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(values), size=len(values))
        bootstrapped[index] = float(values[draw].mean())
    return (
        float(np.quantile(bootstrapped, 0.025)),
        float(np.quantile(bootstrapped, 0.975)),
    )


def summarize_distribution_statistics(
    splits: Mapping[str, Sequence[DatasetSample]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Summarize generated map structure without using search/model outcomes."""

    allowed = {"D1", "D2", "D3", "D4", "D5", "D6"}
    all_samples = [sample for split in splits.values() for sample in split]
    metrics_by_sample = [_map_structure_metrics(sample) for sample in all_samples]
    group_metrics: dict[str, dict[str, list[float]]] = {
        distribution: {} for distribution in sorted(allowed)
    }
    group_counts = {distribution: 0 for distribution in allowed}
    for sample, metrics in zip(all_samples, metrics_by_sample):
        distribution = sample.grid_map.distribution
        if distribution not in allowed:
            raise ValueError(f"Unexpected distribution {distribution!r} in generated splits.")
        group_counts[distribution] += 1
        for name, value in metrics.items():
            group_metrics[distribution].setdefault(name, []).append(value)

    by_distribution: dict[str, dict[str, Any]] = {}
    metric_names = sorted(metrics_by_sample[0]) if metrics_by_sample else []
    for distribution in sorted(allowed):
        values = group_metrics[distribution]
        if not group_counts[distribution]:
            continue
        by_distribution[distribution] = {
            "map_count": group_counts[distribution],
            "metrics": {},
        }
        for metric_index, metric_name in enumerate(metric_names):
            array = np.asarray(values[metric_name], dtype=np.float64)
            low, high = _mean_ci(array, seed + metric_index + 10_000 * int(distribution[1:]))
            by_distribution[distribution]["metrics"][metric_name] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "mean_ci95": [low, high],
            }

    family_comparisons: dict[str, Any] = {}
    for metric_index, metric_name in enumerate(metric_names):
        # Reuse the map-level diagnostics computed above. Recomputing them in
        # this comparison loop would repeat a full legal-neighbor scan many
        # times for each map without adding information.
        id_values = np.asarray(
            [
                value
                for distribution in ("D1", "D2", "D3")
                for value in group_metrics[distribution].get(metric_name, [])
            ],
            dtype=np.float64,
        )
        ood_values = np.asarray(
            [
                value
                for distribution in ("D4", "D5", "D6")
                for value in group_metrics[distribution].get(metric_name, [])
            ],
            dtype=np.float64,
        )
        if not len(id_values) or not len(ood_values):
            continue
        rng = np.random.default_rng(seed + 1_000_000 + metric_index)
        differences = np.empty(2_000, dtype=np.float64)
        for index in range(len(differences)):
            id_draw = id_values[rng.integers(0, len(id_values), size=len(id_values))]
            ood_draw = ood_values[rng.integers(0, len(ood_values), size=len(ood_values))]
            differences[index] = float(id_draw.mean() - ood_draw.mean())
        family_comparisons[metric_name] = {
            "id_mean": float(id_values.mean()),
            "ood_mean": float(ood_values.mean()),
            "id_minus_ood_mean_difference": float(id_values.mean() - ood_values.mean()),
            "difference_ci95": [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ],
        }
    return {
        "interpretation": (
            "Generated-map structural diagnostics only; no model or adaptive-policy "
            "selection uses these values."
        ),
        "by_distribution": by_distribution,
        "id_vs_ood_family_comparisons": family_comparisons,
    }


def _distribution_count(samples: Sequence[DatasetSample]) -> dict[str, int]:
    return {
        distribution: sum(sample.grid_map.distribution == distribution for sample in samples)
        for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")
    }


def prepare_datasets(
    config: Any,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate, leakage-check, save, reload, and summarize the approved splits.

    The OOD archive is written as ``ood_test_sealed.npz``. This function reads
    it only for the user-requested generation diagnostics; the training loader
    below never opens it. Training, model selection, and D95 calibration must
    finish before the evaluation loader releases the OOD samples.
    """

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    destinations = {name: root / filename for name, filename in SPLIT_FILENAMES.items()}
    manifest_path = root / "dataset_manifest.json"
    existing = {name: path.is_file() for name, path in destinations.items()}
    if not overwrite and manifest_path.exists():
        raise FileExistsError(
            f"Dataset artifacts already exist under {root}; pass overwrite=True to regenerate."
        )
    reuse_existing = bool(not overwrite and all(existing.values()))
    if not overwrite and any(existing.values()) and not reuse_existing:
        raise FileExistsError(
            f"Only some split archives exist under {root}; pass overwrite=True to regenerate."
        )

    if reuse_existing:
        reloaded = {
            split_name: load_dataset_samples_npz(path)
            for split_name, path in destinations.items()
        }
    else:
        standard = generate_standard_splits(
            config.train_count,
            config.validation_count,
            config.ood_test_count,
            base_seed=config.base_seed,
            map_config=config.map_generation_config,
            cost_config=DEFAULT_COST_CONFIG,
        )
        seen = {
            map_fingerprint(sample.grid_map.grid)
            for split in standard.values()
            for sample in split
        }
        id_holdout = _generate_id_holdout(
            config.id_holdout_count,
            config.map_generation_config,
            base_seed=config.base_seed,
            already_seen=seen,
            cost_config=DEFAULT_COST_CONFIG,
        )
        splits: dict[str, list[DatasetSample]] = {
            "train": standard["train"],
            "validation": standard["validation"],
            "id_holdout": id_holdout,
            "ood_test": standard["test"],
        }
        assert_no_map_leakage(splits)
        for split_name, samples in splits.items():
            save_dataset_npz(destinations[split_name], samples)
        # Verify that the serialized split can be reconstructed losslessly
        # before reporting it as ready for training/evaluation.
        reloaded = {
            split_name: load_dataset_samples_npz(path)
            for split_name, path in destinations.items()
        }
        for split_name in splits:
            expected = splits[split_name]
            actual = reloaded[split_name]
            if len(expected) != len(actual):
                raise RuntimeError(f"Reload changed the sample count for {split_name}.")
            for first, second in zip(expected, actual):
                if (
                    not np.array_equal(first.grid_map.grid, second.grid_map.grid)
                    or first.grid_map.start != second.grid_map.start
                    or first.grid_map.goal != second.grid_map.goal
                    or not np.array_equal(
                        first.ground_truth.distance_map, second.ground_truth.distance_map
                    )
                    or first.ground_truth.optimal_path != second.ground_truth.optimal_path
                ):
                    raise RuntimeError(f"Dataset serialization round-trip failed in {split_name}.")

    expected_counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "id_holdout": config.id_holdout_count,
        "ood_test": config.ood_test_count,
    }
    for split_name, expected_count in expected_counts.items():
        if len(reloaded[split_name]) != expected_count:
            raise ValueError(
                f"Saved {split_name} has {len(reloaded[split_name])} maps; "
                f"config requires {expected_count}."
            )
        if any(
            sample.grid_map.grid.shape != (config.height, config.width)
            for sample in reloaded[split_name]
        ):
            raise ValueError(f"Saved {split_name} contains maps with unexpected dimensions.")
    expected_by_split = {
        "train": {"D1": config.train_count // 3, "D2": config.train_count // 3, "D3": config.train_count // 3},
        "validation": {
            "D1": config.validation_count // 3,
            "D2": config.validation_count // 3,
            "D3": config.validation_count // 3,
        },
        "id_holdout": {
            "D1": config.id_holdout_count // 3,
            "D2": config.id_holdout_count // 3,
            "D3": config.id_holdout_count // 3,
        },
        "ood_test": {
            "D4": config.ood_test_count // 3,
            "D5": config.ood_test_count // 3,
            "D6": config.ood_test_count // 3,
        },
    }
    for split_name, expected_distribution_counts in expected_by_split.items():
        observed = {
            distribution: sum(
                sample.grid_map.distribution == distribution
                for sample in reloaded[split_name]
            )
            for distribution in ("D1", "D2", "D3", "D4", "D5", "D6")
        }
        if {name: count for name, count in observed.items() if count} != expected_distribution_counts:
            raise ValueError(
                f"Saved {split_name} distribution counts do not match the approved configuration."
            )
    assert_no_map_leakage(reloaded)
    seen = {
        map_fingerprint(sample.grid_map.grid)
        for split in reloaded.values()
        for sample in split
    }

    statistics = summarize_distribution_statistics(
        reloaded, seed=int(config.base_seed % (2**32))
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_config": config.to_dict(),
        "cost_model": asdict(DEFAULT_COST_CONFIG),
        "split_files": {name: path.name for name, path in destinations.items()},
        "split_counts": {name: len(items) for name, items in reloaded.items()},
        "distribution_counts": {
            name: _distribution_count(items) for name, items in reloaded.items()
        },
        "unique_obstacle_layouts": len(seen),
        "split_fingerprints_disjoint": True,
        "ood_test_sealed_until_training_and_calibration_complete": True,
        "distribution_statistics": statistics,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return {"manifest": manifest, "manifest_path": manifest_path, "files": destinations}


def refresh_distribution_statistics(
    data_directory: str | Path,
) -> dict[str, Any]:
    """Recompute structural diagnostics from existing archives without regen."""

    root = Path(data_directory)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {
        split_name: load_dataset_samples_npz(root / filename)
        for split_name, filename in SPLIT_FILENAMES.items()
    }
    assert_no_map_leakage(splits)
    manifest["distribution_statistics"] = summarize_distribution_statistics(
        splits,
        seed=int(manifest["benchmark_config"]["base_seed"] % (2**32)),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return manifest


def load_training_splits(
    data_directory: str | Path,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """Load only train and validation maps; never opens ID/OOD holdouts."""

    root = Path(data_directory)
    training = load_dataset_samples_npz(root / SPLIT_FILENAMES["train"])
    validation = load_dataset_samples_npz(root / SPLIT_FILENAMES["validation"])
    if {item.grid_map.distribution for item in training} != {"D1", "D2", "D3"}:
        raise ValueError("Training split must contain D1-D3 only.")
    if {item.grid_map.distribution for item in validation} != {"D1", "D2", "D3"}:
        raise ValueError("Validation/calibration split must contain D1-D3 only.")
    assert_no_map_leakage({"train": training, "validation": validation})
    return training, validation


def load_evaluation_splits(
    data_directory: str | Path,
    *,
    model_ready_manifest: str | Path,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """Release ID/OOD holdouts only after a completed model/calibration manifest."""

    ready_path = Path(model_ready_manifest)
    if not ready_path.is_file():
        raise PermissionError(
            "ID/OOD holdouts stay sealed until training, model selection, and D95 calibration complete."
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not ready.get("training_complete") or not ready.get("d95_calibrated"):
        raise PermissionError("The model-ready manifest does not confirm all required stages.")
    root = Path(data_directory)
    id_holdout = load_dataset_samples_npz(root / SPLIT_FILENAMES["id_holdout"])
    ood_test = load_dataset_samples_npz(root / SPLIT_FILENAMES["ood_test"])
    if {item.grid_map.distribution for item in id_holdout} != {"D1", "D2", "D3"}:
        raise ValueError("ID holdout must contain D1-D3 only.")
    if {item.grid_map.distribution for item in ood_test} != {"D4", "D5", "D6"}:
        raise ValueError("OOD test split must contain D4-D6 only.")
    assert_no_map_leakage({"id_holdout": id_holdout, "ood_test": ood_test})
    return id_holdout, ood_test
