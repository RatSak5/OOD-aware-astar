"""Dataset generation, leakage checks, and pickle-free NPZ serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, GroundTruthCostConfig
from data.ground_truth import GroundTruth, compute_ground_truth
from data.map_generator import GridMap, MapGenerationConfig, generate_map

DEFAULT_SPLIT_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "train": ("D1", "D2", "D3"),
    "validation": ("D1", "D2", "D3"),
    "test": ("D4", "D5", "D6"),
}
NPZ_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One generated map paired with Dijkstra ground truth and its configs."""

    grid_map: GridMap
    ground_truth: GroundTruth
    generation_config: MapGenerationConfig
    cost_config: GroundTruthCostConfig


def make_dataset_sample(
    generation_config: MapGenerationConfig,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> DatasetSample:
    """Generate a map and calculate its full ground-truth record."""

    generated = generate_map(generation_config, cost_config)
    ground_truth = compute_ground_truth(
        generated.grid, generated.start, generated.goal, cost_config
    )
    if ground_truth.optimal_path is None and not generation_config.allow_unreachable:
        raise RuntimeError("Map generation returned disconnected endpoints unexpectedly.")
    return DatasetSample(generated, ground_truth, generation_config, cost_config)


def map_fingerprint(grid: np.ndarray) -> str:
    """Return a stable fingerprint of the obstacle layout, independent of endpoints."""

    array = np.asarray(grid, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("A map fingerprint requires a two-dimensional array.")
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def assert_no_map_leakage(splits: Mapping[str, Sequence[DatasetSample]]) -> None:
    """Raise when the same obstacle layout appears in more than one split.

    Layouts are compared without start/goal coordinates, so changing endpoints
    on a duplicate map does not evade the train/validation/test leakage check.
    """

    seen: dict[str, tuple[str, int]] = {}
    for split_name, samples in splits.items():
        for index, sample in enumerate(samples):
            fingerprint = map_fingerprint(sample.grid_map.grid)
            if fingerprint in seen:
                previous_split, previous_index = seen[fingerprint]
                raise ValueError(
                    "Duplicate map layout detected: "
                    f"{previous_split}[{previous_index}] and {split_name}[{index}]."
                )
            seen[fingerprint] = (split_name, index)


def _derived_seed(base_seed: int, split_index: int, sample_index: int, retry: int) -> int:
    """Derive a stable independent 64-bit seed for a split sample attempt."""

    state = np.random.SeedSequence([base_seed, split_index, sample_index, retry])
    return int(state.generate_state(1, dtype=np.uint64)[0])


def generate_standard_splits(
    train_count: int,
    validation_count: int,
    ood_test_count: int,
    *,
    base_seed: int = 0,
    map_config: MapGenerationConfig | None = None,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    duplicate_retry_limit: int = 10_000,
) -> dict[str, list[DatasetSample]]:
    """Create independent train, validation, and OOD test samples.

    Training and validation cycle over D1-D3. The OOD test split cycles over
    D4-D6. Every candidate seed is derived from ``base_seed`` and the split,
    sample, and duplicate-retry indices. Duplicate obstacle layouts are
    rejected across all three splits, even if their endpoints differ.
    """

    counts = (train_count, validation_count, ood_test_count)
    if any(count < 0 for count in counts):
        raise ValueError("Split sample counts must be non-negative.")
    if not 0 <= base_seed < 2**64:
        raise ValueError("base_seed must be an unsigned 64-bit integer.")
    if duplicate_retry_limit < 1:
        raise ValueError("duplicate_retry_limit must be positive.")
    template = map_config or MapGenerationConfig()
    split_counts = dict(zip(DEFAULT_SPLIT_DISTRIBUTIONS, counts))
    splits: dict[str, list[DatasetSample]] = {name: [] for name in split_counts}
    seen: set[str] = set()

    for split_index, (split_name, sample_count) in enumerate(split_counts.items()):
        distributions = DEFAULT_SPLIT_DISTRIBUTIONS[split_name]
        for sample_index in range(sample_count):
            distribution = distributions[sample_index % len(distributions)]
            for retry in range(duplicate_retry_limit):
                seed = _derived_seed(base_seed, split_index, sample_index, retry)
                sample_config = replace(
                    template,
                    distribution=distribution,
                    seed=seed,
                )
                sample = make_dataset_sample(sample_config, cost_config)
                fingerprint = map_fingerprint(sample.grid_map.grid)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                splits[split_name].append(sample)
                break
            else:
                raise RuntimeError(
                    f"Could not generate a unique layout for {split_name}[{sample_index}] "
                    f"after {duplicate_retry_limit} attempts."
                )

    assert_no_map_leakage(splits)
    return splits


def _path_arrays(samples: Sequence[DatasetSample]) -> tuple[np.ndarray, np.ndarray]:
    flattened: list[np.ndarray] = []
    offsets = [0]
    for sample in samples:
        if sample.ground_truth.optimal_path is None:
            coordinates = np.empty((0, 2), dtype=np.int32)
        else:
            coordinates = np.asarray(sample.ground_truth.optimal_path, dtype=np.int32)
        flattened.append(coordinates)
        offsets.append(offsets[-1] + len(coordinates))
    if flattened:
        flat_paths = np.concatenate(flattened, axis=0) if offsets[-1] else np.empty((0, 2), dtype=np.int32)
    else:
        flat_paths = np.empty((0, 2), dtype=np.int32)
    return flat_paths, np.asarray(offsets, dtype=np.int64)


def save_dataset_npz(path: str | Path, samples: Sequence[DatasetSample]) -> Path:
    """Write maps and all requested ground truth to a compressed NPZ archive.

    Variable-length paths are stored in one flat coordinate array plus offsets;
    all arrays are numeric or Unicode so the archive can be loaded with
    ``allow_pickle=False``. The destination's parent directory is created.
    """

    if not samples:
        raise ValueError("Cannot save an empty dataset split.")
    expected_shape = samples[0].grid_map.grid.shape
    for sample in samples:
        if sample.grid_map.grid.shape != expected_shape:
            raise ValueError("All maps in one NPZ split must have the same dimensions.")
        if sample.cost_config != samples[0].cost_config:
            raise ValueError("All samples in one NPZ split must share one cost configuration.")

    flat_paths, path_offsets = _path_arrays(samples)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generation_json = np.asarray(
        [json.dumps(asdict(sample.generation_config), sort_keys=True) for sample in samples],
        dtype=np.str_,
    )
    cost_json = json.dumps(asdict(samples[0].cost_config), sort_keys=True)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(NPZ_SCHEMA_VERSION, dtype=np.int32),
            maps=np.stack([sample.grid_map.grid for sample in samples]).astype(np.uint8),
            starts=np.asarray([sample.grid_map.start for sample in samples], dtype=np.int32),
            goals=np.asarray([sample.grid_map.goal for sample in samples], dtype=np.int32),
            distribution_labels=np.asarray(
                [sample.grid_map.distribution for sample in samples], dtype=np.str_
            ),
            structures=np.asarray([sample.grid_map.structure for sample in samples], dtype=np.str_),
            seeds=np.asarray([sample.grid_map.seed for sample in samples], dtype=np.uint64),
            distance_maps=np.stack([sample.ground_truth.distance_map for sample in samples]).astype(
                np.float64
            ),
            optimal_path_coordinates=flat_paths,
            path_offsets=path_offsets,
            optimal_path_costs=np.asarray(
                [sample.ground_truth.optimal_path_cost for sample in samples], dtype=np.float64
            ),
            reachability_masks=np.stack(
                [sample.ground_truth.reachable_mask for sample in samples]
            ).astype(np.bool_),
            cost_model_config=np.asarray(cost_json, dtype=np.str_),
            generation_configs=generation_json,
        )
    return destination


def load_dataset_npz(path: str | Path) -> dict[str, object]:
    """Load a dataset NPZ without enabling pickle, returning copied arrays."""

    with np.load(Path(path), allow_pickle=False) as archive:
        result: dict[str, object] = {
            name: archive[name].copy() for name in archive.files
        }
    version = int(result["schema_version"])
    if version != NPZ_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema version {version}.")
    result["cost_model_config"] = json.loads(str(result["cost_model_config"]))
    result["generation_configs"] = tuple(
        json.loads(str(item)) for item in result["generation_configs"]
    )
    return result


def save_splits_npz(
    output_directory: str | Path,
    splits: Mapping[str, Sequence[DatasetSample]],
) -> dict[str, Path]:
    """Leakage-check and save each non-empty split as ``<name>.npz``."""

    assert_no_map_leakage(splits)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, Path] = {}
    for split_name, samples in splits.items():
        if not samples:
            continue
        destination = output_root / f"{split_name}.npz"
        destinations[split_name] = save_dataset_npz(destination, samples)
    return destinations
