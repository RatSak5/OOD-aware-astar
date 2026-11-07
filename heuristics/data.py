"""Map-wise feature and supervision batches for learned A* heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from algorithms.movement import (
    DEFAULT_COST_CONFIG,
    get_neighbors,
    get_transition_cost,
    validate_map,
)
from data.dataset import DatasetSample

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True, slots=True)
class HeuristicBatch:
    """Tensorized maps, cost-to-go targets, and directed grid edges."""

    inputs: Tensor
    targets: Tensor
    valid_mask: Tensor
    edge_indices: Tensor
    edge_costs: Tensor
    goal_coordinates: Tensor
    distributions: tuple[str, ...]


def build_heuristic_input(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Create obstacle, start, and goal channels without exposing h-star."""

    obstacle_map = validate_map(grid)
    height, width = obstacle_map.shape
    channels = np.zeros((3, height, width), dtype=np.float32)
    channels[0] = obstacle_map.astype(np.float32)
    for label, coordinate in (("start", start), ("goal", goal)):
        if len(coordinate) != 2:
            raise ValueError(f"{label} must be a (row, column) coordinate.")
        row, col = coordinate
        if (
            isinstance(row, (bool, np.bool_))
            or isinstance(col, (bool, np.bool_))
            or not isinstance(row, (int, np.integer))
            or not isinstance(col, (int, np.integer))
        ):
            raise ValueError(f"{label} coordinates must be integers.")
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
        if obstacle_map[row, col] != 0:
            raise ValueError(f"{label} must be on a traversable cell.")
        channels[1 if label == "start" else 2, row, col] = 1.0
    return channels


def collate_heuristic_samples(
    samples: Sequence[DatasetSample],
    *,
    device: str = "cpu",
) -> HeuristicBatch:
    """Build a map-level batch and all valid directed transition edges.

    Maps in a batch must share dimensions. Cells are supervised only when
    traversable and reachable to the sample's goal; masked targets are replaced
    by zero before tensor arithmetic so infinities cannot contaminate losses.
    Every undirected legal adjacency is emitted in both directed orientations.
    """

    if torch is None:
        raise ImportError("Heuristic datasets require PyTorch tensor support.")
    if not samples:
        raise ValueError("Cannot collate an empty heuristic sample batch.")
    shape = samples[0].grid_map.grid.shape
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    goals: list[tuple[int, int]] = []
    edge_rows: list[tuple[int, int, int]] = []
    edge_cost_values: list[float] = []

    for batch_index, sample in enumerate(samples):
        grid = validate_map(sample.grid_map.grid)
        if grid.shape != shape:
            raise ValueError("All maps in a tensor batch must share dimensions.")
        if sample.cost_config != DEFAULT_COST_CONFIG:
            raise ValueError("Heuristic supervision requires the shared unit_move_count cost model.")
        distances = np.asarray(sample.ground_truth.distance_map, dtype=np.float64)
        reachable = np.asarray(sample.ground_truth.reachable_mask, dtype=bool)
        if distances.shape != shape or reachable.shape != shape:
            raise ValueError("Ground-truth distance and reachability maps must match the grid.")
        valid = (grid == 0) & reachable & np.isfinite(distances)
        if not valid.any():
            raise ValueError("A map must contain at least one reachable traversable cell.")
        if not valid[sample.grid_map.goal]:
            raise ValueError("The goal must be included in the valid supervision mask.")
        if not np.allclose(distances[valid], np.rint(distances[valid]), atol=1e-9, rtol=0.0):
            raise ValueError("Unit-move ground-truth distances must be integer-valued.")

        inputs.append(build_heuristic_input(grid, sample.grid_map.start, sample.grid_map.goal))
        targets.append(np.where(valid, distances, 0.0).astype(np.float32))
        masks.append(valid)
        goals.append(sample.grid_map.goal)

        for row, col in np.argwhere(valid):
            source = (int(row), int(col))
            source_index = source[0] * shape[1] + source[1]
            for neighbor in get_neighbors(grid, source, sample.cost_config):
                if not valid[neighbor]:
                    continue
                destination_index = neighbor[0] * shape[1] + neighbor[1]
                cost = get_transition_cost(grid, source, neighbor, sample.cost_config)
                if not np.isfinite(cost):
                    raise RuntimeError("The shared neighbor generator returned an invalid edge.")
                edge_rows.append((batch_index, source_index, destination_index))
                edge_cost_values.append(float(cost))

    edge_index_array = np.asarray(edge_rows, dtype=np.int64).reshape(-1, 3)
    edge_cost_array = np.asarray(edge_cost_values, dtype=np.float32)
    return HeuristicBatch(
        inputs=torch.as_tensor(np.stack(inputs), dtype=torch.float32, device=device),
        targets=torch.as_tensor(np.stack(targets), dtype=torch.float32, device=device),
        valid_mask=torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
        edge_indices=torch.as_tensor(edge_index_array, dtype=torch.long, device=device),
        edge_costs=torch.as_tensor(edge_cost_array, dtype=torch.float32, device=device),
        goal_coordinates=torch.as_tensor(np.asarray(goals), dtype=torch.long, device=device),
        distributions=tuple(sample.grid_map.distribution for sample in samples),
    )
