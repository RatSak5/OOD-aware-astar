"""Input and validation helpers for the paper's node-wise guidance map."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from algorithms.movement import Coord, validate_map


def validate_guidance_map(phi: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Validate and return a float64 guidance map with values in ``[0, 1]``.

    The sigmoid-bounded range is the representation used by Neural A*.
    Values are validated, never clipped or normalized across the map.
    """

    guidance = np.asarray(phi, dtype=np.float64)
    if guidance.ndim != 2 or guidance.size == 0:
        raise ValueError("A guidance map must be a non-empty 2-D array.")
    if shape is not None and guidance.shape != shape:
        raise ValueError(f"Guidance map shape {guidance.shape} does not match map shape {shape}.")
    if not np.isfinite(guidance).all():
        raise ValueError("Guidance-map values must be finite.")
    if np.any(guidance < 0.0) or np.any(guidance > 1.0):
        raise ValueError("Neural A* guidance values must lie in [0, 1].")
    return guidance


def accumulate_guidance_cost(previous_g_phi: float, destination_phi: float) -> float:
    """Apply the paper's per-entry update ``G_phi' = G_phi + Phi(v)``.

    This is a guidance-search update, not the shared ground-truth transition
    cost function. The start node is initialized separately with ``G_phi=0``.
    """

    if not np.isfinite(previous_g_phi) or previous_g_phi < 0.0:
        raise ValueError("previous_g_phi must be finite and non-negative.")
    if not np.isfinite(destination_phi) or not 0.0 <= destination_phi <= 1.0:
        raise ValueError("destination_phi must be finite and lie in [0, 1].")
    return float(previous_g_phi + destination_phi)


def build_encoder_input(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
) -> np.ndarray:
    """Build the paper's two encoder channels as ``[passability, Vs + Vg]``.

    The project stores ``0=free, 1=obstacle`` while the paper's explicit
    binary input uses ``1=passable``; this function performs that documented
    inversion. The returned array has shape ``(2, height, width)``.
    """

    obstacle_map = validate_map(grid)
    height, width = obstacle_map.shape
    for label, coordinate in (("start", start), ("goal", goal)):
        if len(coordinate) != 2:
            raise ValueError(f"{label} must be a (row, column) coordinate.")
        row, col = coordinate
        if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
            raise ValueError(f"{label} coordinates must be integers.")
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
        if obstacle_map[row, col] != 0:
            raise ValueError(f"{label} cell must be traversable.")

    endpoints = np.zeros((height, width), dtype=np.float32)
    endpoints[start] += 1.0
    endpoints[goal] += 1.0
    return np.stack((1.0 - obstacle_map.astype(np.float32), endpoints), axis=0)


def path_to_mask(path: Sequence[Coord] | None, shape: tuple[int, int]) -> np.ndarray:
    """Convert an ordered path into the paper's binary target path map."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Path-mask shape must contain two positive dimensions.")
    mask = np.zeros(shape, dtype=np.float32)
    if path is None:
        return mask
    for row, col in path:
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise ValueError(f"Path coordinate {(row, col)!r} is outside shape {shape}.")
        mask[row, col] = 1.0
    return mask
