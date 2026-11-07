"""Checkpoint loading and prediction helpers for conventional heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from algorithms.movement import Coord
from heuristics.data import build_heuristic_input
from models.heuristic_cnn import HeuristicCNN, HeuristicCNNConfig, decode_heuristic_output

try:
    import torch
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]


def load_heuristic_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Restore the best H1/H2/H3 model and its configuration safely."""

    if torch is None:
        raise ImportError("Heuristic checkpoint loading requires PyTorch.")
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a heuristic model state.")
    config_values = checkpoint.get("model_config")
    if not isinstance(config_values, dict):
        raise ValueError("Checkpoint does not contain the CNN architecture configuration.")
    model_config = HeuristicCNNConfig(**config_values)
    model = HeuristicCNN(
        model_config,
        num_classes=checkpoint.get("num_classes"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_heuristic_map(
    model: Any,
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    method: str,
    device: str = "cpu",
) -> np.ndarray:
    """Return the model's raw, uncorrected conventional heuristic map."""

    if torch is None:
        raise ImportError("Heuristic prediction requires PyTorch.")
    features = build_heuristic_input(grid, start, goal)
    model = model.to(device)
    model.eval()
    inputs = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        raw_output = model(inputs)
        prediction = decode_heuristic_output(raw_output, method)[0]
    return prediction.detach().cpu().numpy().astype(np.float64, copy=True)


def fixed_goal_heuristic(
    prediction_map: np.ndarray,
    fixed_goal: Coord,
) -> Callable[[Coord, Coord], float]:
    """Adapt a raw CNN map to A*'s ``(node, goal) -> h`` callable interface.

    The adapter does not clip or correct predictions. Its values carry no
    optimality guarantee unless admissibility/consistency is established
    independently for that prediction map.
    """

    values = np.asarray(prediction_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("prediction_map must be a finite two-dimensional array.")
    if np.any(values < 0.0):
        raise ValueError("Conventional heuristic predictions must be non-negative.")
    if len(fixed_goal) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in fixed_goal
    ):
        raise ValueError("fixed_goal must contain two integer coordinates.")
    goal = (int(fixed_goal[0]), int(fixed_goal[1]))
    if not (0 <= goal[0] < values.shape[0] and 0 <= goal[1] < values.shape[1]):
        raise ValueError("fixed_goal lies outside prediction_map.")

    def heuristic(node: Coord, requested_goal: Coord) -> float:
        if tuple(requested_goal) != goal:
            raise ValueError("This heuristic map is conditioned on a different goal.")
        if len(node) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in node
        ):
            raise ValueError("Node coordinates must be two integers.")
        if not (0 <= node[0] < values.shape[0] and 0 <= node[1] < values.shape[1]):
            raise ValueError(f"Node {node!r} lies outside prediction_map.")
        return float(values[node])

    heuristic.__name__ = "cnn_predicted_heuristic"
    return heuristic
