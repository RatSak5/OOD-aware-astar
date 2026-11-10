"""CNN guidance inference and end-to-end Neural A* planning wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from pathlib import Path

import numpy as np

from algorithms.movement import Coord, DEFAULT_COST_CONFIG, GroundTruthCostConfig
from neural_astar.guidance import build_encoder_input
from neural_astar.model import NeuralAStarUNet, make_guidance_tensor
from neural_astar.search import NeuralAStarResult, run_neural_astar

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class NeuralAStarInferenceResult:
    """End-to-end inference result with model and search time kept separate."""

    guidance_map: np.ndarray
    search: NeuralAStarResult
    model_inference_seconds: float
    total_seconds: float


def load_neural_astar_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: Any | None = None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a saved Neural A* checkpoint for later guidance inference."""

    if torch is None:
        raise ImportError("Neural A* checkpoint loading requires PyTorch.")
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a Neural A* state dictionary.")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint must be a raw Neural A* state dictionary or contain "
            "'model_state_dict' or 'state_dict'."
        )
    network = (model or NeuralAStarUNet()).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return network, checkpoint


def predict_guidance(
    model: Any,
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Run the CNN on the paper's two input channels and return Phi and time."""

    if torch is None:
        raise ImportError("Neural A* model inference requires PyTorch.")
    features = build_encoder_input(grid, start, goal)
    model = model.to(device)
    model.eval()
    batch = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    started = perf_counter()
    with torch.no_grad():
        output = make_guidance_tensor(model(batch))
    elapsed = perf_counter() - started
    guidance = output[0, 0].detach().cpu().numpy().astype(np.float64, copy=True)
    return guidance, elapsed


def plan_with_neural_astar(
    model: Any,
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    *,
    device: str = "cpu",
    temperature: float | None = None,
    tie_break_epsilon: float = 0.001,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> NeuralAStarInferenceResult:
    """Predict Phi and run Neural A*, preserving inference and search timing."""

    total_started = perf_counter()
    guidance, inference_seconds = predict_guidance(
        model, grid, start, goal, device=device
    )
    search = run_neural_astar(
        grid,
        start,
        goal,
        guidance,
        temperature=temperature,
        tie_break_epsilon=tie_break_epsilon,
        cost_config=cost_config,
    )
    return NeuralAStarInferenceResult(
        guidance_map=guidance,
        search=search,
        model_inference_seconds=inference_seconds,
        total_seconds=perf_counter() - total_started,
    )
