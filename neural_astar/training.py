"""Training utilities for the differentiable Neural A* objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from data.dataset import DatasetSample, assert_no_map_leakage
from neural_astar.differentiable import differentiable_neural_search
from neural_astar.guidance import build_encoder_input, path_to_mask
from neural_astar.model import NeuralAStarUNet, make_guidance_tensor

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        """Minimal console progress indicator when tqdm is not installed."""

        description = kwargs.get("desc", "Progress")
        total = len(iterable)

        def _iterate() -> Any:
            for index, item in enumerate(iterable, start=1):
                yield item
                print(f"\r{description}: {index}/{total}", end="", flush=True)
            if total:
                print()

        return _iterate()


@dataclass(frozen=True, slots=True)
class NeuralAStarTrainingConfig:
    """Reproducible training settings; defaults retain paper optimizer values."""

    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 0.001
    temperature: float | None = None
    tie_break_epsilon: float = 0.001
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.temperature is not None and (
            not np.isfinite(self.temperature) or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive when provided.")
        if not np.isfinite(self.tie_break_epsilon) or self.tie_break_epsilon < 0:
            raise ValueError("tie_break_epsilon must be finite and non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


def mean_l1_search_loss(closed_mask: Any, target_path_mask: Any) -> Any:
    """Paper's mean-L1 objective, Eq. (7): ``|C - P_bar|_1 / |V|``."""

    if torch is None:
        raise ImportError("Neural A* training loss requires PyTorch.")
    if tuple(closed_mask.shape) != tuple(target_path_mask.shape):
        raise ValueError("Closed-list output and target path map must have equal shapes.")
    return torch.abs(closed_mask - target_path_mask).sum() / closed_mask.numel()


def _validate_splits(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
) -> None:
    if not train_samples or not validation_samples:
        raise ValueError("Training and validation sample collections must both be non-empty.")
    for split_name, samples in (("training", train_samples), ("validation", validation_samples)):
        for index, sample in enumerate(samples):
            if sample.grid_map.distribution not in {"D1", "D2", "D3"}:
                raise ValueError(
                    f"{split_name} sample {index} uses {sample.grid_map.distribution}; "
                    "Neural A* training and validation use only D1-D3."
                )
            if sample.ground_truth.optimal_path is None:
                raise ValueError(f"{split_name} sample {index} has no reachable optimal path.")
    assert_no_map_leakage({"train": train_samples, "validation": validation_samples})


def _tensor_inputs(sample: DatasetSample, device: str) -> tuple[Any, Any]:
    if torch is None:
        raise ImportError("Neural A* training requires PyTorch.")
    grid = sample.grid_map.grid
    features = build_encoder_input(grid, sample.grid_map.start, sample.grid_map.goal)
    target = path_to_mask(sample.ground_truth.optimal_path, grid.shape)
    input_tensor = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    return input_tensor, target_tensor


def _epoch_loss(
    model: Any,
    samples: Sequence[DatasetSample],
    config: NeuralAStarTrainingConfig,
) -> float:
    if torch is None:
        raise ImportError("Neural A* training requires PyTorch.")
    total = 0.0
    with torch.no_grad():
        for sample in samples:
            inputs, target = _tensor_inputs(sample, config.device)
            phi = make_guidance_tensor(model(inputs))[0]
            search = differentiable_neural_search(
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                phi,
                temperature=config.temperature,
                tie_break_epsilon=config.tie_break_epsilon,
                cost_config=sample.cost_config,
            )
            total += float(mean_l1_search_loss(search.closed_mask, target).item())
    return total / len(samples)


def train_neural_astar(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    config: NeuralAStarTrainingConfig = NeuralAStarTrainingConfig(),
    model: Any | None = None,
    resume: bool = False,
) -> tuple[Any, list[dict[str, float]]]:
    """Train the guidance CNN and save best/last checkpoints plus CSV/config.

    Training uses the paper's differentiable closed-list output and Eq. (7)
    mean L1 loss, with RMSProp as in Sec. 4.3. Instance searches are evaluated
    serially and losses are averaged within each mini-batch before one update;
    this reproduces the paper's per-instance goal stopping semantics without
    requiring identical search lengths. OOD distributions D4-D6 are rejected.
    """

    if torch is None:
        raise ImportError("Neural A* CNN training requires PyTorch.")
    _validate_splits(train_samples, validation_samples)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    network = (model or NeuralAStarUNet()).to(config.device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=config.learning_rate)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **asdict(config),
        "method": "Yonetani et al. Neural A*",
        "optimizer": "RMSprop",
        "objective": "mean_l1(closed_mask, shortest_path_map)",
        "encoder": "VGG16-block U-Net with sigmoid; project-local initialization",
        "input_channels": ["passability=1-free", "start_plus_goal_one_hot"],
        "adaptations": [
            "shared eight-neighbor movement with no diagonal corner-cutting",
            "per-instance serial search inside each mini-batch",
        ],
    }
    (output_root / "training_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    history: list[dict[str, float]] = []
    best_validation = float("inf")
    rng = np.random.default_rng(config.seed)
    start_epoch = 1
    last_checkpoint_path = output_root / "last_checkpoint.pt"
    if resume and last_checkpoint_path.exists():
        saved = torch.load(last_checkpoint_path, map_location=config.device, weights_only=True)
        saved_config = dict(saved.get("config", {}))
        expected_config = dict(config_payload)
        saved_config.pop("device", None)
        expected_config.pop("device", None)
        if saved_config != expected_config:
            raise ValueError("Cannot resume Neural A* training with a different configuration.")
        network.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        for _ in range(start_epoch - 1):
            rng.permutation(len(train_samples))
        best_path = output_root / "best_checkpoint.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location=config.device, weights_only=True)
            best_validation = float(best_payload["validation_loss"])
        history_path = output_root / "losses.csv"
        if history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8") as stream:
                history = [
                    {key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)
                ]

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = perf_counter()
        network.train()
        order = rng.permutation(len(train_samples)).tolist()
        batches = [
            order[index : index + config.batch_size]
            for index in range(0, len(order), config.batch_size)
        ]
        training_losses: list[float] = []
        for batch_indices in tqdm(batches, desc=f"Epoch {epoch}/{config.epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample_index in batch_indices:
                sample = train_samples[sample_index]
                inputs, target = _tensor_inputs(sample, config.device)
                phi = make_guidance_tensor(network(inputs))[0]
                search = differentiable_neural_search(
                    sample.grid_map.grid,
                    sample.grid_map.start,
                    sample.grid_map.goal,
                    phi,
                    temperature=config.temperature,
                    tie_break_epsilon=config.tie_break_epsilon,
                    cost_config=sample.cost_config,
                )
                losses.append(mean_l1_search_loss(search.closed_mask, target))
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            training_losses.extend(float(loss.detach().item()) for loss in losses)

        validation_loss = _epoch_loss(network, validation_samples, config)
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
            "epoch_seconds": perf_counter() - epoch_started,
        }
        history.append(record)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "config": config_payload,
        }
        torch.save(checkpoint, output_root / "last_checkpoint.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, output_root / "best_checkpoint.pt")

        with (output_root / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("epoch", "train_loss", "validation_loss", "epoch_seconds"),
            )
            writer.writeheader()
            writer.writerows(history)

    if not (output_root / "best_checkpoint.pt").exists():
        raise RuntimeError("Neural A* training ended without a best validation checkpoint.")
    last_checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=True)
    best_checkpoint = torch.load(
        output_root / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    (output_root / "training_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "actual_completed_epochs": int(last_checkpoint["epoch"]),
                "best_epoch": int(best_checkpoint["epoch"]),
                "best_validation_loss": float(best_checkpoint["validation_loss"]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return network, history
