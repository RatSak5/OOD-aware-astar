"""Map-level training, validation, scheduling, and checkpointing for H1-H3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from data.dataset import DatasetSample, assert_no_map_leakage
from heuristics.cea import CEALossConfig
from heuristics.data import collate_heuristic_samples
from heuristics.evaluation import evaluate_heuristic_model
from heuristics.losses import ConstraintAwareLossConfig, compute_heuristic_losses
from models.heuristic_cnn import HeuristicCNN, HeuristicCNNConfig

try:
    import torch
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        """Console progress fallback for environments without tqdm."""

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
class HeuristicTrainingConfig:
    """Run settings for one independent H1, H2, or H3 model."""

    method: str = "H1"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-6
    seed: int = 0
    device: str = "cpu"
    model: HeuristicCNNConfig = HeuristicCNNConfig()
    composite_loss: ConstraintAwareLossConfig = ConstraintAwareLossConfig()
    cea_loss: CEALossConfig = CEALossConfig()

    def __post_init__(self) -> None:
        if self.method not in {"H1", "H2", "H3"}:
            raise ValueError("method must be H1, H2, or H3.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.optimizer.lower() not in {"adam", "adamw", "rmsprop"}:
            raise ValueError("optimizer must be Adam, AdamW, or RMSprop.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 0:
            raise ValueError("early stopping patience must be positive; scheduler patience non-negative.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between zero and one.")
        if not np.isfinite(self.minimum_learning_rate) or self.minimum_learning_rate <= 0:
            raise ValueError("minimum_learning_rate must be finite and positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("CNN heuristic training requires PyTorch.")


def _validate_splits(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
) -> tuple[int, int]:
    if not train_samples or not validation_samples:
        raise ValueError("Training and validation splits must both be non-empty.")
    shapes = set()
    for split_name, samples in (("train", train_samples), ("validation", validation_samples)):
        for index, sample in enumerate(samples):
            if sample.grid_map.distribution not in {"D1", "D2", "D3"}:
                raise ValueError(
                    f"{split_name}[{index}] uses {sample.grid_map.distribution}; "
                    "only D1-D3 are allowed before final OOD evaluation."
                )
            if sample.ground_truth.optimal_path is None:
                raise ValueError(f"{split_name}[{index}] has an unreachable start-goal pair.")
            shapes.add(tuple(sample.grid_map.grid.shape))
    if len(shapes) != 1:
        raise ValueError("All train and validation maps must share a fixed CNN input shape.")
    assert_no_map_leakage({"train": train_samples, "validation": validation_samples})
    height, width = next(iter(shapes))
    return int(height), int(width)


def _build_optimizer(config: HeuristicTrainingConfig, model: Any) -> Any:
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.RMSprop(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _evaluate(
    model: Any,
    samples: Sequence[DatasetSample],
    config: HeuristicTrainingConfig,
    progress_description: str,
) -> dict[str, dict[str, float]]:
    return evaluate_heuristic_model(
        model,
        samples,
        config.method,
        device=config.device,
        composite=config.composite_loss,
        cea=config.cea_loss,
        progress_description=progress_description,
    )


def _history_rows(
    epoch: int,
    split_name: str,
    metrics: dict[str, dict[str, float]],
    learning_rate: float,
    epoch_seconds: float,
) -> list[dict[str, float | str]]:
    rows = []
    for distribution, values in sorted(metrics.items()):
        rows.append(
            {
                "epoch": epoch,
                "split": split_name,
                **values,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
            }
        )
    return rows


def train_heuristic_model(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    config: HeuristicTrainingConfig = HeuristicTrainingConfig(),
    model: Any | None = None,
    resume: bool = False,
) -> tuple[Any, list[dict[str, float | str]]]:
    """Train one method on training maps and choose checkpoints on D1-D3 validation.

    No OOD/test data is accepted by this function. Best checkpoint selection
    uses mean per-map validation objective; the D4-D6 split belongs to the
    separate post-training evaluator.
    """

    _require_torch()
    height, width = _validate_splits(train_samples, validation_samples)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_mode = "ordinal" if config.method == "H3" else "regression"
    model_config = replace(config.model, output_mode=output_mode)
    class_count = height * width if config.method == "H3" else None
    network = model or HeuristicCNN(model_config, num_classes=class_count)
    network = network.to(config.device)
    optimizer = _build_optimizer(config, network)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **asdict(config),
        "resolved_model": asdict(model_config),
        "class_count": class_count,
        "method_description": {
            "H1": "standard masked MSE regression",
            "H2": "proposed asymmetric composite constraint-aware loss",
            "H3": "published Cross-Entropy Admissibility ordinal classification",
        }[config.method],
        "cost_model": "unit_move_count; c(u,v)=1 on every legal edge",
        "input_channels": ["obstacle_occupancy", "start_one_hot", "goal_one_hot"],
    }
    (output_root / "training_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    history: list[dict[str, float | str]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    rng = np.random.default_rng(config.seed)
    start_epoch = 1
    last_checkpoint_path = output_root / "last_checkpoint.pt"
    if resume and last_checkpoint_path.exists():
        saved = torch.load(last_checkpoint_path, map_location=config.device, weights_only=True)
        saved_config = dict(saved.get("training_config", {}))
        expected_config = dict(config_payload)
        saved_config.pop("device", None)
        expected_config.pop("device", None)
        if saved_config != expected_config or saved.get("method") != config.method:
            raise ValueError("Cannot resume heuristic training with a different configuration.")
        network.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler_state = saved.get("scheduler_state_dict")
        if scheduler_state is None:
            raise ValueError("Resume checkpoint is missing the ReduceLROnPlateau state.")
        scheduler.load_state_dict(scheduler_state)
        start_epoch = int(saved["epoch"]) + 1
        epochs_without_improvement = int(saved.get("epochs_without_improvement", 0))
        for _ in range(start_epoch - 1):
            rng.permutation(len(train_samples))
        best_path = output_root / "best_checkpoint.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location=config.device, weights_only=True)
            best_validation_loss = float(best_payload["validation_loss"])
        history_path = output_root / "history.csv"
        if history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = perf_counter()
        network.train()
        order = rng.permutation(len(train_samples)).tolist()
        batches = [
            order[index : index + config.batch_size]
            for index in range(0, len(order), config.batch_size)
        ]
        for batch_indices in tqdm(batches, desc=f"{config.method} epoch {epoch}/{config.epochs}", leave=False):
            batch_samples = [train_samples[index] for index in batch_indices]
            batch = collate_heuristic_samples(batch_samples, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            raw_output = network(batch.inputs)
            total_loss, _, _ = compute_heuristic_losses(
                config.method,
                raw_output,
                batch.targets,
                batch.valid_mask,
                batch.edge_indices,
                batch.edge_costs,
                batch.goal_coordinates,
                composite=config.composite_loss,
                cea=config.cea_loss,
            )
            total_loss.backward()
            optimizer.step()

        train_metrics = _evaluate(
            network, train_samples, config, f"{config.method} train evaluation"
        )
        validation_metrics = _evaluate(
            network, validation_samples, config, f"{config.method} validation evaluation"
        )
        monitored_validation = validation_metrics["all"]["loss_total"]
        scheduler.step(monitored_validation)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = perf_counter() - epoch_started
        history.extend(
            _history_rows(epoch, "train", train_metrics, learning_rate, epoch_seconds)
        )
        history.extend(
            _history_rows(epoch, "validation", validation_metrics, learning_rate, epoch_seconds)
        )

        improved = monitored_validation < best_validation_loss
        if improved:
            best_validation_loss = monitored_validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "method": config.method,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": monitored_validation,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "model_config": asdict(model_config),
            "num_classes": class_count,
            "training_config": config_payload,
        }
        torch.save(checkpoint, output_root / "last_checkpoint.pt")
        if improved:
            torch.save(checkpoint, output_root / "best_checkpoint.pt")

        fieldnames = list(history[0].keys())
        with (output_root / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    if not (output_root / "best_checkpoint.pt").exists():
        raise RuntimeError("Heuristic training ended without a best validation checkpoint.")
    best_checkpoint = torch.load(
        output_root / "best_checkpoint.pt", map_location=config.device, weights_only=True
    )
    network.load_state_dict(best_checkpoint["model_state_dict"])
    network.eval()
    final_checkpoint = torch.load(
        last_checkpoint_path, map_location="cpu", weights_only=True
    )
    completion = {
        "completed": True,
        "actual_completed_epochs": int(final_checkpoint["epoch"]),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": float(best_checkpoint["validation_loss"]),
        "stopped_by_early_stopping": (
            int(final_checkpoint["epoch"]) < config.epochs
            and int(final_checkpoint.get("epochs_without_improvement", 0))
            >= config.early_stopping_patience
        ),
    }
    (output_root / "training_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    return network, history
