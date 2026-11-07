"""Optional SimCLR-style self-supervised training for independent map features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from data.dataset import DatasetSample
from ood.feature_extractor import OODFeatureConfig, OODFeatureExtractor, build_ood_input

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
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
class ContrastiveTrainingConfig:
    """Settings for training the independent OOD map encoder."""

    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    temperature: float = 0.2
    weight_decay: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    model: OODFeatureConfig = OODFeatureConfig()

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch_size at least two.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


def nt_xent_loss(first: Tensor, second: Tensor, temperature: float = 0.2) -> Tensor:
    """Compute the symmetric normalized-temperature cross-entropy objective."""

    if torch is None:
        raise ImportError("Contrastive feature training requires PyTorch.")
    if first.ndim != 2 or second.shape != first.shape or first.shape[0] < 2:
        raise ValueError("Two same-shaped (batch, features) views with batch >= 2 are required.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    batch_size = first.shape[0]
    joined = torch.nn.functional.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = joined @ joined.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        (
            torch.arange(batch_size, 2 * batch_size, device=first.device),
            torch.arange(0, batch_size, device=first.device),
        )
    )
    return torch.nn.functional.cross_entropy(logits, targets)


def _validate_training_samples(samples: Sequence[DatasetSample]) -> tuple[int, int]:
    if len(samples) < 2:
        raise ValueError("Contrastive training needs at least two training maps.")
    shapes = set()
    for index, sample in enumerate(samples):
        if sample.grid_map.distribution not in {"D1", "D2", "D3"}:
            raise ValueError(f"Training sample {index} is not from D1-D3.")
        if sample.ground_truth.optimal_path is None:
            raise ValueError(f"Training sample {index} is unreachable.")
        shapes.add(tuple(sample.grid_map.grid.shape))
    if len(shapes) != 1:
        raise ValueError("All feature training maps must have a common shape.")
    height, width = next(iter(shapes))
    return int(height), int(width)


def _views(batch: Tensor, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
    """Apply graph-preserving horizontal/vertical flips to channel-first maps."""

    first: list[Tensor] = []
    second: list[Tensor] = []
    for sample in batch:
        transformed = []
        for _ in range(2):
            view = sample
            if int(rng.integers(0, 2)):
                view = torch.flip(view, dims=(-1,))
            if int(rng.integers(0, 2)):
                view = torch.flip(view, dims=(-2,))
            transformed.append(view)
        first.append(transformed[0])
        second.append(transformed[1])
    return torch.stack(first), torch.stack(second)


def _batches(order: list[int], batch_size: int) -> list[list[int]]:
    batches = [order[index : index + batch_size] for index in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def train_contrastive_feature_extractor(
    training_samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    config: ContrastiveTrainingConfig = ContrastiveTrainingConfig(),
    validation_samples: Sequence[DatasetSample] = (),
    resume: bool = False,
) -> tuple[OODFeatureExtractor, list[dict[str, float]]]:
    """Train and save the self-supervised encoder using D1-D3 maps only.

    Optimizer batches use D1-D3 training maps only. Optional validation maps
    are used only for a no-gradient loss diagnostic; they never affect
    optimizer updates or best-checkpoint selection. Detector embeddings use
    the encoder output before its contrastive projector head.
    """

    if torch is None:
        raise ImportError("Contrastive feature training requires PyTorch.")
    _validate_training_samples(training_samples)
    if validation_samples:
        _validate_training_samples(validation_samples)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = OODFeatureExtractor(config.model).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    tensors = [
        torch.as_tensor(
            build_ood_input(sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal),
            dtype=torch.float32,
        )
        for sample in training_samples
    ]
    validation_tensors = [
        torch.as_tensor(
            build_ood_input(sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal),
            dtype=torch.float32,
        )
        for sample in validation_samples
    ]
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **asdict(config),
        "feature_input_channels": ["obstacle_occupancy", "start_one_hot", "goal_one_hot"],
        "embedding_used_for_detector": "pre-projector encoder output",
        "augmentations": ["horizontal flip", "vertical flip", "180-degree rotation"],
        "training_distributions": ["D1", "D2", "D3"],
        "validation_use": "no-gradient diagnostic loss only; no optimizer batches or checkpoint selection",
        "checkpoint_selection": (
            "best is the epoch with the lowest average contrastive training loss; "
            "no validation or OOD maps are used for selection"
        ),
    }
    (output_root / "feature_training_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    history: list[dict[str, float]] = []
    best_training_loss = float("inf")
    start_epoch = 1
    last_checkpoint_path = output_root / "last_checkpoint.pt"
    if resume and last_checkpoint_path.exists():
        saved = torch.load(last_checkpoint_path, map_location=config.device, weights_only=True)
        saved_config = dict(saved.get("training_config", {}))
        expected_config = dict(config_payload)
        saved_config.pop("device", None)
        expected_config.pop("device", None)
        if saved_config != expected_config:
            raise ValueError("Cannot resume OOD feature training with a different configuration.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        start_epoch = int(saved["epoch"]) + 1
        best_path = output_root / "best_checkpoint.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location=config.device, weights_only=True)
            best_training_loss = float(best_payload["contrastive_loss"])
        history_path = output_root / "feature_history.csv"
        if history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8") as stream:
                history = [
                    {key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)
                ]

    for epoch in range(start_epoch, config.epochs + 1):
        started = perf_counter()
        model.train()
        order = rng.permutation(len(tensors)).tolist()
        epoch_losses: list[float] = []
        for indices in tqdm(
            _batches(order, config.batch_size),
            desc=f"OOD features epoch {epoch}/{config.epochs}",
            leave=False,
        ):
            batch = torch.stack([tensors[index] for index in indices]).to(config.device)
            first_view, second_view = _views(batch, rng)
            optimizer.zero_grad(set_to_none=True)
            first = model.project(model(first_view))
            second = model.project(model(second_view))
            loss = nt_xent_loss(first, second, config.temperature)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        validation_loss: float | None = None
        if validation_tensors:
            model.eval()
            validation_rng = np.random.default_rng(config.seed + 10_000)
            validation_losses: list[float] = []
            with torch.no_grad():
                validation_order = list(range(len(validation_tensors)))
                for indices in _batches(validation_order, config.batch_size):
                    batch = torch.stack(
                        [validation_tensors[index] for index in indices]
                    ).to(config.device)
                    first_view, second_view = _views(batch, validation_rng)
                    first = model.project(model(first_view))
                    second = model.project(model(second_view))
                    validation_losses.append(
                        float(nt_xent_loss(first, second, config.temperature).cpu().item())
                    )
            validation_loss = float(np.mean(validation_losses))
        row = {
            "epoch": float(epoch),
            "contrastive_loss": float(np.mean(epoch_losses)),
            "validation_contrastive_loss": validation_loss,
            "epoch_seconds": float(perf_counter() - started),
        }
        history.append(row)
        with (output_root / "feature_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)
        improved = row["contrastive_loss"] < best_training_loss
        if improved:
            best_training_loss = row["contrastive_loss"]
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config.model),
            "training_config": config_payload,
            "epoch": epoch,
            "seed": config.seed,
            "contrastive_loss": row["contrastive_loss"],
            "best_training_loss": best_training_loss,
            "numpy_rng_state": rng.bit_generator.state,
        }
        torch.save(checkpoint, last_checkpoint_path)
        # Keep the established checkpoint filename as an alias for the final epoch.
        torch.save(checkpoint, output_root / "feature_extractor.pt")
        if improved:
            torch.save(checkpoint, output_root / "best_checkpoint.pt")

    best_path = output_root / "best_checkpoint.pt"
    if not best_path.exists():
        raise RuntimeError("OOD feature training ended without a best checkpoint.")
    best_payload = torch.load(best_path, map_location=config.device, weights_only=True)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()
    (output_root / "training_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "actual_completed_epochs": int(
                    torch.load(last_checkpoint_path, map_location="cpu", weights_only=True)["epoch"]
                ),
                "best_epoch": int(best_payload["epoch"]),
                "best_training_contrastive_loss": float(best_payload["contrastive_loss"]),
                "selection_uses_validation_data": False,
                "validation_loss_is_diagnostic_only": bool(validation_tensors),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model, history


def load_feature_extractor(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> tuple[OODFeatureExtractor, dict[str, Any]]:
    """Load a trained feature encoder checkpoint with its architecture config."""

    if torch is None:
        raise ImportError("Feature checkpoint loading requires PyTorch.")
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not contain a feature extractor state.")
    model = OODFeatureExtractor(OODFeatureConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
