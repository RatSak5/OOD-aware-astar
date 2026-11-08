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
    optimizer: str = "RMSprop"
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8
    rmsprop_momentum: float = 0.0
    rmsprop_weight_decay: float = 0.0
    rmsprop_centered: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    warmup_steps: int = 0
    normalization: str = "groupnorm8"
    initialization: str = "pytorch_default"
    temperature: float | None = None
    tie_break_epsilon: float = 0.001
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("epochs, batch_size, and gradient_accumulation_steps must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.optimizer != "RMSprop":
            raise ValueError("Neural A* training keeps RMSprop; other optimizers are unsupported.")
        if self.warmup_steps != 0:
            raise ValueError("The selected Neural A* configuration does not use learning-rate warmup.")
        if self.normalization != "groupnorm8":
            raise ValueError("The supported NeuralAStarUNet uses GroupNorm with eight groups.")
        if self.initialization != "pytorch_default":
            raise ValueError("The selected Neural A* configuration keeps PyTorch default initialization.")
        if self.temperature is not None and (
            not np.isfinite(self.temperature) or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive when provided.")
        if not np.isfinite(self.tie_break_epsilon) or self.tie_break_epsilon < 0:
            raise ValueError("tie_break_epsilon must be finite and non-negative.")
        if self.rmsprop_alpha <= 0 or self.rmsprop_alpha >= 1:
            raise ValueError("rmsprop_alpha must lie strictly between 0 and 1.")
        if not np.isfinite(self.rmsprop_eps) or self.rmsprop_eps <= 0:
            raise ValueError("rmsprop_eps must be finite and positive.")
        if self.rmsprop_momentum < 0 or self.rmsprop_weight_decay < 0:
            raise ValueError("RMSprop momentum and weight decay must be non-negative.")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive when provided.")
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


def _summarize_phi_health(phi_maps: Sequence[Any], logits_maps: Sequence[Any]) -> dict[str, float]:
    """Summarize unsupervised forward health probes without changing their outputs."""

    if not phi_maps or len(phi_maps) != len(logits_maps):
        raise ValueError("Phi health probes require matching non-empty Phi and logit collections.")
    spatial_stds = [float(phi.detach().float().std().item()) for phi in phi_maps]
    map_means = [float(phi.detach().float().mean().item()) for phi in phi_maps]
    all_phi = torch.cat([phi.detach().float().reshape(-1) for phi in phi_maps])
    all_logits = torch.cat([value.detach().float().reshape(-1) for value in logits_maps])
    return {
        "phi_spatial_std_mean": float(np.mean(spatial_stds)),
        "phi_spatial_std_min": float(np.min(spatial_stds)),
        "phi_spatial_std_max": float(np.max(spatial_stds)),
        "phi_across_map_mean_std": float(np.std(map_means)),
        "phi_fraction_lt_0_01": float((all_phi < 0.01).float().mean().item()),
        "phi_fraction_gt_0_99": float((all_phi > 0.99).float().mean().item()),
        "phi_saturated_fraction": float(((all_phi < 0.01) | (all_phi > 0.99)).float().mean().item()),
        "logit_mean_abs": float(all_logits.abs().mean().item()),
        "logit_max_abs": float(all_logits.abs().max().item()),
    }


class NeuralAStarCollapseError(RuntimeError):
    """Raised when Neural A* guidance loses useful numeric variation."""


def _assert_phi_healthy(summary: dict[str, float], *, context: str) -> None:
    """Fail fast on spatial collapse, sigmoid saturation, or runaway logits."""

    problems = []
    if summary["phi_spatial_std_max"] < 1e-3:
        problems.append(
            f"all probe spatial std values are below 1e-3 (maximum {summary['phi_spatial_std_max']:.3g})"
        )
    if summary["phi_saturated_fraction"] > 0.95:
        problems.append(
            f"{summary['phi_saturated_fraction']:.1%} of Phi cells are saturated outside [0.01, 0.99]"
        )
    if summary["logit_mean_abs"] > 8.0:
        problems.append(f"mean absolute pre-sigmoid logit {summary['logit_mean_abs']:.3g} exceeds 8")
    if problems:
        raise NeuralAStarCollapseError(
            f"Neural A* collapse guard failed at {context}: " + "; ".join(problems)
        )


def _probe_phi_health(
    model: Any,
    samples: Sequence[DatasetSample],
    device: str,
) -> dict[str, float]:
    """Run forward-only probes and include pre-sigmoid head values."""

    if not samples:
        raise ValueError("Neural A* collapse guard requires at least one probe sample.")
    captured_logits: list[Any] = []
    head = getattr(model, "guidance_head", None)
    if head is None:
        # Keep support for small test/integration models that expose their final
        # linear projection under the conventional name ``projection``.
        head = getattr(model, "projection", None)
    if head is None:
        raise TypeError("Neural A* collapse guard requires a model with guidance_head logits.")
    hook = head.register_forward_hook(
        lambda _module, _inputs, output: captured_logits.append(output.detach())
    )
    phi_maps: list[Any] = []
    try:
        was_training = bool(model.training)
        model.eval()
        with torch.no_grad():
            for sample in samples:
                inputs, _ = _tensor_inputs(sample, device)
                phi_maps.append(make_guidance_tensor(model(inputs))[0].detach())
        if was_training:
            model.train()
    finally:
        hook.remove()
    return _summarize_phi_health(phi_maps, captured_logits)


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
    optimizer = torch.optim.RMSprop(
        network.parameters(),
        lr=config.learning_rate,
        alpha=config.rmsprop_alpha,
        eps=config.rmsprop_eps,
        momentum=config.rmsprop_momentum,
        weight_decay=config.rmsprop_weight_decay,
        centered=config.rmsprop_centered,
    )
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **asdict(config),
        "method": "Yonetani et al. Neural A*",
        "optimizer": config.optimizer,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clip_norm": config.gradient_clip_norm,
        "warmup_steps": config.warmup_steps,
        "normalization": config.normalization,
        "normalization_details": "GroupNorm(8) after every convolution in encoder and decoder VGG blocks",
        "initialization": config.initialization,
        "initialization_details": "PyTorch default initialization; no output-head override",
        "rmsprop": {
            "alpha": config.rmsprop_alpha,
            "eps": config.rmsprop_eps,
            "momentum": config.rmsprop_momentum,
            "weight_decay": config.rmsprop_weight_decay,
            "centered": config.rmsprop_centered,
        },
        "objective": "mean_l1(closed_mask, shortest_path_map)",
        "encoder": "VGG16-block U-Net with GroupNorm(8) and sigmoid; PyTorch default initialization",
        "input_channels": ["passability=1-free", "start_plus_goal_one_hot"],
        "adaptations": [
            "shared eight-neighbor movement with no diagonal corner-cutting",
            "per-instance serial search inside each mini-batch",
        ],
    }
    config_path = output_root / "training_config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config_payload:
            raise ValueError(
                "Existing Neural A* training_config.json does not match the requested configuration; "
                "use a fresh run directory or restore the matching checkpoint/configuration."
            )
    else:
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    history: list[dict[str, float]] = []
    best_validation = float("inf")
    rng = np.random.default_rng(config.seed)
    start_epoch = 1
    last_checkpoint_path = output_root / "last_checkpoint.pt"
    if not resume and last_checkpoint_path.exists():
        raise FileExistsError(
            "Neural A* checkpoints already exist; resume them explicitly or select a fresh output directory."
        )
    if resume and last_checkpoint_path.exists():
        saved = torch.load(last_checkpoint_path, map_location=config.device, weights_only=True)
        saved_config = dict(saved.get("config", {}))
        expected_config = dict(config_payload)
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

    health_path = output_root / "step_health.csv"
    health_fields = (
        "epoch", "optimizer_step", "loss", "phi_spatial_std_mean", "phi_spatial_std_min", "phi_spatial_std_max",
        "phi_across_map_mean_std", "phi_fraction_lt_0_01", "phi_fraction_gt_0_99",
        "phi_saturated_fraction", "logit_mean_abs", "logit_max_abs", "gradient_norm",
        "health_probe", "epoch_end",
    )
    health_rows: list[dict[str, str]] = []
    if resume and health_path.exists():
        with health_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != health_fields:
                raise ValueError(
                    "Existing step_health.csv has an incompatible schema; use a fresh run directory."
                )
            health_rows = list(reader)
    else:
        with health_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=health_fields).writeheader()

    train_probes = list(train_samples[:4])
    validation_probes = list(validation_samples[:2])
    probe_samples = train_probes + validation_probes
    optimizer_steps_per_epoch = int(
        np.ceil(len(train_samples) / (config.batch_size * config.gradient_accumulation_steps))
    )
    completed_global_steps = (start_epoch - 1) * optimizer_steps_per_epoch
    previous_epoch_losses: tuple[float, float] | None = None
    if history:
        previous_epoch_losses = (
            float(history[-1]["train_loss"]),
            float(history[-1]["validation_loss"]),
        )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = perf_counter()
        network.train()
        order = rng.permutation(len(train_samples)).tolist()
        batches = [
            order[index : index + config.batch_size * config.gradient_accumulation_steps]
            for index in range(
                0, len(order), config.batch_size * config.gradient_accumulation_steps
            )
        ]
        training_losses: list[float] = []
        final_step_health_probe = False
        for batch_number, batch_indices in enumerate(
            tqdm(batches, desc=f"Epoch {epoch}/{config.epochs}", leave=False), start=1
        ):
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
            if config.gradient_clip_norm is None:
                gradient_norm = float(torch.sqrt(sum(
                    parameter.grad.detach().float().pow(2).sum()
                    for parameter in network.parameters()
                    if parameter.grad is not None
                )).item())
            else:
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    network.parameters(), config.gradient_clip_norm
                ).item())
            optimizer.step()
            completed_global_steps = (epoch - 1) * optimizer_steps_per_epoch + batch_number
            training_losses.extend(float(loss.detach().item()) for loss in losses)

            health_row: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_step": completed_global_steps,
                "loss": float(batch_loss.detach().item()),
                **{name: float("nan") for name in (
                    "phi_spatial_std_mean", "phi_spatial_std_min", "phi_spatial_std_max",
                    "phi_across_map_mean_std", "phi_fraction_lt_0_01", "phi_fraction_gt_0_99",
                    "phi_saturated_fraction", "logit_mean_abs", "logit_max_abs",
                )},
                "gradient_norm": gradient_norm,
                "health_probe": 0,
                "epoch_end": int(batch_number == len(batches)),
            }
            is_health_probe = (
                completed_global_steps in {10, 20, 30, 50}
                or completed_global_steps % 100 == 0
            )
            if is_health_probe:
                health = _probe_phi_health(network, probe_samples, config.device)
                health_row.update(health)
                health_row["health_probe"] = 1
                final_step_health_probe = batch_number == len(batches)
                print(
                    "Neural A* health "
                    f"step={completed_global_steps} loss={health_row['loss']:.6f} "
                    f"phi_std={health['phi_spatial_std_mean']:.4g} "
                    f"across_map_std={health['phi_across_map_mean_std']:.4g} "
                    f"sat={health['phi_saturated_fraction']:.1%} "
                    f"mean_abs_logit={health['logit_mean_abs']:.4g} grad_norm={gradient_norm:.4g}",
                    flush=True,
                )
            health_rows.append({key: str(value) for key, value in health_row.items()})
            with health_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=health_fields).writerow(health_row)
            if is_health_probe:
                _assert_phi_healthy(health, context=f"optimizer step {completed_global_steps}")

        validation_loss = _epoch_loss(network, validation_samples, config)
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
            "epoch_seconds": perf_counter() - epoch_started,
        }
        history.append(record)
        epoch_health = _probe_phi_health(network, probe_samples, config.device)
        if not (batches and final_step_health_probe):
            epoch_health_row = {
                "epoch": epoch,
                "optimizer_step": completed_global_steps,
                "loss": record["train_loss"],
                **epoch_health,
                "gradient_norm": gradient_norm if batches else float("nan"),
                "health_probe": 1,
                "epoch_end": 1,
            }
            health_rows.append({key: str(value) for key, value in epoch_health_row.items()})
            with health_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=health_fields).writerow(epoch_health_row)
        _assert_phi_healthy(epoch_health, context=f"end of epoch {epoch}")
        if previous_epoch_losses is not None and (
            round(record["train_loss"], 6) == round(previous_epoch_losses[0], 6)
            and round(validation_loss, 6) == round(previous_epoch_losses[1], 6)
        ):
            raise NeuralAStarCollapseError(
                "Neural A* collapse guard stopped training: train and validation losses are "
                f"unchanged to 6 decimals at epoch {epoch}."
            )
        previous_epoch_losses = (record["train_loss"], validation_loss)
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
                "configuration": config_payload,
                "optimization_deviations_from_original": [
                    "GroupNorm(8) added after every encoder/decoder convolution to control measured activation growth."
                ],
                "step_health_csv": str(health_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return network, history
