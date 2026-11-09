"""Training utilities for the differentiable Neural A* objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import logging
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from data.dataset import DatasetSample, assert_no_map_leakage
from neural_astar.differentiable import differentiable_neural_search
from neural_astar.guidance import build_encoder_input, path_to_mask
from neural_astar.model import NeuralAStarUNet, make_guidance_tensor

_LOGGER = logging.getLogger(__name__)
_LOGIT_WARNING_THRESHOLD = 8.0
_PHI_COLLAPSE_STD_THRESHOLD = 1e-3
_HEALTH_GROUP_FIELDS = (
    "map_count", "logit_signed_mean", "logit_mean_abs", "logit_std_mean", "logit_min",
    "logit_max", "logit_median_mean", "logit_p05_mean", "logit_p95_mean",
    "logit_fraction_gt_4", "logit_fraction_lt_minus_4", "logit_fraction_abs_gt_4",
    "logit_fraction_abs_gt_8", "phi_mean", "phi_std_mean", "phi_min", "phi_max",
    "phi_fraction_lt_0_01", "phi_fraction_gt_0_99",
)
_LEGACY_HEALTH_FIELDS = (
    "epoch", "optimizer_step", "loss", "phi_spatial_std_mean", "phi_spatial_std_min",
    "phi_spatial_std_max", "phi_across_map_mean_std", "phi_fraction_lt_0_01",
    "phi_fraction_gt_0_99", "phi_saturated_fraction", "logit_mean_abs", "logit_max_abs",
    "gradient_norm", "health_probe", "epoch_end",
)
_HEALTH_GROUP_CSV_FIELDS = tuple(
    f"{group}_{field}"
    for group in ("all_cell", "traversable", "obstacle")
    for field in _HEALTH_GROUP_FIELDS
)
_TRAVERSABLE_SPATIAL_CSV_FIELDS = (
    "traversable_phi_spatial_std_mean", "traversable_phi_spatial_std_min",
    "traversable_phi_spatial_std_max", "traversable_phi_across_map_mean_std",
)
_PRE_DIAGNOSTIC_HEALTH_FIELDS = (
    *_LEGACY_HEALTH_FIELDS, *_HEALTH_GROUP_CSV_FIELDS, *_TRAVERSABLE_SPATIAL_CSV_FIELDS,
)

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


def _guidance_head(model: Any) -> Any:
    head = getattr(model, "guidance_head", None)
    if head is None:
        # Keep support for small test/integration models that expose their final
        # linear projection under the conventional name ``projection``.
        head = getattr(model, "projection", None)
    if head is None:
        raise TypeError("Neural A* health checks require a model with guidance_head logits.")
    return head


def _forward_guidance_with_logits(model: Any, inputs: Any) -> tuple[Any, list[Any]]:
    """Run a model and fail immediately if its pre-sigmoid head is non-finite."""

    captured_logits: list[Any] = []
    hook = _guidance_head(model).register_forward_hook(
        lambda _module, _inputs, output: captured_logits.append(output.detach())
    )
    try:
        output = model(inputs)
    finally:
        hook.remove()
    if not captured_logits:
        raise RuntimeError("Neural A* guidance head did not produce logits during forward pass.")
    for logits in captured_logits:
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("Neural A* guidance logits contain NaN or infinite values.")
    return make_guidance_tensor(output), captured_logits


def _require_finite_tensor(value: Any, label: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"Neural A* {label} contains NaN or infinite values.")


def _require_finite_gradients(parameters: Sequence[Any]) -> None:
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient.detach()).all().item()):
            raise FloatingPointError("Neural A* gradients contain NaN or infinite values.")


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
            phi, _ = _forward_guidance_with_logits(model, inputs)
            phi = phi[0]
            search = differentiable_neural_search(
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                phi,
                temperature=config.temperature,
                tie_break_epsilon=config.tie_break_epsilon,
                cost_config=sample.cost_config,
            )
            loss = mean_l1_search_loss(search.closed_mask, target)
            _require_finite_tensor(loss, "validation loss")
            total += float(loss.item())
    mean_loss = total / len(samples)
    if not np.isfinite(mean_loss):
        raise FloatingPointError("Neural A* validation loss contains NaN or infinite values.")
    return mean_loss


def _summarize_phi_health(
    phi_maps: Sequence[Any],
    logits_maps: Sequence[Any],
    traversable_masks: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Summarize per-map Phi/logit health, separated by traversability when known.

    The legacy unprefixed metrics retain their previous all-cell definitions. New
    ``all_cell_*``, ``traversable_*``, and ``obstacle_*`` fields aggregate
    per-map statistics, so a large map cannot silently dominate another map.
    Callers without masks receive explicitly all-cell-labelled summaries only.
    """

    if not phi_maps or len(phi_maps) != len(logits_maps):
        raise ValueError("Phi health probes require matching non-empty Phi and logit collections.")
    if traversable_masks is not None and len(traversable_masks) != len(phi_maps):
        raise ValueError("Phi health probes require one traversability mask per map.")

    groups: dict[str, list[dict[str, float]]] = {"all_cell": [], "traversable": [], "obstacle": []}
    legacy_spatial_stds: list[float] = []
    legacy_map_means: list[float] = []
    legacy_phi: list[Any] = []
    legacy_logits: list[Any] = []

    def map_statistics(phi_values: Any, logit_values: Any) -> dict[str, float]:
        return {
            "logit_signed_mean": float(logit_values.mean().item()),
            "logit_mean_abs": float(logit_values.abs().mean().item()),
            "logit_std": float(logit_values.std(unbiased=logit_values.numel() > 1).item()),
            "logit_min": float(logit_values.min().item()),
            "logit_max": float(logit_values.max().item()),
            "logit_median": float(torch.quantile(logit_values, 0.5).item()),
            "logit_p05": float(torch.quantile(logit_values, 0.05).item()),
            "logit_p95": float(torch.quantile(logit_values, 0.95).item()),
            "logit_fraction_gt_4": float((logit_values > 4).float().mean().item()),
            "logit_fraction_lt_minus_4": float((logit_values < -4).float().mean().item()),
            "logit_fraction_abs_gt_4": float((logit_values.abs() > 4).float().mean().item()),
            "logit_fraction_abs_gt_8": float((logit_values.abs() > 8).float().mean().item()),
            "phi_mean": float(phi_values.mean().item()),
            # Match the pre-existing torch.std sample convention. For a
            # one-cell obstacle subset, use its defined population value 0.
            "phi_std": float(phi_values.std(unbiased=phi_values.numel() > 1).item()),
            "phi_min": float(phi_values.min().item()),
            "phi_max": float(phi_values.max().item()),
            "phi_fraction_lt_0_01": float((phi_values < 0.01).float().mean().item()),
            "phi_fraction_gt_0_99": float((phi_values > 0.99).float().mean().item()),
        }

    def aggregate_group(name: str, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        mean_fields = (
            "logit_signed_mean", "logit_mean_abs", "logit_std", "logit_median", "logit_p05",
            "logit_p95", "logit_fraction_gt_4", "logit_fraction_lt_minus_4",
            "logit_fraction_abs_gt_4", "logit_fraction_abs_gt_8", "phi_mean", "phi_std",
            "phi_fraction_lt_0_01", "phi_fraction_gt_0_99",
        )
        for field in mean_fields:
            if field in {"logit_std", "phi_std"}:
                output_field = f"{name}_{field}_mean"
            elif field in {"logit_median", "logit_p05", "logit_p95"}:
                output_field = f"{name}_{field}_mean"
            else:
                output_field = f"{name}_{field}"
            result[output_field] = float(np.mean([row[field] for row in rows]))
        result[f"{name}_logit_min"] = float(min(row["logit_min"] for row in rows))
        result[f"{name}_logit_max"] = float(max(row["logit_max"] for row in rows))
        result[f"{name}_phi_min"] = float(min(row["phi_min"] for row in rows))
        result[f"{name}_phi_max"] = float(max(row["phi_max"] for row in rows))
        result[f"{name}_map_count"] = float(len(rows))

    for index, (phi, logits) in enumerate(zip(phi_maps, logits_maps)):
        phi_values = phi.detach().float().reshape(-1)
        logit_values = logits.detach().float().reshape(-1)
        if phi_values.numel() != logit_values.numel():
            raise ValueError(f"Phi health probe {index} has mismatched logit and guidance sizes.")
        _require_finite_tensor(phi_values, "guidance values")
        _require_finite_tensor(logit_values, "guidance logits")
        if bool(((phi_values < 0.0) | (phi_values > 1.0)).any().item()):
            raise ValueError("Neural A* guidance values must lie in [0, 1].")
        legacy_phi.append(phi_values)
        legacy_logits.append(logit_values)
        legacy_spatial_stds.append(float(phi_values.std().item()))
        legacy_map_means.append(float(phi_values.mean().item()))

        if traversable_masks is None:
            masks = {"all_cell": torch.ones_like(phi_values, dtype=torch.bool)}
        else:
            mask = torch.as_tensor(traversable_masks[index], device=phi_values.device).bool().reshape(-1)
            if mask.numel() != phi_values.numel():
                raise ValueError(f"Phi health probe {index} has a traversability mask with the wrong size.")
            masks = {
                "all_cell": torch.ones_like(mask),
                "traversable": mask,
                "obstacle": ~mask,
            }
        for group_name, mask in masks.items():
            if bool(mask.any().item()):
                groups[group_name].append(map_statistics(phi_values[mask], logit_values[mask]))

    result: dict[str, float] = {}
    for group_name, rows in groups.items():
        aggregate_group(group_name, rows)

    all_phi = torch.cat(legacy_phi)
    all_logits = torch.cat(legacy_logits)
    result.update({
        # Backward-compatible fields. Their values keep the pre-change
        # all-cell/flattened meaning expected by existing health consumers.
        "phi_spatial_std_mean": float(np.mean(legacy_spatial_stds)),
        "phi_spatial_std_min": float(np.min(legacy_spatial_stds)),
        "phi_spatial_std_max": float(np.max(legacy_spatial_stds)),
        "phi_across_map_mean_std": float(np.std(legacy_map_means)),
        "phi_fraction_lt_0_01": float((all_phi < 0.01).float().mean().item()),
        "phi_fraction_gt_0_99": float((all_phi > 0.99).float().mean().item()),
        "phi_saturated_fraction": float(((all_phi < 0.01) | (all_phi > 0.99)).float().mean().item()),
        "logit_mean_abs": float(all_logits.abs().mean().item()),
        "logit_max_abs": float(all_logits.abs().max().item()),
    })
    if traversable_masks is not None:
        traversable_stds = [row["phi_std"] for row in groups["traversable"]]
        below_threshold_count = sum(
            value < _PHI_COLLAPSE_STD_THRESHOLD for value in traversable_stds
        )
        result.update({
            "traversable_phi_spatial_std_mean": float(np.mean(traversable_stds)) if traversable_stds else float("nan"),
            "traversable_phi_spatial_std_min": float(np.min(traversable_stds)) if traversable_stds else float("nan"),
            "traversable_phi_spatial_std_max": float(np.max(traversable_stds)) if traversable_stds else float("nan"),
            "traversable_phi_across_map_mean_std": float(np.std([row["phi_mean"] for row in groups["traversable"]])) if groups["traversable"] else float("nan"),
            "traversable_phi_maps_below_1e_3_count": float(below_threshold_count),
            "traversable_phi_maps_below_1e_3_proportion": (
                float(below_threshold_count / len(traversable_stds)) if traversable_stds else float("nan")
            ),
        })
    return result


class NeuralAStarCollapseError(RuntimeError):
    """Legacy exception type retained for callers; low variation is now diagnostic."""


def _assert_phi_healthy(summary: dict[str, float], *, context: str) -> None:
    """Report high logits and low spatial variation without treating either as failure."""

    if summary["logit_mean_abs"] > _LOGIT_WARNING_THRESHOLD:
        _LOGGER.warning(
            "Neural A* health warning at %s: mean absolute pre-sigmoid logit %.6g "
            "exceeds threshold %.6g; continuing training.",
            context,
            summary["logit_mean_abs"],
            _LOGIT_WARNING_THRESHOLD,
        )
    below_count = summary.get("traversable_phi_maps_below_1e_3_count")
    below_proportion = summary.get("traversable_phi_maps_below_1e_3_proportion")
    map_count = summary.get("traversable_map_count")
    if (
        below_count is not None
        and below_proportion is not None
        and map_count is not None
        and map_count > 0
        and below_count == map_count
    ):
        _LOGGER.warning(
            "Neural A* low-spatial-variation diagnostic at %s: %d/%d probe maps "
            "(%.1f%%) have traversable-cell Phi spatial std below reference threshold "
            "%.4g; this is non-fatal and training continues.",
            context,
            int(below_count),
            int(map_count),
            100.0 * below_proportion,
            _PHI_COLLAPSE_STD_THRESHOLD,
        )


def _probe_phi_health(
    model: Any,
    samples: Sequence[DatasetSample],
    device: str,
) -> dict[str, float]:
    """Run forward-only probes and include pre-sigmoid head values."""

    if not samples:
        raise ValueError("Neural A* collapse guard requires at least one probe sample.")
    phi_maps: list[Any] = []
    logits_maps: list[Any] = []
    traversable_masks: list[Any] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for sample in samples:
                inputs, _ = _tensor_inputs(sample, device)
                phi, captured_logits = _forward_guidance_with_logits(model, inputs)
                phi_maps.append(phi[0].detach())
                logits_maps.append(captured_logits[-1])
                # GridMap stores 0=traversable and 1=obstacle. This is the
                # canonical map encoding, distinct from encoder channel 0's
                # inverted passability representation (1=traversable).
                traversable_masks.append(sample.grid_map.grid == 0)
    finally:
        if was_training:
            model.train()
    return _summarize_phi_health(phi_maps, logits_maps, traversable_masks)


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
        *_PRE_DIAGNOSTIC_HEALTH_FIELDS,
        "traversable_phi_maps_below_1e_3_count",
        "traversable_phi_maps_below_1e_3_proportion",
    )
    health_rows: list[dict[str, str]] = []
    if resume and health_path.exists():
        with health_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_fields = tuple(reader.fieldnames or ())
            if existing_fields not in {
                health_fields, _PRE_DIAGNOSTIC_HEALTH_FIELDS, _LEGACY_HEALTH_FIELDS
            }:
                raise ValueError(
                    "Existing step_health.csv has an incompatible schema; use a fresh run directory."
                )
            health_rows = list(reader)
        if existing_fields != health_fields:
            # Extend either recognized older CSV schema in place. Old rows
            # retain their original values; new fields stay blank historically.
            with health_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=health_fields)
                writer.writeheader()
                writer.writerows(health_rows)
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
                phi, _ = _forward_guidance_with_logits(network, inputs)
                phi = phi[0]
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
            _require_finite_tensor(batch_loss, "training loss")
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
            parameters = list(network.parameters())
            _require_finite_gradients(parameters)
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Neural A* gradient norm is NaN or infinite.")
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
                    *(f"{group}_{field}" for group in ("all_cell", "traversable", "obstacle")
                      for field in _HEALTH_GROUP_FIELDS),
                    "traversable_phi_spatial_std_mean", "traversable_phi_spatial_std_min",
                    "traversable_phi_spatial_std_max", "traversable_phi_across_map_mean_std",
                    "traversable_phi_maps_below_1e_3_count",
                    "traversable_phi_maps_below_1e_3_proportion",
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
                    f"traversable_phi_std={health['traversable_phi_spatial_std_mean']:.4g} "
                    f"low_std_maps={int(health['traversable_phi_maps_below_1e_3_count'])}/"
                    f"{int(health['traversable_map_count'])} "
                    f"({health['traversable_phi_maps_below_1e_3_proportion']:.1%}) "
                    f"across_map_std={health['phi_across_map_mean_std']:.4g} "
                    f"sat={health['phi_saturated_fraction']:.1%} "
                    f"mean_abs_logit={health['logit_mean_abs']:.4g} "
                    f"traversable_mean_abs_logit={health['traversable_logit_mean_abs']:.4g} "
                    f"traversable_logit_std={health['traversable_logit_std_mean']:.4g} "
                    f"grad_norm={gradient_norm:.4g}",
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
            _LOGGER.warning(
                "Neural A* loss-progress diagnostic at epoch %d: train and validation "
                "losses match the prior epoch to six decimals; continuing training.",
                epoch,
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
