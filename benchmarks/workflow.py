"""Dataset, model training, and calibration stages for Prompt 8."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.config import BenchmarkConfig
from benchmarks.data import load_training_splits, prepare_datasets
from ood.contrastive import ContrastiveTrainingConfig, train_contrastive_feature_extractor
from ood.mahalanobis import (
    MahalanobisConfig,
    calibrate_d95,
    fit_mahalanobis_detector,
    normalized_ood_score,
    save_detector,
)
from training.heuristic_training import HeuristicTrainingConfig, train_heuristic_model
from neural_astar.training import NeuralAStarTrainingConfig, train_neural_astar
from adaptive.ood_adaptive_seeastar import extract_map_embedding


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_metadata(*, cpu_threads: int) -> dict[str, Any]:
    """Capture the runtime environment used for reproducibility reports."""

    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "configured_cpu_threads": int(cpu_threads),
        }
    except ImportError:
        torch_info = {"version": None, "cuda_available": False}
    try:
        import matplotlib

        matplotlib_version = str(matplotlib.__version__)
    except ImportError:
        matplotlib_version = None
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": None,
        "ram_note": "Not available from the restricted Windows system-information API.",
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib_version,
        "torch": torch_info,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _training_curve_summary(
    history: list[dict[str, Any]],
    *,
    validation_key: str,
    training_key: str,
    label: str,
) -> dict[str, Any]:
    validation = [row for row in history if row.get("split") == "validation" and row.get("distribution") == "all"]
    training = [row for row in history if row.get("split") == "train" and row.get("distribution") == "all"]
    if not validation:
        validation = [row for row in history if validation_key in row]
    if not training:
        training = [row for row in history if training_key in row]
    if not validation or not training:
        return {
            "label": label,
            "status": "insufficient_curve_data",
            "epochs_observed": len(history),
        }
    best = min(validation, key=lambda row: float(row[validation_key]))
    first_validation = float(validation[0][validation_key])
    final_validation = float(validation[-1][validation_key])
    best_epoch = int(best.get("epoch", len(validation)))
    training_at_best = next(
        (float(row[training_key]) for row in training if int(row.get("epoch", -1)) == best_epoch),
        float(training[-1][training_key]),
    )
    best_validation = float(best[validation_key])
    improvement = (
        (first_validation - best_validation) / abs(first_validation)
        if first_validation != 0.0
        else 0.0
    )
    overfit_after_best = final_validation > best_validation * 1.05 + 1e-12
    return {
        "label": label,
        "epochs_observed": len(validation),
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "training_loss_at_best_epoch": training_at_best,
        "validation_loss_at_first_epoch": first_validation,
        "validation_loss_at_last_epoch": final_validation,
        "validation_relative_improvement": float(improvement),
        "validation_minus_training_loss_at_best": best_validation - training_at_best,
        "validation_rose_more_than_5_percent_after_best": bool(overfit_after_best),
        "possible_underfitting": bool(improvement < 0.05),
        "interpretation_note": (
            "Heuristic losses are compared only within a method; the loss scales "
            "differ across H1, H2, and H3. Flags are diagnostics, not automatic "
            "retraining or model-selection rules."
        ),
    }


def _neural_curve_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"status": "missing_history"}
    best = min(history, key=lambda row: float(row["validation_loss"]))
    first = float(history[0]["validation_loss"])
    best_validation = float(best["validation_loss"])
    final = float(history[-1]["validation_loss"])
    relative_improvement = (
        (first - best_validation) / abs(first) if first != 0.0 else 0.0
    )
    return {
        "epochs_observed": len(history),
        "best_validation_epoch": int(best["epoch"]),
        "best_validation_loss": best_validation,
        "best_training_loss": float(best["train_loss"]),
        "first_validation_loss": first,
        "last_validation_loss": final,
        "validation_relative_improvement": float(relative_improvement),
        "validation_rose_more_than_5_percent_after_best": bool(
            final > best_validation * 1.05 + 1e-12
        ),
        "possible_underfitting": bool(relative_improvement < 0.05),
        "checkpoint_selection": "best D1-D3 validation loss",
    }


def _plot_training_curves(
    histories: dict[str, list[dict[str, Any]]],
    output_directory: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    heuristic_figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, method in zip(axes, ("H1", "H2", "H3")):
        rows = histories[method]
        for split, style in (("train", "-"), ("validation", "--")):
            selected = [
                row
                for row in rows
                if row.get("split") == split and row.get("distribution") == "all"
            ]
            axis.plot(
                [row["epoch"] for row in selected],
                [row["loss_total"] for row in selected],
                style,
                label=split,
            )
        axis.set_title(method)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Method-specific objective")
        axis.grid(alpha=0.25)
        axis.legend()
    heuristic_path = output_directory / "heuristic_learning_curves.png"
    heuristic_figure.savefig(heuristic_path, dpi=180)
    plt.close(heuristic_figure)
    paths["heuristics"] = str(heuristic_path)

    neural_rows = histories["NeuralAStar"]
    neural_figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot([row["epoch"] for row in neural_rows], [row["train_loss"] for row in neural_rows], label="train")
    axis.plot([row["epoch"] for row in neural_rows], [row["validation_loss"] for row in neural_rows], label="validation")
    axis.set(title="Neural A* search loss", xlabel="Epoch", ylabel="Mean L1 search loss")
    axis.grid(alpha=0.25)
    axis.legend()
    neural_path = output_directory / "neural_astar_learning_curve.png"
    neural_figure.savefig(neural_path, dpi=180)
    plt.close(neural_figure)
    paths["neural_astar"] = str(neural_path)

    ood_rows = histories["OODFeature"]
    ood_figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot([row["epoch"] for row in ood_rows], [row["contrastive_loss"] for row in ood_rows])
    axis.set(title="OOD feature contrastive training", xlabel="Epoch", ylabel="NT-Xent training loss")
    axis.grid(alpha=0.25)
    ood_path = output_directory / "ood_feature_learning_curve.png"
    ood_figure.savefig(ood_path, dpi=180)
    plt.close(ood_figure)
    paths["ood_feature_training_only"] = str(ood_path)
    return paths


def prepare_benchmark_data(
    config: BenchmarkConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate dataset archives under the config's output directory."""

    destination = Path(config.output_directory) / "data"
    return prepare_datasets(config, destination, overwrite=overwrite)


def train_models_and_calibrate(
    config: BenchmarkConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train H1-H3, Neural A*, and the OOD encoder; fit/calibrate Mahalanobis.

    This stage loads only the training and validation/calibration archives. It
    does not open the ID holdout or OOD test file.
    """

    import torch

    root = Path(config.output_directory)
    data_root = root / "data"
    model_root = root / "models"
    summary_path = model_root / "model_training_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Training artifacts already exist under {model_root}; pass overwrite=True to retrain."
        )
    train_samples, validation_samples = load_training_splits(data_root)
    if len(train_samples) != config.train_count or len(validation_samples) != config.validation_count:
        raise ValueError("Saved train/validation counts do not match the approved config.")

    torch.set_num_threads(config.cpu_threads)
    config_path = root / "benchmark_config.json"
    config.save(config_path)
    _write_json(root / "environment.json", environment_metadata(cpu_threads=config.cpu_threads))
    model_root.mkdir(parents=True, exist_ok=True)
    histories: dict[str, list[dict[str, Any]]] = {}
    model_records: dict[str, Any] = {}
    checkpoint_paths: dict[str, Path] = {}

    for index, method in enumerate(("H1", "H2", "H3")):
        method_root = model_root / "heuristics" / method
        training_config = HeuristicTrainingConfig(
            method=method,
            epochs=config.heuristic_epochs,
            batch_size=config.heuristic_batch_size,
            seed=(config.base_seed + index) % (2**32),
            device=config.device,
        )
        started = perf_counter()
        _, history = train_heuristic_model(
            train_samples,
            validation_samples,
            method_root,
            config=training_config,
        )
        elapsed = perf_counter() - started
        histories[method] = [dict(row) for row in history]
        checkpoint_path = method_root / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Training did not write {checkpoint_path}.")
        checkpoint_paths[method] = checkpoint_path
        model_records[method] = {
            "training_seconds": float(elapsed),
            "training_config": asdict(training_config),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "curve_summary": _training_curve_summary(
                histories[method],
                validation_key="loss_total",
                training_key="loss_total",
                label=method,
            ),
        }

    neural_root = model_root / "neural_astar"
    neural_config = NeuralAStarTrainingConfig(
        epochs=config.neural_astar_epochs,
        batch_size=config.neural_astar_batch_size,
        seed=(config.base_seed + 10) % (2**32),
        device=config.device,
    )
    neural_started = perf_counter()
    _, neural_history = train_neural_astar(
        train_samples,
        validation_samples,
        neural_root,
        config=neural_config,
    )
    neural_seconds = perf_counter() - neural_started
    histories["NeuralAStar"] = [dict(row) for row in neural_history]
    neural_checkpoint = neural_root / "best_checkpoint.pt"
    if not neural_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not write {neural_checkpoint}.")
    checkpoint_paths["NeuralAStar"] = neural_checkpoint
    model_records["NeuralAStar"] = {
        "training_seconds": float(neural_seconds),
        "training_config": asdict(neural_config),
        "checkpoint": str(neural_checkpoint),
        "checkpoint_sha256": _sha256(neural_checkpoint),
        "curve_summary": _neural_curve_summary(histories["NeuralAStar"]),
    }

    ood_root = model_root / "ood_feature"
    ood_config = ContrastiveTrainingConfig(
        epochs=config.ood_epochs,
        batch_size=config.ood_batch_size,
        seed=(config.base_seed + 20) % (2**32),
        device=config.device,
    )
    ood_started = perf_counter()
    feature_extractor, ood_history = train_contrastive_feature_extractor(
        train_samples,
        ood_root,
        config=ood_config,
    )
    ood_training_seconds = perf_counter() - ood_started
    histories["OODFeature"] = [dict(row) for row in ood_history]
    feature_checkpoint = ood_root / "feature_extractor.pt"
    if not feature_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not write {feature_checkpoint}.")
    checkpoint_paths["OODFeature"] = feature_checkpoint
    model_records["OODFeature"] = {
        "training_seconds": float(ood_training_seconds),
        "training_config": asdict(ood_config),
        "checkpoint": str(feature_checkpoint),
        "checkpoint_sha256": _sha256(feature_checkpoint),
        "epochs_observed": len(ood_history),
        "first_training_loss": float(ood_history[0]["contrastive_loss"]),
        "last_training_loss": float(ood_history[-1]["contrastive_loss"]),
        "validation_note": (
            "The existing contrastive trainer has no validation objective or checkpoint selection. "
            "ID calibration and post-training OOD discrimination are reported separately."
        ),
    }

    # Detector fitting uses D1-D3 training embeddings only; D95 uses only the
    # independent D1-D3 validation/calibration split.
    detector = fit_mahalanobis_detector(
        train_samples,
        feature_extractor,
        config=MahalanobisConfig(),
        device=config.device,
    )
    calibrate_d95(
        detector,
        validation_samples,
        feature_extractor,
        device=config.device,
    )
    calibration_distances_started = perf_counter()
    calibration_embeddings = np.stack(
        [
            extract_map_embedding(
                feature_extractor,
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                device=config.device,
            )
            for sample in validation_samples
        ],
        axis=0,
    )
    calibration_distances = detector.distance(calibration_embeddings)
    calibration_inference_seconds = perf_counter() - calibration_distances_started
    calibration_score_rows = [
        {
            "map_id": f"{sample.grid_map.distribution}:{sample.grid_map.seed}",
            "distribution": sample.grid_map.distribution,
            "map_seed": int(sample.grid_map.seed),
            "mahalanobis_distance": float(distance),
            "smooth_score": float(normalized_ood_score(float(distance), float(detector.d95))),
        }
        for sample, distance in zip(validation_samples, calibration_distances)
    ]
    calibration_distances_path = ood_root / "validation_calibration_distances.csv"
    with calibration_distances_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_score_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_score_rows)
    detector_path = ood_root / "mahalanobis_detector.npz"
    save_detector(detector, detector_path)
    detector_metadata_path = detector_path.with_suffix(".json")
    detector_record = {
        "path": str(detector_path),
        "metadata_path": str(detector_metadata_path),
        "training_map_count": detector.num_training_maps,
        "calibration_map_count": detector.num_calibration_maps,
        "d95": float(detector.d95),
        "shrinkage": float(detector.shrinkage),
        "ridge_added": float(detector.ridge_added),
        "training_fingerprints": list(detector.training_fingerprints),
        "calibration_fingerprints": list(detector.calibration_fingerprints),
        "calibration_distance_summary": {
            "minimum": float(calibration_distances.min()),
            "mean": float(calibration_distances.mean()),
            "median": float(np.median(calibration_distances)),
            "maximum": float(calibration_distances.max()),
            "q95": float(np.quantile(calibration_distances, 0.95, method="linear")),
            "d95_inference_seconds": float(calibration_inference_seconds),
            "per_map_csv": str(calibration_distances_path),
        },
    }
    model_records["OODDetector"] = detector_record

    curves = _plot_training_curves(histories, root / "figures" / "training")
    summary = {
        "schema_version": 1,
        "training_complete": True,
        "ood_test_opened": False,
        "id_holdout_opened": False,
        "configuration": config.to_dict(),
        "split_sizes": {
            "train": len(train_samples),
            "validation_calibration": len(validation_samples),
        },
        "models": model_records,
        "training_curve_figures": curves,
        "limits": [
            "Training uses the approved default epoch counts and architecture settings.",
            "No training configuration was changed based on ID/OOD holdout performance.",
        ],
    }
    _write_json(summary_path, summary)
    model_ready = {
        "schema_version": 1,
        "training_complete": True,
        "d95_calibrated": True,
        "ood_test_opened": False,
        "id_holdout_opened": False,
        "model_training_summary": str(summary_path),
        "detector_path": str(detector_path),
        "d95": float(detector.d95),
        "calibration_fingerprints": list(detector.calibration_fingerprints),
        "model_checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in checkpoint_paths.items()
        },
    }
    model_ready_path = root / "model_ready_manifest.json"
    _write_json(model_ready_path, model_ready)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "model_ready_manifest": model_ready_path,
        "feature_extractor": feature_extractor,
        "detector": detector,
    }
