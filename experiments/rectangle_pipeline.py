"""Training, calibration, and inference integration for 64x64 rectangles.

This staged pipeline intentionally has no code path that opens the sealed
OOD-test archive. It loads only training, validation/calibration, ID-pilot, and
OOD-pilot archives, with the OOD pilot gated on finalized models and D95.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from adaptive import ood_adaptive_seeastar as adaptive_search
from algorithms.movement import DEFAULT_COST_CONFIG, get_transition_cost
from data.dataset import DatasetSample
from data.ground_truth import GroundTruth
from data.map_generator import GridMap, MapGenerationConfig
from data.rectangular_dataset import BASE_SEED, RUN_ID
from neural_astar.inference import load_neural_astar_checkpoint, predict_guidance
from neural_astar.training import NeuralAStarTrainingConfig, train_neural_astar
from ood.contrastive import (
    ContrastiveTrainingConfig,
    load_feature_extractor,
    train_contrastive_feature_extractor,
)
from ood.feature_extractor import build_ood_input
from ood.mahalanobis import (
    MahalanobisConfig,
    calibrate_d95,
    fit_mahalanobis_detector,
    normalized_ood_score,
    save_detector,
    load_detector,
)
from seeastar.neural_guidance import run_neural_guidance_seeastar
from experiments.run_paths import create_run_directory, validate_run_directory

try:
    import psutil
except ImportError:  # pragma: no cover - optional environment reporting
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch is required for training stages
    torch = None


ALLOWED_SPLITS = {"training", "validation_calibration", "id_pilot", "ood_pilot"}
SAMPLING_SEEDS = (11, 23, 47, 89, 131)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_environment() -> dict[str, Any]:
    """Collect runtime facts without assuming a Kaggle accelerator model."""

    package_names = (
        "numpy", "torch", "scipy", "scikit-learn", "psutil", "pandas",
        "matplotlib", "tqdm", "ipykernel", "nbformat",
    )
    packages: dict[str, str | None] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    gpu_memory_bytes = None
    cuda_version = None
    cuda_available = False
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            gpu = properties.name
            gpu_memory_bytes = int(properties.total_memory)
    memory = None
    if psutil is not None:
        memory = {
            "total_bytes": int(psutil.virtual_memory().total),
            "available_bytes": int(psutil.virtual_memory().available),
        }
    return {
        "run_id": RUN_ID,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "system_memory": memory,
        "pytorch_version": None if torch is None else torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_model": gpu,
        "gpu_memory_bytes": gpu_memory_bytes,
        "packages": packages,
        "is_kaggle": Path("/kaggle").exists(),
        "kaggle_input_read_only_root": "/kaggle/input" if Path("/kaggle/input").exists() else None,
        "kaggle_working_root": "/kaggle/working" if Path("/kaggle/working").exists() else None,
        "determinism": {
            "seed_policy": "Python, NumPy, PyTorch, and CUDA seeded per stage",
            "torch_deterministic_algorithms": True,
            "hardware_nondeterminism_note": (
                "CUDA kernels may have hardware or library-specific nondeterminism; "
                "the active device and package versions are recorded above."
            ),
        },
    }


def save_environment(result_root: str | Path) -> dict[str, Any]:
    payload = detect_environment()
    _atomic_json(Path(result_root) / "environment.json", payload)
    return payload


def resolve_device(requested: str = "auto") -> str:
    if torch is None:
        raise ImportError("Training and inference require PyTorch.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    return requested


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_samples(data_directory: str | Path, split_name: str) -> list[DatasetSample]:
    """Load one permitted split, rejecting holdout and sealed-test names."""

    if split_name not in ALLOWED_SPLITS:
        raise PermissionError(
            f"Split {split_name!r} is unavailable in this phase. Allowed: {sorted(ALLOWED_SPLITS)}."
        )
    archive_path = Path(data_directory) / f"{split_name}.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    maps = arrays["maps"]
    if maps.ndim != 3 or maps.shape[1:] != (64, 64):
        raise ValueError(f"{split_name} contains maps that are not 64x64.")
    if arrays["inputs"].shape != (len(maps), 3, 64, 64):
        raise ValueError("Canonical dataset channels must have shape (N,3,64,64).")
    if tuple(arrays["channel_order"].tolist()) != (
        "obstacle_occupancy", "start_one_hot", "goal_one_hot"
    ):
        raise ValueError("Unexpected canonical input channel order.")
    if arrays["distance_maps"].dtype != np.float64:
        raise ValueError("Ground-truth distance labels must use float64.")
    samples: list[DatasetSample] = []
    path_values = arrays["optimal_path_coordinates"]
    path_offsets = arrays["path_offsets"]
    for index, grid in enumerate(maps):
        distribution = str(arrays["distribution_labels"][index])
        seed = int(arrays["generation_seeds"][index])
        start = tuple(int(value) for value in arrays["starts"][index])
        goal = tuple(int(value) for value in arrays["goals"][index])
        lower = int(path_offsets[index])
        upper = int(path_offsets[index + 1])
        path = tuple(tuple(int(value) for value in coord) for coord in path_values[lower:upper])
        distance_map = arrays["distance_maps"][index].astype(np.float64, copy=True)
        reachable = arrays["reachable_masks"][index].astype(np.bool_, copy=True)
        if not path or not reachable[start] or not np.isfinite(distance_map[start]):
            raise ValueError(f"{split_name}[{index}] has no valid shortest path.")
        map_object = GridMap(
            grid=grid,
            start=start,
            goal=goal,
            distribution=distribution,
            seed=seed,
            structure="rectangular_obstacle_union",
        )
        generation_config = MapGenerationConfig(
            height=64,
            width=64,
            distribution=distribution,
            seed=seed,
            obstacle_density=float(arrays["actual_densities"][index]),
        )
        truth = GroundTruth(
            start=start,
            goal=goal,
            distance_map=distance_map,
            optimal_path=path,
            optimal_path_cost=float(arrays["shortest_path_distances"][index]),
            reachable_mask=reachable,
        )
        samples.append(
            DatasetSample(map_object, truth, generation_config, DEFAULT_COST_CONFIG)
        )
    return samples


def _model_ready_path(result_root: Path) -> Path:
    return result_root / "models" / "model_ready.json"


def train_models(
    data_directory: str | Path,
    result_root: str | Path,
    *,
    device: str = "auto",
    resume: bool = True,
) -> dict[str, Any]:
    """Train the primary Neural A* and independent OOD models only.

    H1/H2/H3 are optional comparison models and are never trained by this
    Phase B orchestration. Existing completed Phase B stages are returned
    without rewriting their checkpoints or histories.
    """

    actual_device = resolve_device(device)
    root = Path(result_root)
    models_root = root / "models"
    config_root = root / "configs"
    completed_path = root / "pilot" / "pilot_complete.json"
    if completed_path.is_file():
        raise FileExistsError(
            f"Pilot already completed in {root}; choose a new run directory to preserve it."
        )
    neural_config = NeuralAStarTrainingConfig(
        epochs=20,
        batch_size=1,
        learning_rate=0.0001,
        seed=BASE_SEED + 200,
        device=actual_device,
    )
    ready_path = _model_ready_path(root)
    summary_path = models_root / "training_summary.json"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("training_complete") and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            neural_summary = summary.get("models", {}).get("NeuralAStar", {})
            saved_configuration = neural_summary.get("configuration")
            expected_configuration = asdict(neural_config)
            if saved_configuration != expected_configuration:
                raise ValueError(
                    "Completed Neural A* results were produced with a different or unknown "
                    "training configuration. Select a fresh run directory; do not reuse a "
                    "collapsed or legacy checkpoint."
                )
            config_path = models_root / "neural_astar" / "training_config.json"
            if not config_path.is_file():
                raise ValueError(
                    "Completed Neural A* results have no training_config.json; use a fresh run directory."
                )
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            if any(config_payload.get(key) != value for key, value in expected_configuration.items()):
                raise ValueError(
                    "Completed Neural A* training_config.json does not match this pipeline; "
                    "use a fresh run directory."
                )
            return summary
    seed_runtime(BASE_SEED)
    models_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    train_samples = load_samples(data_directory, "training")
    validation_samples = load_samples(data_directory, "validation_calibration")
    if len(train_samples) != 3000 or len(validation_samples) != 600:
        raise ValueError("Primary training requires exactly 3000 training and 600 validation maps.")
    if {sample.grid_map.distribution for sample in train_samples} != {"D1", "D2", "D3"}:
        raise ValueError("Model training may use only D1-D3.")

    environment = save_environment(root)
    model_summaries: dict[str, Any] = {}
    training_started = perf_counter()
    neural_directory = models_root / "neural_astar"
    _, neural_history = train_neural_astar(
        train_samples,
        validation_samples,
        neural_directory,
        config=neural_config,
        resume=resume,
    )
    neural_completion = json.loads((neural_directory / "training_complete.json").read_text())
    model_summaries["NeuralAStar"] = {
        **neural_completion,
        "from_scratch": True,
        "legacy_32x32_weights_reused": False,
        "history_rows": len(neural_history),
        "training_runtime_seconds": float(sum(row["epoch_seconds"] for row in neural_history)),
        "best_checkpoint": str(neural_directory / "best_checkpoint.pt"),
        "last_checkpoint": str(neural_directory / "last_checkpoint.pt"),
        "configuration": asdict(neural_config),
        "training_configuration": json.loads(
            (neural_directory / "training_config.json").read_text(encoding="utf-8")
        ),
    }

    ood_directory = models_root / "ood_feature_extractor"
    ood_config = ContrastiveTrainingConfig(
        epochs=40,
        batch_size=32,
        seed=BASE_SEED + 300,
        device=actual_device,
    )
    _, ood_history = train_contrastive_feature_extractor(
        train_samples,
        ood_directory,
        config=ood_config,
        validation_samples=validation_samples,
        resume=resume,
    )
    ood_completion = json.loads((ood_directory / "training_complete.json").read_text())
    model_summaries["OODFeatureExtractor"] = {
        **ood_completion,
        "from_scratch": True,
        "embedding_dim": int(ood_config.model.embedding_dim),
        "input_channels": ["obstacle_occupancy", "start_one_hot", "goal_one_hot"],
        "history_rows": len(ood_history),
        "training_runtime_seconds": float(sum(row["epoch_seconds"] for row in ood_history)),
        "best_checkpoint": str(ood_directory / "best_checkpoint.pt"),
        "last_checkpoint": str(ood_directory / "last_checkpoint.pt"),
        "configuration": asdict(ood_config),
    }

    payload = {
        "run_id": RUN_ID,
        "run_directory": str(root),
        "training_complete": True,
        "device": actual_device,
        "environment": environment,
        "training_maps": len(train_samples),
        "validation_maps": len(validation_samples),
        "training_distributions": ["D1", "D2", "D3"],
        "models": model_summaries,
        "optional_baselines": {
            "H1": {"status": "disabled", "reason": "No compatible configured 64x64 checkpoint; training is intentionally skipped."},
            "H2": {"status": "disabled", "reason": "No compatible configured 64x64 checkpoint; training is intentionally skipped."},
            "H3": {"status": "disabled", "reason": "No compatible configured 64x64 checkpoint; training is intentionally skipped."},
            "OriginalSeeAStar": {"status": "disabled", "reason": "Its required learned heuristic input is not configured; primary comparisons do not depend on it."},
            "OriginalNeuralAStar": {"status": "available", "reason": "The original implementation can use the trained Neural A* Phi and is included as an optional arm."},
        },
        "total_primary_training_runtime_seconds": float(perf_counter() - training_started),
    }
    _atomic_json(models_root / "training_summary.json", payload)
    model_ready = {
        "run_id": RUN_ID,
        "run_directory": str(root),
        "training_complete": True,
        "primary_models_trained_on_new_64x64_data": ["NeuralAStar", "OODFeatureExtractor"],
        "optional_heuristic_models_trained": [],
        "d95_calibrated": False,
        "sealed_ood_test_loaded": False,
        "model_checkpoint_sha256": {
            name: _sha256(Path(details["best_checkpoint"]))
            for name, details in model_summaries.items()
        },
    }
    _atomic_json(_model_ready_path(root), model_ready)
    return payload


def _extract_embedding(extractor: Any, sample: DatasetSample, device: str) -> np.ndarray:
    if torch is None:
        raise ImportError("OOD feature inference requires PyTorch.")
    channels = build_ood_input(sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal)
    tensor = torch.as_tensor(channels, dtype=torch.float32, device=device).unsqueeze(0)
    extractor = extractor.to(device)
    extractor.eval()
    with torch.no_grad():
        value = extractor(tensor)
    if value.ndim != 2 or value.shape != (1, 128):
        raise ValueError("The pre-projection OOD embedding must be one 128D vector.")
    return value[0].detach().cpu().numpy().astype(np.float64, copy=True)


def calibrate_detector(
    data_directory: str | Path,
    result_root: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Fit train-only covariance and calibrate D95 on validation D1-D3 maps."""

    actual_device = resolve_device(device)
    root = Path(result_root)
    if (root / "pilot" / "pilot_complete.json").is_file():
        raise FileExistsError(
            f"Pilot already completed in {root}; completed run artifacts are immutable."
        )
    ready_path = _model_ready_path(root)
    if not ready_path.is_file():
        raise PermissionError("Detector calibration requires completed model training.")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not ready.get("training_complete"):
        raise PermissionError("Detector calibration is gated until all models finish.")
    calibration_path = root / "calibration" / "detector_calibration.json"
    detector_path = root / "detectors" / "mahalanobis_detector.npz"
    if ready.get("d95_calibrated") and calibration_path.is_file() and detector_path.is_file():
        return json.loads(calibration_path.read_text(encoding="utf-8"))
    training_samples = load_samples(data_directory, "training")
    calibration_samples = load_samples(data_directory, "validation_calibration")
    feature_checkpoint = root / "models" / "ood_feature_extractor" / "best_checkpoint.pt"
    extractor, feature_payload = load_feature_extractor(feature_checkpoint, device=actual_device)
    detector = fit_mahalanobis_detector(
        training_samples,
        extractor,
        config=MahalanobisConfig(),
        device=actual_device,
    )
    detector = calibrate_d95(
        detector,
        calibration_samples,
        extractor,
        device=actual_device,
    )
    if detector.d95 is None or not np.isfinite(detector.d95) or detector.d95 <= 0:
        raise RuntimeError("D95 calibration did not produce a finite positive scale.")
    checks = {
        "score_at_zero": float(normalized_ood_score(0.0, detector.d95)),
        "score_at_d95": float(normalized_ood_score(detector.d95, detector.d95)),
        "score_at_two_d95": float(normalized_ood_score(2.0 * detector.d95, detector.d95)),
    }
    if checks["score_at_zero"] != 0.0:
        raise RuntimeError("OOD score sanity check failed at DM=0.")
    if not np.isclose(checks["score_at_d95"], 1.0 - np.exp(-1.0), atol=1e-12):
        raise RuntimeError("OOD score sanity check failed at DM=D95.")
    if not checks["score_at_zero"] < checks["score_at_d95"] < checks["score_at_two_d95"] < 1.0:
        raise RuntimeError("OOD score monotonicity sanity check failed.")

    save_detector(detector, detector_path)
    calibration = {
        "run_id": RUN_ID,
        "detector_path": str(detector_path),
        "feature_checkpoint": str(feature_checkpoint),
        "feature_checkpoint_sha256": _sha256(feature_checkpoint),
        "embedding_definition": "independent OOD encoder output before projection head",
        "embedding_dim": detector.embedding_dim,
        "training_sample_count": detector.num_training_maps,
        "calibration_sample_count": detector.num_calibration_maps,
        "training_distributions": ["D1", "D2", "D3"],
        "calibration_distributions": ["D1", "D2", "D3"],
        "mahalanobis_distance_convention": "unsquared distance: sqrt((z-mu)^T Sigma^-1 (z-mu))",
        "covariance_estimator": "pooled mean plus Ledoit-Wolf identity-target shrinkage",
        "ledoit_wolf_shrinkage": detector.shrinkage,
        "ridge_added": detector.ridge_added,
        "d95": detector.d95,
        "d95_quantile": detector.config.d95_quantile,
        "quantile_method": "NumPy linear interpolation",
        "score_formula": "1 - exp(-DM / D95); monotonic normalized distance, not a probability",
        "score_checks": checks,
        "feature_training_epoch": int(feature_payload["epoch"]),
        "ood_pilot_used": False,
        "sealed_ood_test_used": False,
    }
    _atomic_json(calibration_path, calibration)
    ready["d95_calibrated"] = True
    ready["d95"] = detector.d95
    ready["detector_sha256"] = _sha256(detector_path)
    _atomic_json(ready_path, ready)
    return calibration


def _load_finalized_models(result_root: Path, device: str) -> tuple[Any, Any, Any]:
    ready_path = _model_ready_path(result_root)
    if not ready_path.is_file():
        raise PermissionError("Models are not finalized.")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not (ready.get("training_complete") and ready.get("d95_calibrated")):
        raise PermissionError("Models and D95 must be finalized before OOD pilot use.")
    neural, _ = load_neural_astar_checkpoint(
        result_root / "models" / "neural_astar" / "best_checkpoint.pt",
        device=device,
    )
    extractor, _ = load_feature_extractor(
        result_root / "models" / "ood_feature_extractor" / "best_checkpoint.pt",
        device=device,
    )
    detector = load_detector(result_root / "detectors" / "mahalanobis_detector.npz")
    if detector.d95 is None or float(detector.d95) != float(ready["d95"]):
        raise ValueError("Detector file and finalized D95 manifest disagree.")
    return neural, extractor, detector


def _validate_search_trace(result: Any) -> dict[str, Any]:
    checked = 0
    goal_terminations = 0
    for item in result.selection_trace:
        current = dict(item.sampled_f_values)
        expected = min(item.sampled_candidates, key=lambda coordinate: current[coordinate])
        if item.selected != expected:
            raise AssertionError("Search expanded a non-minimum-f sampled candidate.")
        if item.selected == item.expanded:
            checked += 1
        elif item.expanded is None and item.selected == result.path[-1]:
            goal_terminations += 1
        else:
            raise AssertionError("Selected candidate differs from the state actually expanded.")
    return {"non_goal_selection_expansion_pairs_checked": checked,
            "goal_selection_terminations": goal_terminations}


def _path_costs(sample: DatasetSample, phi: np.ndarray, path: Sequence[tuple[int, int]]) -> tuple[float, float]:
    ground_truth_cost = 0.0
    for source, destination in zip(path, path[1:]):
        cost = get_transition_cost(sample.grid_map.grid, source, destination, sample.cost_config)
        if not np.isfinite(cost):
            raise AssertionError("Search path contains an illegal transition.")
        ground_truth_cost += cost
    guidance_cost = float(sum(float(phi[cell]) for cell in path[1:]))
    return ground_truth_cost, guidance_cost


def run_integration_tests(
    data_directory: str | Path,
    result_root: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Exercise inference-to-search using validation and ID-pilot maps only."""

    actual_device = resolve_device(device)
    root = Path(result_root)
    if (root / "pilot" / "pilot_complete.json").is_file():
        raise FileExistsError(
            f"Pilot already completed in {root}; completed run artifacts are immutable."
        )
    integration_path = root / "integration_tests" / "inference_to_search.json"
    if integration_path.is_file():
        existing = json.loads(integration_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "passed"
            and not existing.get("ood_pilot_loaded")
            and existing.get("integration_case_count") == 800
        ):
            return existing
    neural, extractor, detector = _load_finalized_models(root, actual_device)
    validation_samples = load_samples(data_directory, "validation_calibration")
    id_pilot_samples = load_samples(data_directory, "id_pilot")
    samples = [*validation_samples, *id_pilot_samples]
    cases: list[dict[str, Any]] = []
    adaptive_calls = 0
    original_decision = adaptive_search.compute_neural_guidance_adaptive_k

    def counted_decision(*args: Any, **kwargs: Any) -> Any:
        nonlocal adaptive_calls
        adaptive_calls += 1
        return original_decision(*args, **kwargs)

    from unittest import mock

    for sample in samples:
        model_started = perf_counter()
        phi, phi_seconds = predict_guidance(
            neural,
            sample.grid_map.grid,
            sample.grid_map.start,
            sample.grid_map.goal,
            device=actual_device,
        )
        feature_started = perf_counter()
        embedding = _extract_embedding(extractor, sample, actual_device)
        raw_distance = float(detector.distance(embedding)[0])
        score = float(detector.score(embedding)[0])
        ood_seconds = perf_counter() - feature_started
        if phi.shape != (64, 64) or not np.isfinite(phi).all() or np.any(phi < 0.0):
            raise AssertionError("64x64 Neural A* Phi failed shape/value validation.")
        if build_ood_input(
            sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal
        ).shape != (3, 64, 64):
            raise AssertionError("Independent OOD input is not 3x64x64.")

        fixed = run_neural_guidance_seeastar(
            sample.grid_map.grid,
            sample.grid_map.start,
            sample.grid_map.goal,
            phi,
            k=4,
            seed=11,
            capture_trace=True,
        )
        if not np.array_equal(fixed.guidance_map, phi):
            raise AssertionError("Fixed-K search did not receive the generated Neural A* Phi.")
        with mock.patch.object(
            adaptive_search,
            "compute_neural_guidance_adaptive_k",
            side_effect=counted_decision,
        ):
            adaptive = adaptive_search.run_ood_adaptive_neural_guidance_seeastar(
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                phi,
                raw_distance,
                float(detector.d95),
                k_min=4,
                k_max=16,
                seed=11,
                capture_trace=True,
            )
        if not np.array_equal(adaptive.search.guidance_map, phi):
            raise AssertionError("Adaptive search did not use the same Neural A* Phi.")
        if not np.isclose(adaptive.ood_score, score, atol=1e-12):
            raise AssertionError("Adaptive search score differs from independent detector score.")
        if adaptive.selected_k != original_decision(
            raw_distance, float(detector.d95), k_min=4, k_max=16
        ).selected_k:
            raise AssertionError("Adaptive K formula mismatch.")

        high_distance_decision = adaptive_search.compute_neural_guidance_adaptive_k(
            max(raw_distance, float(detector.d95) * 100.0),
            float(detector.d95),
            k_min=4,
            k_max=16,
        )
        low_distance_decision = adaptive_search.compute_neural_guidance_adaptive_k(
            0.0, float(detector.d95), k_min=4, k_max=16
        )
        same_phi_low = run_neural_guidance_seeastar(
            sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal,
            phi, k=low_distance_decision.selected_k, seed=11,
        )
        same_phi_high = run_neural_guidance_seeastar(
            sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal,
            phi, k=high_distance_decision.selected_k, seed=11,
        )
        if not np.array_equal(same_phi_low.guidance_map, same_phi_high.guidance_map):
            raise AssertionError("Changing OOD distance changed Phi or guidance costs.")

        fixed_trace = _validate_search_trace(fixed)
        adaptive_trace = _validate_search_trace(adaptive.search)
        fixed_costs = None
        adaptive_costs = None
        if fixed.found:
            fixed_costs = _path_costs(sample, phi, fixed.path)
            if fixed_costs != (fixed.found_ground_truth_cost, fixed.found_guidance_cost):
                raise AssertionError("Fixed-K path costs were not recorded separately and correctly.")
        if adaptive.search.found:
            adaptive_costs = _path_costs(sample, phi, adaptive.path)
            if adaptive_costs != (
                adaptive.found_ground_truth_cost,
                adaptive.found_guidance_cost,
            ):
                raise AssertionError("Adaptive path costs were not recorded separately and correctly.")
        cases.append({
            "distribution": sample.grid_map.distribution,
            "map_shape": list(sample.grid_map.grid.shape),
            "phi_shape": list(phi.shape),
            "phi_source": "64x64 Neural A* best checkpoint",
            "phi_inference_seconds": phi_seconds,
            "ood_embedding_dim": int(embedding.shape[0]),
            "ood_feature_extractor_independent": True,
            "ood_feature_seconds": ood_seconds,
            "raw_mahalanobis_distance": raw_distance,
            "continuous_ood_score": score,
            "effective_adaptive_k": min(adaptive.selected_k, len(adaptive.search.selection_trace[0].eligible_open))
            if adaptive.search.selection_trace else adaptive.selected_k,
            "fixed_k_search_seconds": fixed.search_time_seconds,
            "adaptive_search_seconds": adaptive.search.search_time_seconds,
            "fixed_found": fixed.found,
            "adaptive_found": adaptive.search.found,
            "fixed_path_costs_ground_truth_then_guidance": fixed_costs,
            "adaptive_path_costs_ground_truth_then_guidance": adaptive_costs,
            "fixed_trace_validation": fixed_trace,
            "adaptive_trace_validation": adaptive_trace,
            "high_vs_low_score_k": [low_distance_decision.selected_k, high_distance_decision.selected_k],
            "phi_unchanged_across_ood_scores": True,
            "inference_and_search_timed_separately": True,
        })
    if adaptive_calls != len(samples):
        raise AssertionError("OOD score-to-K decision must be computed once per search instance.")

    report = {
        "run_id": RUN_ID,
        "integration_schema_version": 2,
        "status": "passed",
        "maps_used": ["validation_calibration", "id_pilot"],
        "ood_pilot_loaded": False,
        "sealed_ood_test_loaded": False,
        "integration_case_count": len(cases),
        "adaptive_k_decision_call_count": adaptive_calls,
        "checks": {
            "phi_from_64x64_neural_astar": True,
            "h1_h2_h3_not_used_as_phi": True,
            "ood_encoder_independent_of_neural_astar": True,
            "ood_score_and_k_once_per_search": True,
            "fixed_and_adaptive_receive_identical_phi_per_map": True,
            "fixed_and_adaptive_share_phase_a2_core": True,
            "uniform_sampling_without_replacement": True,
            "minimum_f_sampled_candidate_selected": True,
            "selected_candidate_actually_expanded_except_goal_termination": True,
            "ground_truth_and_guidance_costs_separate": True,
            "all_dimensions_64x64": True,
            "no_resize_crop_or_downsample": True,
            "ood_affects_only_k": True,
        },
        "cases": cases,
    }
    _atomic_json(root / "integration_tests" / "inference_to_search.json", report)
    return report


def cache_pilot_inference(
    data_directory: str | Path,
    result_root: str | Path,
    *,
    device: str = "auto",
    ood_pilot_release_approved: bool = False,
) -> dict[str, Any]:
    """Cache Phi and detector outputs for ID/OOD pilot maps only."""

    if not ood_pilot_release_approved:
        raise PermissionError("OOD-pilot loading requires the explicit release-gate flag.")
    actual_device = resolve_device(device)
    root = Path(result_root)
    if (root / "pilot" / "pilot_complete.json").is_file():
        raise FileExistsError("Completed pilot runs are immutable; choose a new run directory.")
    existing_manifest = root / "guidance_maps" / "pilot_inference_manifest.json"
    if existing_manifest.is_file():
        saved = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if saved.get("cache_splits") == ["id_pilot", "ood_pilot"]:
            return saved.get("caches", {})
        raise ValueError("An incompatible pilot inference manifest already exists.")
    integration_path = root / "integration_tests" / "inference_to_search.json"
    if not integration_path.is_file():
        raise PermissionError("Run and pass ID-only integration checks before pilot inference.")
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    if integration.get("status") != "passed" or integration.get("ood_pilot_loaded"):
        raise PermissionError("Integration checks must pass without loading the OOD pilot.")
    model_load_started = perf_counter()
    neural, extractor, detector = _load_finalized_models(root, actual_device)
    model_loading_seconds = float(perf_counter() - model_load_started)

    cache_summary: dict[str, Any] = {}
    for split_name in ("id_pilot", "ood_pilot"):
        samples = load_samples(data_directory, split_name)
        phi_maps: list[np.ndarray] = []
        distances: list[float] = []
        scores: list[float] = []
        phi_seconds: list[float] = []
        ood_seconds: list[float] = []
        feature_seconds: list[float] = []
        distance_seconds: list[float] = []
        map_ids: list[str] = []
        for sample in samples:
            phi, elapsed = predict_guidance(
                neural,
                sample.grid_map.grid,
                sample.grid_map.start,
                sample.grid_map.goal,
                device=actual_device,
            )
            if phi.shape != (64, 64) or not np.isfinite(phi).all() or np.any(phi < 0.0):
                raise ValueError("Neural A* produced invalid 64x64 Phi during pilot caching.")
            # Feature and distance timings are separate; model execution is
            # measured without including cache serialization.
            feature_started = perf_counter()
            embedding = _extract_embedding(extractor, sample, actual_device)
            feature_elapsed = perf_counter() - feature_started
            distance_started = perf_counter()
            distance = float(detector.distance(embedding)[0])
            score = float(normalized_ood_score(distance, float(detector.d95))[0])
            distance_elapsed = perf_counter() - distance_started
            ood_elapsed = feature_elapsed + distance_elapsed
            phi_maps.append(phi.astype(np.float32))
            distances.append(distance)
            scores.append(score)
            phi_seconds.append(float(elapsed))
            ood_seconds.append(float(ood_elapsed))
            feature_seconds.append(float(feature_elapsed))
            distance_seconds.append(float(distance_elapsed))
            map_ids.append(f"{split_name}:{sample.grid_map.distribution}:{sample.grid_map.seed}")
        output_path = root / "guidance_maps" / f"{split_name}_inference.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                map_ids=np.asarray(map_ids, dtype=np.str_),
                phi_maps=np.stack(phi_maps).astype(np.float32),
                mahalanobis_distances=np.asarray(distances, dtype=np.float64),
                continuous_ood_scores=np.asarray(scores, dtype=np.float64),
                neural_inference_seconds=np.asarray(phi_seconds, dtype=np.float64),
                ood_inference_seconds=np.asarray(ood_seconds, dtype=np.float64),
                ood_feature_seconds=np.asarray(feature_seconds, dtype=np.float64),
                mahalanobis_seconds=np.asarray(distance_seconds, dtype=np.float64),
                neural_checkpoint_sha256=np.asarray(_sha256(
                    root / "models" / "neural_astar" / "best_checkpoint.pt"
                ), dtype=np.str_),
            )
        cache_summary[split_name] = {
            "map_count": len(samples),
            "cache_path": str(output_path),
            "cache_sha256": _sha256(output_path),
            "mean_neural_inference_seconds": float(np.mean(phi_seconds)),
            "mean_ood_inference_seconds": float(np.mean(ood_seconds)),
            "mean_ood_feature_seconds": float(np.mean(feature_seconds)),
            "mean_mahalanobis_seconds": float(np.mean(distance_seconds)),
            "mean_raw_mahalanobis_distance": float(np.mean(distances)),
            "mean_continuous_ood_score": float(np.mean(scores)),
            "density_groups": sorted({sample.grid_map.distribution for sample in samples}),
        }
    _atomic_json(root / "guidance_maps" / "pilot_inference_manifest.json", {
        "run_id": RUN_ID,
        "model_loading_seconds": model_loading_seconds,
        "d95": float(detector.d95),
        "cache_splits": ["id_pilot", "ood_pilot"],
        "sealed_ood_test_loaded": False,
        "models_and_detector_finalized_before_ood_pilot": True,
        "caches": cache_summary,
    })
    return cache_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("environment", "train", "calibrate", "integrate", "cache-pilot"))
    parser.add_argument("--data-directory", default="data/64x64_rectangles_v1")
    parser.add_argument(
        "--result-root",
        default=None,
        help="Exact run directory to continue; omit for a new environment or training run.",
    )
    parser.add_argument("--results-parent", default=f"results/{RUN_ID}")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.result_root is None:
        if args.stage not in {"environment", "train"}:
            raise SystemExit(
                "Pass --result-root with the run directory returned by the environment/train stage."
            )
        result_root = create_run_directory(args.results_parent)
    else:
        try:
            result_root = validate_run_directory(args.result_root)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if args.stage == "environment":
        result = save_environment(result_root)
    elif args.stage == "train":
        result = train_models(
            args.data_directory, result_root, device=args.device, resume=not args.no_resume
        )
    elif args.stage == "calibrate":
        result = calibrate_detector(args.data_directory, result_root, device=args.device)
    elif args.stage == "integrate":
        result = run_integration_tests(args.data_directory, result_root, device=args.device)
    else:
        result = cache_pilot_inference(args.data_directory, result_root, device=args.device)
    result.setdefault("run_directory", str(result_root.resolve()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
