"""One-per-map model inference cache for the staged benchmark pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from adaptive.ood_adaptive_seeastar import extract_map_embedding
from benchmarks.data import load_evaluation_splits
from benchmarks.reliability import heuristic_error_metrics
from data.dataset import DatasetSample, map_fingerprint
from heuristics.inference import load_heuristic_checkpoint, predict_heuristic_map
from neural_astar.inference import load_neural_astar_checkpoint, predict_guidance
from ood.contrastive import load_feature_extractor
from ood.mahalanobis import load_detector, normalized_ood_score


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_models(model_root: str | Path, *, device: str = "cpu") -> tuple[dict[str, Any], dict[str, float]]:
    """Load the four CNNs and detector once, returning independent load times."""

    root = Path(model_root)
    models: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for method in ("H1", "H2", "H3"):
        started = perf_counter()
        models[method], _ = load_heuristic_checkpoint(
            root / "heuristics" / method / "best_checkpoint.pt", device=device
        )
        timings[f"load_{method}_seconds"] = perf_counter() - started
    started = perf_counter()
    models["NeuralAStar"], _ = load_neural_astar_checkpoint(
        root / "neural_astar" / "best_checkpoint.pt", device=device
    )
    timings["load_neural_astar_seconds"] = perf_counter() - started
    started = perf_counter()
    models["OODFeature"], _ = load_feature_extractor(
        root / "ood_feature" / "feature_extractor.pt", device=device
    )
    timings["load_ood_feature_seconds"] = perf_counter() - started
    started = perf_counter()
    models["OODDetector"] = load_detector(root / "ood_feature" / "mahalanobis_detector.npz")
    timings["load_ood_detector_seconds"] = perf_counter() - started
    detector = models["OODDetector"]
    if detector.d95 is None or detector.d95 <= detector.config.d95_minimum:
        raise ValueError("The saved OOD detector has no safely calibrated D95.")
    return models, timings


def evaluate_map_inference(
    samples: Sequence[DatasetSample],
    models: dict[str, Any],
    *,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Predict all map-conditioned outputs once and retain raw maps unchanged."""

    detector = models["OODDetector"]
    if detector.d95 is None:
        raise ValueError("The detector must be calibrated before evaluation inference.")
    rows: list[dict[str, Any]] = []
    maps: dict[str, list[np.ndarray]] = {name: [] for name in ("H1", "H2", "H3", "phi")}
    for sample_index, sample in enumerate(samples):
        grid = sample.grid_map.grid
        start, goal = sample.grid_map.start, sample.grid_map.goal
        map_id = f"{sample.grid_map.distribution}:{sample.grid_map.seed}"
        entry: dict[str, Any] = {
            "sample_index": sample_index,
            "map_id": map_id,
            "map_fingerprint": map_fingerprint(grid),
            "split": "id_holdout" if sample.grid_map.distribution in {"D1", "D2", "D3"} else "ood_test",
            "distribution": sample.grid_map.distribution,
            "map_seed": int(sample.grid_map.seed),
            "start": list(start),
            "goal": list(goal),
            "optimal_ground_truth_cost": float(sample.ground_truth.optimal_path_cost),
        }
        for method in ("H1", "H2", "H3"):
            started = perf_counter()
            prediction = predict_heuristic_map(
                models[method], grid, start, goal, method=method, device=device
            )
            inference_seconds = perf_counter() - started
            maps[method].append(prediction)
            entry[f"{method}_cnn_inference_seconds"] = float(inference_seconds)
            errors = heuristic_error_metrics(sample, prediction)
            for metric, value in errors.items():
                entry[f"{method}_{metric}"] = value

        phi, neural_seconds = predict_guidance(models["NeuralAStar"], grid, start, goal, device=device)
        maps["phi"].append(phi)
        entry["neural_astar_cnn_inference_seconds"] = float(neural_seconds)

        feature_seconds_started = perf_counter()
        embedding = extract_map_embedding(models["OODFeature"], grid, start, goal, device=device)
        feature_seconds = perf_counter() - feature_seconds_started
        detector_started = perf_counter()
        distance = float(detector.distance(embedding)[0])
        score = float(normalized_ood_score(distance, float(detector.d95)))
        detector_seconds = perf_counter() - detector_started
        entry.update(
            {
                "ood_feature_cnn_inference_seconds": float(feature_seconds),
                "ood_detector_inference_seconds": float(detector_seconds),
                "ood_total_inference_seconds": float(feature_seconds + detector_seconds),
                "mahalanobis_distance": distance,
                "smooth_ood_score": score,
                "d95": float(detector.d95),
            }
        )
        rows.append(entry)
    arrays = {
        name: np.stack(values).astype(np.float64, copy=False)
        for name, values in maps.items()
    }
    return rows, arrays


def prepare_evaluation_inference_cache(
    output_directory: str | Path,
    *,
    model_ready_manifest: str | Path,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Open evaluation splits only after training and D95 calibration are gated."""

    root = Path(output_directory)
    cache_root = root / "evaluation"
    cache_path = cache_root / "map_inference_cache.npz"
    rows_path = cache_root / "per_map_inference_and_reliability.csv"
    manifest_path = cache_root / "inference_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluation cache already exists at {cache_root}.")
    id_samples, ood_samples = load_evaluation_splits(
        root / "data", model_ready_manifest=model_ready_manifest
    )
    samples = [*id_samples, *ood_samples]
    if len(id_samples) != 90 or len(ood_samples) != 90:
        raise ValueError("Evaluation split counts differ from the approved 90+90 configuration.")
    models, loading_seconds = load_models(root / "models", device=device)
    rows, arrays = evaluate_map_inference(samples, models, device=device)
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        H1=arrays["H1"],
        H2=arrays["H2"],
        H3=arrays["H3"],
        phi=arrays["phi"],
        map_ids=np.asarray([row["map_id"] for row in rows], dtype=np.str_),
        map_fingerprints=np.asarray([row["map_fingerprint"] for row in rows], dtype=np.str_),
    )
    _write_rows(rows_path, rows)
    manifest = {
        "schema_version": 1,
        "created_after_training_and_d95_calibration": True,
        "ood_test_opened_after_model_ready_gate": True,
        "id_holdout_map_count": len(id_samples),
        "ood_test_map_count": len(ood_samples),
        "map_count": len(samples),
        "d95": float(models["OODDetector"].d95),
        "model_loading_seconds": loading_seconds,
        "inference_cache": str(cache_path),
        "per_map_metrics": str(rows_path),
        "raw_prediction_maps_preserved": True,
        "neural_astar_inference_included": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest, "rows": rows, "arrays": arrays, "samples": samples}


def load_inference_cache(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Restore cached raw map-level prediction fields without pickle loading."""

    with np.load(Path(path), allow_pickle=False) as archive:
        ids = [str(value) for value in archive["map_ids"]]
        arrays = {name: archive[name].astype(np.float64, copy=True) for name in ("H1", "H2", "H3", "phi")}
    if len(set(ids)) != len(ids):
        raise ValueError("Inference cache contains duplicate map ids.")
    if any(values.shape[0] != len(ids) for values in arrays.values()):
        raise ValueError("Inference cache arrays are not aligned with map ids.")
    return ids, arrays
