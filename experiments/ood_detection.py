"""Train, calibrate, and evaluate the feature-space OOD detector."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.dataset import DatasetSample, assert_no_map_leakage
from ood.contrastive import (
    ContrastiveTrainingConfig,
    train_contrastive_feature_extractor,
)
from ood.evaluation import evaluate_ood_detector, save_evaluation_csvs
from ood.mahalanobis import (
    MahalanobisConfig,
    calibrate_d95,
    fit_mahalanobis_detector,
    save_detector,
)
from visualization.ood import (
    plot_heuristic_error_scatter,
    plot_ood_score_distributions,
)


def _json_safe(value: Any) -> Any:
    """Replace non-finite metrics with JSON null and normalize containers."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_splits(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    ood_samples: Sequence[DatasetSample],
) -> None:
    if len(train_samples) < 2 or not validation_samples or not ood_samples:
        raise ValueError("Need >=2 training maps and non-empty validation/OOD splits.")
    for label, samples, allowed in (
        ("train", train_samples, {"D1", "D2", "D3"}),
        ("validation", validation_samples, {"D1", "D2", "D3"}),
        ("OOD test", ood_samples, {"D4", "D5", "D6"}),
    ):
        for index, sample in enumerate(samples):
            if sample.grid_map.distribution not in allowed:
                raise ValueError(f"{label}[{index}] must use one of {sorted(allowed)}.")
        if label in {"train", "validation"} and {
            sample.grid_map.distribution for sample in samples
        } != allowed:
            raise ValueError(f"{label} must include each of D1, D2, and D3.")
    if {sample.grid_map.distribution for sample in ood_samples} != {"D4", "D5", "D6"}:
        raise ValueError("OOD evaluation must contain samples from each of D4, D5, and D6.")
    shapes = {
        tuple(sample.grid_map.grid.shape)
        for split in (train_samples, validation_samples, ood_samples)
        for sample in split
    }
    if len(shapes) != 1:
        raise ValueError("All feature maps must share fixed dimensions.")
    assert_no_map_leakage(
        {"train": train_samples, "validation": validation_samples, "ood_test": ood_samples}
    )


def run_ood_detection(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    ood_test_samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    training_config: ContrastiveTrainingConfig = ContrastiveTrainingConfig(),
    detector_config: MahalanobisConfig = MahalanobisConfig(),
    heuristic_models: Mapping[str, tuple[Any, str]] | None = None,
    bootstrap_repetitions: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run independent CNN training, ID-only fit/calibration, and evaluation.

    The D4-D6 samples are first passed to the detector after feature training,
    statistics fitting, and D95 calibration have completed.
    """

    _validate_splits(train_samples, validation_samples, ood_test_samples)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    feature_model, feature_history = train_contrastive_feature_extractor(
        train_samples,
        root,
        config=training_config,
    )
    detector = fit_mahalanobis_detector(
        train_samples,
        feature_model,
        config=detector_config,
        device=training_config.device,
    )
    calibrate_d95(
        detector,
        validation_samples,
        feature_model,
        device=training_config.device,
    )
    save_detector(detector, root / "mahalanobis_detector.npz")

    results = evaluate_ood_detector(
        detector,
        feature_model,
        validation_samples,
        ood_test_samples,
        heuristic_models=heuristic_models,
        device=training_config.device,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    results["feature_training_history"] = feature_history
    results["configuration"] = {
        "feature_training": asdict(training_config),
        "mahalanobis": asdict(detector_config),
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "training_distributions": ["D1", "D2", "D3"],
        "calibration_distributions": ["D1", "D2", "D3"],
        "ood_evaluation_distributions": ["D4", "D5", "D6"],
    }
    (root / "ood_evaluation.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8"
    )
    save_evaluation_csvs(results, root)
    plots = plot_ood_score_distributions(
        results["score_summaries_by_distribution"], results["scores"], root / "plots"
    )
    plots.update(plot_heuristic_error_scatter(results["heuristic_errors"], root / "plots"))
    return {
        "feature_extractor": feature_model,
        "detector": detector,
        "results": results,
        "plots": plots,
        "output_directory": root,
    }
