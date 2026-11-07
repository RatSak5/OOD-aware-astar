"""Train H1/H2/H3 independently, then evaluate held-out D4-D6 maps."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from data.dataset import DatasetSample, assert_no_map_leakage
from heuristics.evaluation import evaluate_heuristic_model
from training.heuristic_training import HeuristicTrainingConfig, train_heuristic_model
from visualization.heuristic_learning import (
    plot_heuristic_histories,
    plot_ood_heuristic_metrics,
)


def _validate_comparison_splits(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    ood_samples: Sequence[DatasetSample],
) -> tuple[int, int]:
    if not train_samples or not validation_samples or not ood_samples:
        raise ValueError("Train, validation, and OOD test splits must all be non-empty.")
    for split_name, samples, allowed in (
        ("training", train_samples, {"D1", "D2", "D3"}),
        ("validation", validation_samples, {"D1", "D2", "D3"}),
        ("OOD test", ood_samples, {"D4", "D5", "D6"}),
    ):
        for index, sample in enumerate(samples):
            if sample.grid_map.distribution not in allowed:
                raise ValueError(
                    f"{split_name} sample {index} has distribution "
                    f"{sample.grid_map.distribution}, expected one of {sorted(allowed)}."
                )
    shapes = {
        tuple(sample.grid_map.grid.shape)
        for split in (train_samples, validation_samples, ood_samples)
        for sample in split
    }
    if len(shapes) != 1:
        raise ValueError("All splits must share dimensions for comparable H3 class support.")
    assert_no_map_leakage(
        {"train": train_samples, "validation": validation_samples, "ood_test": ood_samples}
    )
    height, width = next(iter(shapes))
    return int(height), int(width)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_heuristic_comparison(
    train_samples: Sequence[DatasetSample],
    validation_samples: Sequence[DatasetSample],
    ood_test_samples: Sequence[DatasetSample],
    output_directory: str | Path,
    *,
    base_config: HeuristicTrainingConfig = HeuristicTrainingConfig(),
) -> dict[str, Any]:
    """Train all three methods; use D4-D6 only for post-selection evaluation.

    The returned mapping contains models, per-epoch histories, and separate
    validation/OOD metrics. OOD maps are never passed to training, scheduling,
    early stopping, or checkpoint selection.
    """

    _validate_comparison_splits(train_samples, validation_samples, ood_test_samples)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    histories: dict[str, list[dict[str, float | str]]] = {}
    validation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    ood_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for method in ("H1", "H2", "H3"):
        method_config = replace(base_config, method=method)
        method_root = root / method
        model, history = train_heuristic_model(
            train_samples,
            validation_samples,
            method_root,
            config=method_config,
        )
        models[method] = model
        histories[method] = history
        validation_metrics[method] = evaluate_heuristic_model(
            model,
            validation_samples,
            method,
            device=method_config.device,
            composite=method_config.composite_loss,
            cea=method_config.cea_loss,
            progress_description=f"{method} final validation",
        )
        # This is intentionally the first use of the OOD samples for each model.
        ood_metrics[method] = evaluate_heuristic_model(
            model,
            ood_test_samples,
            method,
            device=method_config.device,
            composite=method_config.composite_loss,
            cea=method_config.cea_loss,
            progress_description=f"{method} D4-D6 evaluation",
        )

    validation_rows = [
        {"method": method, **metrics}
        for method, grouped in validation_metrics.items()
        for metrics in grouped.values()
    ]
    ood_rows = [
        {"method": method, **metrics}
        for method, grouped in ood_metrics.items()
        for metrics in grouped.values()
    ]
    _write_metrics_csv(root / "validation_metrics.csv", validation_rows)
    _write_metrics_csv(root / "ood_metrics.csv", ood_rows)
    plot_heuristic_histories(histories, root / "plots")
    plot_ood_heuristic_metrics(ood_metrics, root / "plots")
    return {
        "models": models,
        "histories": histories,
        "validation_metrics": validation_metrics,
        "ood_metrics": ood_metrics,
        "output_directory": root,
    }
