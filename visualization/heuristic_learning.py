"""Plots training losses and raw heuristic errors/constraint violations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def plot_heuristic_histories(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save loss, prediction-error, and validation-violation learning curves."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Heuristic training plots require Matplotlib.") from error
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    plot_fields = (
        ("loss_curves.png", "loss_total", "Objective loss", True),
        ("prediction_error_curves.png", "heuristic_mae", "Heuristic MAE", False),
        (
            "admissibility_violation_curves.png",
            "admissibility_violation_pct",
            "Admissibility violations (%)",
            False,
        ),
        (
            "consistency_violation_curves.png",
            "consistency_violation_pct",
            "Consistency violations (%)",
            False,
        ),
    )
    saved: dict[str, Path] = {}
    for filename, field, title, plot_both_splits in plot_fields:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for method, rows in histories.items():
            for split in (("train", "validation") if plot_both_splits else ("validation",)):
                points = [
                    row
                    for row in rows
                    if row.get("split") == split
                    and row.get("distribution") == "all"
                    and field in row
                ]
                if not points:
                    continue
                points.sort(key=lambda row: int(row["epoch"]))
                axis.plot(
                    [int(row["epoch"]) for row in points],
                    [float(row[field]) for row in points],
                    label=f"{method} {split}",
                    linestyle="-" if split == "validation" else "--",
                )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend()
        path = destination / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path
    return saved


def plot_ood_heuristic_metrics(
    metrics_by_method: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Plot raw OOD error and both raw constraint violation rates by D4-D6."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Heuristic evaluation plots require Matplotlib.") from error
    distributions = ("D4", "D5", "D6")
    methods = tuple(metrics_by_method)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    plots = (
        ("ood_prediction_error.png", "heuristic_mae", "Raw heuristic MAE"),
        (
            "ood_admissibility_violations.png",
            "admissibility_violation_pct",
            "Raw admissibility violations (%)",
        ),
        (
            "ood_consistency_violations.png",
            "consistency_violation_pct",
            "Raw consistency violations (%)",
        ),
    )
    for filename, field, title in plots:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        bar_width = 0.8 / max(len(methods), 1)
        centers = list(range(len(distributions)))
        for method_index, method in enumerate(methods):
            values = [
                float(metrics_by_method[method].get(distribution, {}).get(field, 0.0))
                for distribution in distributions
            ]
            offsets = [
                center - 0.4 + bar_width * (method_index + 0.5)
                for center in centers
            ]
            axis.bar(offsets, values, width=bar_width, label=method)
        axis.set_xticks(centers, distributions)
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        path = destination / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path
    return saved
