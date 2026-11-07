"""Plots for continuous OOD scores and their empirical relation to h-error."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def plot_ood_score_distributions(
    score_summaries: Sequence[Mapping[str, Any]],
    score_records: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save raw Mahalanobis-distance and normalized-score distribution plots."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("OOD plots require Matplotlib.") from error
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    distributions = [row["distribution"] for row in score_summaries]
    saved: dict[str, Path] = {}
    for key, label, filename in (
        ("mahalanobis_distance", "Raw Mahalanobis distance", "ood_raw_distance_by_distribution.png"),
        ("normalized_score", "Continuous normalized score s", "ood_score_by_distribution.png"),
    ):
        values = [
            [float(row[key]) for row in score_records if row["distribution"] == distribution]
            for distribution in distributions
        ]
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.boxplot(values, labels=distributions, showfliers=True)
        axis.set_xlabel("Map distribution")
        axis.set_ylabel(label)
        axis.set_title(f"{label} by map distribution")
        axis.grid(True, axis="y", alpha=0.25)
        path = destination / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path
    return saved


def plot_heuristic_error_scatter(
    heuristic_error_rows: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save raw-distance/score scatterplots against independently measured E_h."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("OOD plots require Matplotlib.") from error
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    colors = {"D1": "#1f77b4", "D2": "#ff7f0e", "D3": "#2ca02c", "D4": "#d62728", "D5": "#9467bd", "D6": "#8c564b"}
    saved: dict[str, Path] = {}
    for method in sorted({str(row["heuristic_method"]) for row in heuristic_error_rows}):
        selected = [row for row in heuristic_error_rows if row["heuristic_method"] == method]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for distribution in ("D1", "D2", "D3", "D4", "D5", "D6"):
            group = [row for row in selected if row["distribution"] == distribution]
            if not group:
                continue
            for axis, key, label in (
                (axes[0], "ood_distance", "Raw Mahalanobis distance"),
                (axes[1], "ood_score", "Continuous normalized score s"),
            ):
                axis.scatter(
                    [float(row[key]) for row in group],
                    [float(row["heuristic_mae"]) for row in group],
                    label=distribution,
                    color=colors[distribution],
                    alpha=0.75,
                    s=24,
                )
                axis.set_xlabel(label)
                axis.set_ylabel("Heuristic error E_h (map-wise MAE)")
                axis.grid(True, alpha=0.25)
        axes[0].set_title(f"{method}: raw distance vs heuristic error")
        axes[1].set_title(f"{method}: normalized score vs heuristic error")
        axes[1].legend(title="Distribution", fontsize="small")
        filename = f"ood_heuristic_error_scatter_{method}.png"
        path = destination / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path
    return saved
