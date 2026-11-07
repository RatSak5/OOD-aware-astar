"""Static research plots for fixed and OOD-adaptive uniform SeeA* runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def plot_adaptive_seeastar_results(
    rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write score/K, per-distribution, efficiency, and paired-comparison plots."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Adaptive SeeA* plots require Matplotlib.") from error
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    adaptive_methods = {
        "adaptive_high_ood_smaller_k",
        "adaptive_reverse_high_ood_larger_k",
        "adaptive_clipped_high_ood_smaller_k",
    }
    palette = {
        "fixed_k_min": "#4c78a8",
        "fixed_k_mid": "#72b7b2",
        "fixed_k_max": "#54a24b",
        "adaptive_high_ood_smaller_k": "#e45756",
        "adaptive_reverse_high_ood_larger_k": "#f2cf5b",
        "adaptive_clipped_high_ood_smaller_k": "#b279a2",
    }

    adaptive_rows = [row for row in rows if row["method"] in adaptive_methods]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in sorted(adaptive_methods):
        group = [row for row in adaptive_rows if row["method"] == method]
        if group:
            axis.scatter(
                [float(row["policy_ood_score"]) for row in group],
                [int(row["selected_k"]) for row in group],
                label=method,
                color=palette[method],
                alpha=0.7,
                s=26,
            )
    axis.set_xlabel("Continuous policy OOD score")
    axis.set_ylabel("K selected once for the map")
    axis.set_title("Uniform SeeA*: OOD score versus selected K")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    filename = "adaptive_ood_score_vs_k.png"
    path = destination / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved[filename] = path

    proposed = [
        row for row in rows if row["method"] == "adaptive_high_ood_smaller_k"
    ]
    distributions = ("D1", "D2", "D3", "D4", "D5", "D6")
    values = [
        [int(row["selected_k"]) for row in proposed if row["distribution"] == name]
        for name in distributions
    ]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot(values, labels=distributions, showfliers=True)
    axis.set_xlabel("Map distribution")
    axis.set_ylabel("Proposed adaptive K")
    axis.set_title("Adaptive K by map distribution")
    axis.grid(True, axis="y", alpha=0.25)
    filename = "adaptive_k_by_distribution.png"
    path = destination / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved[filename] = path

    metric_specs = (
        ("expanded_nodes", "Expanded nodes", "adaptive_expansions_vs_k.png"),
        (
            "expanded_grid_coverage",
            "Unique expanded cells / traversable cells",
            "adaptive_expansion_coverage_vs_k.png",
        ),
        (
            "expanded_spatial_spread",
            "Normalized spatial spread of expanded cells",
            "adaptive_expansion_spread_vs_k.png",
        ),
        (
            "search_runtime_seconds",
            "Search runtime (seconds)",
            "adaptive_search_runtime_vs_k.png",
        ),
        (
            "total_end_to_end_seconds",
            "End-to-end runtime (seconds)",
            "adaptive_runtime_vs_k.png",
        ),
        (
            "sampling_overhead_seconds",
            "Sampling overhead (seconds)",
            "adaptive_sampling_overhead_vs_k.png",
        ),
        ("optimality_ratio", "Optimality ratio", "adaptive_optimality_ratio_vs_k.png"),
    )
    for metric, y_label, filename in metric_specs:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for method in sorted(palette):
            group = [
                row
                for row in rows
                if row["method"] == method
                and row.get(metric) is not None
                and str(row.get(metric)) not in {"nan", "inf", "-inf"}
            ]
            if not group:
                continue
            axis.scatter(
                [int(row["selected_k"]) for row in group],
                [float(row[metric]) for row in group],
                label=method,
                color=palette[method],
                alpha=0.55,
                s=22,
            )
        axis.set_xlabel("Selected or fixed K")
        axis.set_ylabel(y_label)
        axis.set_title(f"{y_label} versus K")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="x-small", ncol=2)
        path = destination / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path

    paired_methods = sorted(
        {
            (str(row["adaptive_method"]), str(row["fixed_reference"]))
            for row in paired_rows
        }
    )
    paired_metrics = (
        ("delta_expanded_nodes", "Adaptive minus fixed K: expansions"),
        ("delta_total_end_to_end_seconds", "Adaptive minus fixed K: runtime (s)"),
        ("delta_optimality_ratio", "Adaptive minus fixed K: optimality ratio"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (metric, label) in zip(axes, paired_metrics):
        data = [
            [
                float(row[metric])
                for row in paired_rows
                if (row["adaptive_method"], row["fixed_reference"]) == method
                and row.get(metric) is not None
                and str(row.get(metric)) not in {"nan", "inf", "-inf"}
            ]
            for method in paired_methods
        ]
        if any(data):
            axis.boxplot(
                data,
                labels=[
                    f"{adaptive.replace('adaptive_', '')}\nvs {fixed.replace('fixed_k_', 'K')}"
                    for adaptive, fixed in paired_methods
                ],
            )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelrotation=35, labelsize="x-small")
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Paired adaptive versus fixed-K comparisons")
    filename = "adaptive_vs_fixed_paired_deltas.png"
    path = destination / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved[filename] = path
    return saved

