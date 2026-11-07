"""Plots, human-readable pilot report, and downloadable Phase B result bundle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import zipfile
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _plot_pilot(root: Path, raw: list[dict[str, str]], aggregate: list[dict[str, str]]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - Kaggle runtime dependency
        raise ImportError("matplotlib is required to produce the requested pilot plots.") from error
    output = root / "pilot" / "plots"
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = output / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    def box(metric: str, title: str, ylabel: str, name: str, *, success_only: bool = False) -> None:
        methods = sorted({r["algorithm"] for r in raw})
        values, labels = [], []
        for method in methods:
            vals = [
                _number(r, metric) for r in raw
                if r["algorithm"] == method
                and (not success_only or r.get("success", "").lower() == "true")
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                values.append(vals)
                labels.append(method)
        fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(labels)), 6))
        if values:
            ax.boxplot(values, labels=labels, showfliers=False)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
        else:
            ax.text(0.5, 0.5, "No valid observations", ha="center", va="center")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        save(fig, name)

    box("expanded_nodes", "Expanded nodes by algorithm", "Expanded nodes (goal included)", "01_expanded_nodes.png")
    box("search_time_ms", "Search runtime by algorithm", "Search-only runtime (ms)", "02_search_runtime.png")
    box("generated_nodes", "Generated unique nodes by algorithm", "Generated unique nodes", "03_generated_nodes.png")
    box("ground_truth_path_cost", "Ground-truth path cost by algorithm", "Legal transitions (unit cost)", "04_ground_truth_path_cost.png", success_only=True)
    box("path_cost_ratio", "Path-cost ratio relative to Dijkstra", "C_GT(path) / C*_GT", "05_path_cost_ratio.png", success_only=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    methods = sorted({r["algorithm"] for r in aggregate})
    groups = sorted({r["density_group"] for r in aggregate})
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        rows = {r["density_group"]: r for r in aggregate if r["algorithm"] == method}
        vals = [float(rows[g]["success_rate"]) if g in rows else np.nan for g in groups]
        ax.bar(x + i * width, vals, width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2, groups)
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by density distribution")
    ax.set_ylabel("Map-level success rate")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "06_success_rate_by_algorithm.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title, ylabel in (
        (axes[0], "mean_expanded_nodes", "Expanded nodes by D1-D6", "Map-averaged expansions"),
        (axes[1], "mean_search_time_seconds", "Search runtime by D1-D6", "Seconds"),
    ):
        for method in sorted({r["algorithm"] for r in aggregate}):
            rows = [r for r in aggregate if r["algorithm"] == method]
            rows.sort(key=lambda r: r["density_group"])
            ax.plot([r["density_group"] for r in rows], [_number(r, metric) for r in rows], marker="o", label=method)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Density group")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    save(fig, "07_results_by_density_D1_D6.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    methods = ["neural_guidance_fixed_k4", "neural_guidance_fixed_k5", "neural_guidance_fixed_k10", "neural_guidance_fixed_k16", "neural_guidance_adaptive_k4_16"]
    labels = ["Fixed K=4", "Fixed K=5", "Fixed K=10", "Fixed K=16", "Adaptive 4..16"]
    groups = sorted({r["density_group"] for r in aggregate})
    for method, label in zip(methods, labels):
        rows = {r["density_group"]: r for r in aggregate if r["method"] == method}
        ax.plot(groups, [_number(rows[g], "mean_expanded_nodes") if g in rows else np.nan for g in groups], marker="o", label=label)
    ax.set_title("Adaptive-K versus fixed-K Neural-guidance SeeA*")
    ax.set_xlabel("Density group")
    ax.set_ylabel("Map-averaged expanded nodes")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "08_adaptive_vs_fixed_K.png")

    fixed = [r for r in raw if r["method_family"] == "neural_guidance_fixed"]
    for metric, title, ylabel, name in (
        ("expanded_nodes", "OOD score versus expanded nodes", "Expanded nodes", "09_ood_score_vs_expanded.png"),
        ("search_time_ms", "OOD score versus search runtime", "Search-only runtime (ms)", "10_ood_score_vs_runtime.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        for group in sorted({r["density_group"] for r in fixed}):
            pairs = [(float(r["continuous_ood_score"]), _number(r, metric)) for r in fixed if r["density_group"] == group]
            pairs = [(x, y) for x, y in pairs if y is not None]
            if pairs:
                ax.scatter([x for x, _ in pairs], [y for _, y in pairs], s=15, alpha=0.55, label=group)
        ax.set_title(title)
        ax.set_xlabel("Continuous OOD score (not a probability)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.25)
        save(fig, name)

    neural_history = _read_csv(root / "models" / "neural_astar" / "losses.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    if neural_history:
        ax.plot([float(r["epoch"]) for r in neural_history], [float(r["train_loss"]) for r in neural_history], label="Training")
        ax.plot([float(r["epoch"]) for r in neural_history], [float(r["validation_loss"]) for r in neural_history], label="Validation")
    ax.set_title("Neural A* training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L1 loss")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "11_neural_astar_learning_curves.png")

    ood_history = _read_csv(root / "models" / "ood_feature_extractor" / "feature_history.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    if ood_history:
        ax.plot([float(r["epoch"]) for r in ood_history], [float(r["contrastive_loss"]) for r in ood_history], label="Training")
        validation_values = [r.get("validation_contrastive_loss", "") for r in ood_history]
        if validation_values and all(v != "" for v in validation_values):
            ax.plot([float(r["epoch"]) for r in ood_history], [float(v) for v in validation_values], label="Validation")
    ax.set_title("Independent OOD encoder contrastive loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "12_ood_encoder_learning_curves.png")
    return saved


def _make_report(root: Path, summary: dict[str, Any], aggregate: list[dict[str, str]], stats: dict[str, Any]) -> Path:
    training = _json(root / "models" / "training_summary.json")
    calibration = _json(root / "calibration" / "detector_calibration.json")
    environment = _json(root / "environment.json")
    rows = [
        "# Phase B experimental report",
        "",
        f"Run: `{summary.get('run_id')}/{summary.get('run_variant')}`",
        "",
        "## Scope and design",
        "",
        "The pilot evaluates the same obstacle grid, start and goal for each algorithm. ID and OOD pilot maps are separate, and the independent unit for comparisons is the map layout. Stochastic seeds are averaged within each map before paired inference. This is pilot evidence only; it is not the final benchmark.",
        "",
        "Movement is eight-connected, diagonal corner-cutting is prohibited, and every legal transition costs 1.0. Expanded nodes count valid non-stale OPEN removals including the goal. Generated nodes count unique first discoveries inserted into OPEN, including the start.",
        "",
        "Euclidean A* is reported as potentially non-admissible under unit-cost diagonal movement. The modified Neural-guidance SeeA* result is not claimed to be optimal. Guidance cost, transition cost, number of moves and geometric length are separate quantities.",
        "",
        "## Training and detector",
        "",
        f"- Neural A*: {json.dumps(training.get('models', {}).get('NeuralAStar', {}), sort_keys=True)}",
        f"- Independent OOD encoder: {json.dumps(training.get('models', {}).get('OODFeatureExtractor', {}), sort_keys=True)}",
        f"- Detector calibration: {json.dumps(calibration, sort_keys=True)}",
        f"- Runtime environment: {json.dumps({k: environment.get(k) for k in ('python_version','pytorch_version','cuda_version','gpu_model','gpu_memory_bytes','cpu_model','logical_cpu_count','system_memory')}, sort_keys=True)}",
        "",
        "## Compared algorithms",
        "",
        "Mandatory: Dijkstra, A* with Chebyshev, A* with Euclidean, fixed-K Neural-guidance SeeA* (K=4,5,10,16), and OOD-adaptive K (bounds 4..16). Original Neural A* is included as an optional baseline. Original SeeA* and H1/H2/H3 remain disabled unless compatible explicit inputs are configured.",
        "",
        "## Density-stratified results",
        "",
        "| Split | Group | Algorithm | Maps | Runs | Success | Mean expanded | Median expanded | Mean generated | Median generated | Mean search seconds | Median search seconds | Mean path cost | Mean Dijkstra cost | Mean gap | Mean ratio | Mean moves | Mean geometric length | Mean guidance cost | Timeouts |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        fields = [
            r.get("split", ""), r.get("density_group", ""), r.get("algorithm", ""), r.get("number_of_maps", ""), r.get("number_of_runs", ""),
            _fmt(r.get("success_rate")), _fmt(r.get("mean_expanded_nodes")), _fmt(r.get("median_expanded_nodes")),
            _fmt(r.get("mean_generated_nodes")), _fmt(r.get("median_generated_nodes")),
            _fmt(r.get("mean_search_time_seconds")), _fmt(r.get("median_search_time_seconds")),
            _fmt(r.get("mean_ground_truth_path_cost")), _fmt(r.get("mean_dijkstra_optimal_cost")),
            _fmt(r.get("mean_absolute_optimality_gap")), _fmt(r.get("mean_path_cost_ratio")),
            _fmt(r.get("mean_number_of_moves")), _fmt(r.get("mean_geometric_path_length")),
            _fmt(r.get("mean_guidance_accumulated_cost")), r.get("timeout_count", ""),
        ]
        rows.append("| " + " | ".join(str(x) for x in fields) + " |")
    rows += [
        "",
        "## Paired map-level inference and OOD associations",
        "",
        "Paired bootstrap intervals and sign-flip tests use map-level values after averaging repeated seeds. Holm-adjusted p-values are reported in `pilot/aggregate/paired_statistics.json`. The same file includes fixed-K Spearman correlations of the continuous OOD score with search effort and path quality by density group. Pilot results are not used to select the final algorithm.",
        "",
        f"Statistics file: `pilot/aggregate/paired_statistics.json` ({len(stats.get('comparisons', []))} paired comparisons; {len(stats.get('ood_score_correlations_fixed_k_only', []))} correlation records).",
        "",
        "## Timing and inference accounting",
        "",
        "Search time is measured inside each graph-search implementation with a monotonic timer and excludes model inference, model loading, serialization and plotting. Neural A* Phi, OOD embeddings, Mahalanobis distances and scores are computed once per map and cached. The raw CSV repeats those shared per-map times as provenance; per-search end-to-end time adds only the inference required by that algorithm. Aggregate search runtime remains search-only.",
        "",
        "## Reliability, failures and limitations",
        "",
        "Timeouts, exceptions, invalid paths, unsuccessful searches and numerical or memory failures remain in the raw rows. Aggregates use map-level seed averages. A reduction in expansions does not imply a reduction in runtime because uniform sampling adds work per selection. OOD distance is not assumed to imply poor guidance. Results are preliminary pilot evidence and cannot establish confirmatory superiority.",
        "",
        "## Artifacts",
        "",
        "- Raw results: `pilot/raw/search_results.csv`",
        "- Same-map comparison: `pilot/per_map/per_map_comparison.csv`",
        "- Aggregate table: `pilot/aggregate/algorithm_comparison.csv`",
        "- Paired analysis: `pilot/aggregate/paired_statistics.json`",
        "- Sampling traces: `pilot/raw/sampling_traces.jsonl` (first fixed-K and adaptive trace per density group and split)",
        "- Plots: `pilot/plots/`",
    ]
    path = root / "reports" / "phase_b_experimental_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.4f}" if math.isfinite(numeric) else "—"


def finalize_phase_b_results(
    result_root: str | Path,
    *,
    pilot_summary: dict[str, Any],
) -> Path:
    """Create plots, report, README and a self-contained downloadable ZIP."""

    root = Path(result_root)
    zip_path = root / "phase_b_results.zip"
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result package: {zip_path}")
    raw = _read_csv(root / "pilot" / "raw" / "search_results.csv")
    aggregate = _read_csv(root / "pilot" / "aggregate" / "algorithm_comparison.csv")
    stats = _json(root / "pilot" / "aggregate" / "paired_statistics.json")
    plots = _plot_pilot(root, raw, aggregate)
    report = _make_report(root, pilot_summary, aggregate, stats)
    readme = root / "README_results.md"
    readme.write_text(
        "# Phase B pilot results\n\n"
        "This package contains the local-run training and pilot results for `64x64_rectangles_v1`. "
        "The primary outputs are `reports/phase_b_experimental_report.md`, "
        "`pilot/raw/search_results.csv`, `pilot/per_map/per_map_comparison.csv`, and "
        "`pilot/aggregate/algorithm_comparison.csv`. Plots are in `pilot/plots/`. "
        "Search timings exclude inference; shared model and detector inference times are recorded separately. "
        "Repeated seeds are averaged within each map before paired statistics. Failures remain in the raw CSV.\n",
        encoding="utf-8",
    )
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "phase_b_results.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path in {zip_path, temporary}:
                continue
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
    temporary.replace(zip_path)
    if not plots or not report.is_file() or not readme.is_file() or zip_path.stat().st_size == 0:
        raise RuntimeError("Phase B result package is incomplete.")
    return zip_path
