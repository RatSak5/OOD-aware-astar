"""Plots that keep Neural A* guidance and its search trace distinct."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from algorithms.movement import Coord, validate_map
from neural_astar.search import NeuralAStarResult


def visualize_neural_astar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    result: NeuralAStarResult,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot map, Phi, entry charges, accumulated G, selected order, and path.

    The displayed entry-charge map is ``Phi`` with the start set to zero,
    reflecting the search initialization ``G(start)=0``. It is not a new
    transition-cost rule. The ground-truth unit move cost is not visualized as
    a guidance quantity.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on local packages
        raise ImportError("Neural A* visualizations require matplotlib.") from error

    obstacle_map = validate_map(grid)
    if result.guidance_map.shape != obstacle_map.shape:
        raise ValueError("Search result and obstacle map shapes must match.")
    if result.accumulated_guidance_cost.shape != obstacle_map.shape:
        raise ValueError("Accumulated guidance-cost map shape does not match obstacle map.")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    free_map = 1 - obstacle_map
    axes_flat[0].imshow(free_map, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes_flat[0].scatter([start[1]], [start[0]], marker="o", color="lime", label="Start")
    axes_flat[0].scatter([goal[1]], [goal[0]], marker="*", color="red", label="Goal")
    axes_flat[0].set_title("Obstacle map")
    axes_flat[0].legend(loc="upper right")

    obstacle_mask = obstacle_map.astype(bool)
    phi_view = np.ma.masked_where(obstacle_mask, result.guidance_map)
    axes_flat[1].imshow(phi_view, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes_flat[1].set_title("Predicted guidance map Φ")
    figure.colorbar(axes_flat[1].images[0], ax=axes_flat[1], fraction=0.046)

    entry_cost = result.guidance_map.copy()
    entry_cost[start] = 0.0
    entry_view = np.ma.masked_where(obstacle_mask, entry_cost)
    axes_flat[2].imshow(entry_view, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes_flat[2].set_title("Node-entry guidance charge Φ (start=0)")
    figure.colorbar(axes_flat[2].images[0], ax=axes_flat[2], fraction=0.046)

    accumulated_view = np.ma.masked_where(obstacle_mask, result.accumulated_guidance_cost)
    axes_flat[3].imshow(accumulated_view, cmap="magma", interpolation="nearest")
    axes_flat[3].set_title("Accumulated guidance cost Gφ")
    figure.colorbar(axes_flat[3].images[0], ax=axes_flat[3], fraction=0.046)

    order_map = np.full(obstacle_map.shape, np.nan, dtype=np.float64)
    for index, coordinate in enumerate(result.selected_order, start=1):
        order_map[coordinate] = index
    axes_flat[4].imshow(np.ma.masked_invalid(order_map), cmap="plasma", interpolation="nearest")
    axes_flat[4].set_title("Selected / expanded node order")
    figure.colorbar(axes_flat[4].images[0], ax=axes_flat[4], fraction=0.046)

    axes_flat[5].imshow(free_map, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if result.path is not None:
        axes_flat[5].plot(
            [coordinate[1] for coordinate in result.path],
            [coordinate[0] for coordinate in result.path],
            color="red",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Selected path",
        )
    axes_flat[5].scatter([start[1]], [start[0]], marker="o", color="lime", label="Start")
    axes_flat[5].scatter([goal[1]], [goal[0]], marker="*", color="red", label="Goal")
    axes_flat[5].set_title("Final path" if result.found else "No path found")
    axes_flat[5].legend(loc="upper right")

    for axis in axes_flat:
        axis.set_xticks([])
        axis.set_yticks([])
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure, axes
