"""PyTorch differentiable search module from Yonetani et al. (2021)."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from algorithms.movement import Coord, DEFAULT_COST_CONFIG, GroundTruthCostConfig, get_neighbors, validate_map

try:  # PyTorch is an optional runtime dependency for search-only installations.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


if torch is not None:

    class _IdentityBackwardArgmax(torch.autograd.Function):
        """Hard one-hot argmax forward with the paper's identity backward."""

        @staticmethod
        def forward(ctx: Any, probabilities: Tensor) -> Tensor:
            selected = torch.zeros_like(probabilities)
            flat_index = int(torch.argmax(probabilities.reshape(-1)).item())
            selected.reshape(-1)[flat_index] = 1.0
            return selected

        @staticmethod
        def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor]:
            return (gradient,)


@dataclass(frozen=True, slots=True)
class DifferentiableSearchResult:
    """Differentiable closed-list output plus its hard-forward search trace."""

    closed_mask: Tensor
    accumulated_guidance_cost: Tensor
    selected_order: tuple[Coord, ...]
    path: tuple[Coord, ...] | None
    found: bool


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Neural A* training requires PyTorch. Install a PyTorch build in the project environment."
        )


def identity_backward_argmax(probabilities: Tensor) -> Tensor:
    """Apply the paper's hard one-hot argmax with identity gradient."""

    _require_torch()
    return _IdentityBackwardArgmax.apply(probabilities)


def differentiable_neural_search(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: Tensor,
    *,
    temperature: float | None = None,
    tie_break_epsilon: float = 0.001,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> DifferentiableSearchResult:
    """Run one differentiable Neural A* forward pass for a single map.

    The hard OPEN/CLOSED/neighborhood decisions follow Algorithm 2. The
    selector uses Eq. (3) and the identity-backward hard argmax; the guidance
    updates use Eqs. (4)–(6). As in the paper, state masks are discrete, and the
    prior ``G`` entering Eq. (6) is detached to limit recurrent gradients. The
    output closed mask follows Eq. (8), so the selected goal is not added to
    CLOSED (its goal-verification multiplier is zero).
    """

    _require_torch()
    obstacle_map = validate_map(grid)
    if phi.ndim == 3 and phi.shape[0] == 1:
        phi = phi[0]
    if phi.ndim != 2 or tuple(phi.shape) != obstacle_map.shape:
        raise ValueError("phi must have shape (height, width) matching the map.")
    if not phi.is_floating_point():
        raise ValueError("phi must be a floating-point tensor.")
    if not bool(torch.isfinite(phi.detach()).all().item()):
        raise ValueError("phi must contain only finite values.")
    if bool(((phi.detach() < 0.0) | (phi.detach() > 1.0)).any().item()):
        raise ValueError("Neural A* guidance values must lie in [0, 1].")
    if not isinstance(cost_config, GroundTruthCostConfig):
        raise TypeError("cost_config must be a GroundTruthCostConfig instance.")
    height, width = obstacle_map.shape
    for label, coord in (("start", start), ("goal", goal)):
        row, col = coord
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} coordinate {coord!r} is outside the map.")
        if obstacle_map[row, col] != 0:
            raise ValueError(f"{label} cell must be traversable.")
    start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    tau = sqrt(width) if temperature is None else float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not np.isfinite(tie_break_epsilon) or tie_break_epsilon < 0:
        raise ValueError("tie_break_epsilon must be finite and non-negative.")

    device, dtype = phi.device, phi.dtype
    rows = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    cols = torch.arange(width, device=device, dtype=dtype).view(1, width)
    row_distance = torch.abs(rows - goal[0]).expand(height, width)
    col_distance = torch.abs(cols - goal[1]).expand(height, width)
    heuristic = torch.maximum(row_distance, col_distance)
    heuristic = heuristic + tie_break_epsilon * torch.sqrt(
        row_distance.square() + col_distance.square()
    )

    open_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    closed_eligibility = torch.zeros_like(open_mask)
    closed_output = torch.zeros((height, width), dtype=dtype, device=device)
    accumulated = torch.zeros((height, width), dtype=dtype, device=device)
    open_mask[start] = True
    parents: dict[Coord, Coord | None] = {start: None}
    selected_order: list[Coord] = []
    found = False

    while bool(open_mask.any().item()):
        priority = accumulated + heuristic
        masked_logits = torch.where(
            open_mask,
            -priority / tau,
            torch.full_like(priority, -torch.inf),
        )
        probabilities = torch.softmax(masked_logits.reshape(-1), dim=0).reshape(height, width)
        selected_map = identity_backward_argmax(probabilities)
        flat_index = int(torch.argmax(selected_map.detach().reshape(-1)).item())
        selected = tuple(int(value) for value in np.unravel_index(flat_index, (height, width)))
        selected_order.append(selected)

        start_mask = torch.zeros_like(open_mask)
        start_mask[goal] = True
        eta = 1.0 - (selected_map * start_mask.to(dtype)).sum()
        closed_output = closed_output + eta * selected_map
        if selected == goal:
            found = True
            break

        selected_mask = torch.zeros_like(open_mask)
        selected_mask[selected] = True
        open_mask = open_mask & ~selected_mask
        closed_eligibility = closed_eligibility | selected_mask
        neighbors = get_neighbors(obstacle_map, selected, cost_config)
        all_neighbor_mask = torch.zeros_like(open_mask)
        for neighbor in neighbors:
            all_neighbor_mask[neighbor] = True
        new_neighbors = all_neighbor_mask & ~open_mask & ~closed_eligibility
        existing_open_neighbors = all_neighbor_mask & open_mask & ~closed_eligibility

        # Eq. (6): only the previous accumulated state is detached. Phi remains
        # differentiable; Eq. (5)'s min retains gradients through existing OPEN
        # costs, matching the paper's stated detach points.
        previous_costs = accumulated.detach()
        base_cost = (previous_costs * selected_map).sum()
        proposed_costs = base_cost + phi
        accumulated = torch.where(
            new_neighbors,
            proposed_costs,
            torch.where(
                existing_open_neighbors,
                torch.minimum(accumulated, proposed_costs),
                accumulated,
            ),
        )

        selected_cost_value = float(accumulated.detach()[selected].item())
        for neighbor in neighbors:
            if closed_eligibility[neighbor]:
                continue
            candidate = selected_cost_value + float(phi.detach()[neighbor].item())
            if not open_mask[neighbor] or candidate < float(previous_costs[neighbor].item()):
                parents[neighbor] = selected
        open_mask = open_mask | new_neighbors

    path: tuple[Coord, ...] | None = None
    if found:
        reverse_path = [goal]
        current = goal
        while current != start:
            parent = parents.get(current)
            if parent is None:
                raise RuntimeError("Neural A* parent map could not reconstruct the selected goal.")
            reverse_path.append(parent)
            current = parent
            if len(reverse_path) > int(np.count_nonzero(obstacle_map == 0)):
                raise RuntimeError("Neural A* parent map contains a cycle.")
        path = tuple(reversed(reverse_path))

    return DifferentiableSearchResult(
        closed_mask=closed_output,
        accumulated_guidance_cost=accumulated,
        selected_order=tuple(selected_order),
        path=path,
        found=found,
    )
