"""H1 regression and proposed H2 constraint-aware objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from heuristics.cea import CEALossConfig, cross_entropy_admissibility_loss
from models.heuristic_cnn import decode_heuristic_output


@dataclass(frozen=True, slots=True)
class ConstraintAwareLossConfig:
    """Initial H2 coefficients from Prompt 4; all are configurable."""

    alpha_over: float = 10.0
    beta_under: float = 1.0
    lambda_fit: float = 1.0
    lambda_adm: float = 10.0
    lambda_cons: float = 10.0
    lambda_goal: float = 10.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def compute_heuristic_losses(
    method: str,
    raw_output: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    edge_indices: Tensor,
    edge_costs: Tensor,
    goal_coordinates: Tensor,
    *,
    composite: ConstraintAwareLossConfig = ConstraintAwareLossConfig(),
    cea: CEALossConfig = CEALossConfig(),
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Return total objective, named loss terms, and uncorrected ``h_pred``.

    ``edge_indices`` has rows ``(batch_index, flattened_u, flattened_v)`` and
    includes all valid directed legal transitions. The edge costs come from
    Prompt 1's ground-truth transition-cost function, never Neural A* Phi.
    """

    if torch is None:
        raise ImportError("Heuristic learning losses require PyTorch.")
    prediction = decode_heuristic_output(raw_output, method)
    if prediction.shape != target.shape or target.shape != valid_mask.shape:
        raise ValueError("Predictions, targets, and validity masks must match (B,H,W).")
    if not bool(valid_mask.any().item()):
        raise ValueError("At least one reachable traversable target is required.")
    zeros = {
        "loss_fit": _zero(prediction),
        "loss_adm": _zero(prediction),
        "loss_cons": _zero(prediction),
        "loss_goal": _zero(prediction),
        "loss_cea": _zero(raw_output),
    }

    if method == "H1":
        error_squared = (prediction - target).square()
        loss_fit = error_squared[valid_mask].mean()
        terms = {**zeros, "loss_fit": loss_fit}
        return loss_fit, terms, prediction

    if method == "H2":
        error = prediction - target
        fit_per_cell = (
            composite.alpha_over * torch.relu(error).square()
            + composite.beta_under * torch.relu(-error).square()
        )
        loss_fit = fit_per_cell[valid_mask].mean()
        loss_adm = torch.relu(error).square()[valid_mask].mean()

        if edge_indices.numel():
            batch_index, source_index, destination_index = edge_indices.unbind(dim=1)
            flattened = prediction.reshape(prediction.shape[0], -1)
            source_values = flattened[batch_index, source_index]
            destination_values = flattened[batch_index, destination_index]
            inconsistency = source_values - edge_costs - destination_values
            loss_cons = torch.relu(inconsistency).square().mean()
        else:
            loss_cons = _zero(prediction)

        batch_indices = torch.arange(
            prediction.shape[0], device=prediction.device, dtype=torch.long
        )
        goal_rows = goal_coordinates[:, 0].long()
        goal_cols = goal_coordinates[:, 1].long()
        goal_values = prediction[batch_indices, goal_rows, goal_cols]
        loss_goal = goal_values.square().mean()

        total = (
            composite.lambda_fit * loss_fit
            + composite.lambda_adm * loss_adm
            + composite.lambda_cons * loss_cons
            + composite.lambda_goal * loss_goal
        )
        terms = {
            **zeros,
            "loss_fit": loss_fit,
            "loss_adm": loss_adm,
            "loss_cons": loss_cons,
            "loss_goal": loss_goal,
        }
        return total, terms, prediction

    if method == "H3":
        loss_cea = cross_entropy_admissibility_loss(raw_output, target.long(), valid_mask, cea)
        terms = {**zeros, "loss_cea": loss_cea}
        return loss_cea, terms, prediction

    raise ValueError("method must be H1, H2, or H3.")
