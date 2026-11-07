"""Published Cross-Entropy Admissibility objective from Futuhi et al. (2026)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True, slots=True)
class CEALossConfig:
    """CEA parameters; ``beta`` and ``eta`` are project-tunable settings."""

    beta: float = 1.0
    eta: float = 0.01
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("CEA beta must be finite and positive.")
        if not isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("CEA eta must be finite and positive.")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("CEA epsilon must be between zero and one.")


def cross_entropy_admissibility_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    config: CEALossConfig = CEALossConfig(),
) -> Tensor:
    """Compute Futuhi et al.'s Eq. (5) over valid spatial states.

    ``logits`` is ``(batch, classes, height, width)`` and the class indices
    encode the integer cost labels ``0, ..., classes-1``. For positive target
    ``h*``, the first term is ``-log(sum_(k=1)^h* (k/h*)^beta p_k)``; the second
    term is ``-eta*log(p_h*)``. At ``h*=0`` the printed equation's ratio has a
    zero denominator, so this implementation uses the exact-class boundary
    term ``-log(p_0)`` plus its target cross-entropy term.
    """

    if torch is None:
        raise ImportError("CEA loss requires PyTorch.")
    if logits.ndim != 4 or target.ndim != 3 or valid_mask.ndim != 3:
        raise ValueError("Expected logits (B,C,H,W) and target/mask (B,H,W).")
    if tuple(logits.shape[0:1] + logits.shape[2:]) != tuple(target.shape):
        raise ValueError("Logit spatial dimensions must match targets.")
    if tuple(target.shape) != tuple(valid_mask.shape):
        raise ValueError("Target and validity mask dimensions must match.")
    if not bool(valid_mask.any().item()):
        raise ValueError("CEA requires at least one valid target cell.")
    if bool((target[valid_mask] < 0).any().item()):
        raise ValueError("CEA class targets must be nonnegative.")
    if bool((target[valid_mask] >= logits.shape[1]).any().item()):
        raise ValueError("A target cost class exceeds the model output class range.")
    if bool((target[valid_mask] != target[valid_mask].round()).any().item()):
        raise ValueError("CEA requires integer-valued cost-to-go labels.")

    probabilities = torch.softmax(logits, dim=1)
    class_values = torch.arange(
        logits.shape[1], device=logits.device, dtype=logits.dtype
    ).view(1, -1, 1, 1)
    target_float = target.to(dtype=logits.dtype)
    denominator = target_float.clamp_min(1.0).unsqueeze(1)
    weights = (class_values / denominator).clamp_min(0.0).pow(config.beta)
    admissible_classes = (class_values >= 1.0) & (class_values <= target_float.unsqueeze(1))
    weights = weights * admissible_classes.to(dtype=logits.dtype)
    admissible_mass = (probabilities * weights).sum(dim=1)
    zero_target = target == 0
    admissible_mass = torch.where(zero_target, probabilities[:, 0], admissible_mass)

    target_probability = probabilities.gather(1, target.long().unsqueeze(1)).squeeze(1)
    sample_loss = -torch.log(admissible_mass.clamp_min(config.epsilon))
    sample_loss = sample_loss - config.eta * torch.log(
        target_probability.clamp_min(config.epsilon)
    )
    return sample_loss[valid_mask].mean()
