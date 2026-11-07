"""Configurable U-Net-style CNN for conventional heuristic prediction."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class HeuristicCNNConfig:
    """CNN architecture settings independent of a training run."""

    base_channels: int = 32
    depth: int = 4
    normalization: str = "group"
    activation: str = "silu"
    dropout: float = 0.0
    output_mode: str = "regression"

    def __post_init__(self) -> None:
        if self.base_channels < 4:
            raise ValueError("base_channels must be at least 4.")
        if not 2 <= self.depth <= 5:
            raise ValueError("depth must be between 2 and 5.")
        if self.normalization not in {"group", "batch", "none"}:
            raise ValueError("normalization must be 'group', 'batch', or 'none'.")
        if self.activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be 'relu', 'gelu', or 'silu'.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.output_mode not in {"regression", "ordinal"}:
            raise ValueError("output_mode must be 'regression' or 'ordinal'.")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("HeuristicCNN requires PyTorch.")


if torch is not None:

    def _normalization(channels: int, kind: str) -> nn.Module:
        if kind == "group":
            groups = gcd(channels, 8)
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
        if kind == "batch":
            return nn.BatchNorm2d(channels)
        return nn.Identity()


    def _activation(kind: str) -> nn.Module:
        if kind == "relu":
            return nn.ReLU(inplace=True)
        if kind == "gelu":
            return nn.GELU()
        return nn.SiLU(inplace=True)


    class _ConvBlock(nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            config: HeuristicCNNConfig,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for index in range(2):
                in_channels = input_channels if index == 0 else output_channels
                layers.extend(
                    (
                        nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1, bias=False),
                        _normalization(output_channels, config.normalization),
                        _activation(config.activation),
                    )
                )
                if config.dropout:
                    layers.append(nn.Dropout2d(config.dropout))
            self.layers = nn.Sequential(*layers)

        def forward(self, inputs: Tensor) -> Tensor:
            return self.layers(inputs)


    class HeuristicCNN(nn.Module):
        """Predict a nonnegative scalar map or logits over integer cost classes.

        Input channels are blocked-cell occupancy, start one-hot, and goal
        one-hot. Regression mode applies Softplus; ordinal mode returns logits
        and leaves softmax/decoding to the CEA objective and inference helper.
        """

        input_channels = 3

        def __init__(
            self,
            config: HeuristicCNNConfig = HeuristicCNNConfig(),
            *,
            num_classes: int | None = None,
        ) -> None:
            super().__init__()
            if config.output_mode == "ordinal":
                if num_classes is None or num_classes < 2:
                    raise ValueError("ordinal output requires num_classes >= 2.")
                output_channels = num_classes
            else:
                if num_classes not in (None, 1):
                    raise ValueError("regression output has exactly one channel.")
                output_channels = 1
            self.config = config
            self.num_classes = output_channels
            widths = [config.base_channels * min(2**level, 8) for level in range(config.depth)]
            self.encoders = nn.ModuleList()
            current_channels = self.input_channels
            for width in widths:
                self.encoders.append(_ConvBlock(current_channels, width, config))
                current_channels = width
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.decoders = nn.ModuleList()
            for level in range(config.depth - 2, -1, -1):
                skip_width = widths[level]
                self.decoders.append(
                    _ConvBlock(current_channels + skip_width, skip_width, config)
                )
                current_channels = skip_width
            self.head = nn.Conv2d(widths[0], output_channels, kernel_size=1)

        def forward(self, inputs: Tensor) -> Tensor:
            if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
                raise ValueError("CNN input must have shape (batch, 3, height, width).")
            skips: list[Tensor] = []
            features = inputs
            for index, encoder in enumerate(self.encoders):
                features = encoder(features)
                skips.append(features)
                if index < len(self.encoders) - 1:
                    features = self.pool(features)
            for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
                features = torch.nn.functional.interpolate(
                    features,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features = decoder(torch.cat((features, skip), dim=1))
            output = self.head(features)
            if self.config.output_mode == "regression":
                return torch.nn.functional.softplus(output)
            return output

else:

    class HeuristicCNN:  # type: ignore[no-redef]
        """Placeholder that gives a clear message when PyTorch is unavailable."""

        input_channels = 3

        def __init__(
            self,
            config: HeuristicCNNConfig = HeuristicCNNConfig(),
            *,
            num_classes: int | None = None,
        ) -> None:
            del config, num_classes
            _require_torch()


def decode_heuristic_output(raw_output: Tensor, method: str) -> Tensor:
    """Return raw scalar predictions for H1/H2 or argmax cost classes for H3."""

    _require_torch()
    if method in {"H1", "H2"}:
        if raw_output.ndim != 4 or raw_output.shape[1] != 1:
            raise ValueError("H1/H2 outputs must have one spatial prediction channel.")
        return raw_output[:, 0]
    if method == "H3":
        if raw_output.ndim != 4 or raw_output.shape[1] < 2:
            raise ValueError("H3 ordinal output must have at least two cost-class channels.")
        return torch.argmax(raw_output, dim=1).to(dtype=raw_output.dtype)
    raise ValueError("method must be H1, H2, or H3.")
