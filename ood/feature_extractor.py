"""Independent CNN feature extractor for obstacle/start/goal maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np

from algorithms.movement import Coord, validate_map

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on local packages
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class OODFeatureConfig:
    """Architecture parameters for the independent map embedding network."""

    base_channels: int = 32
    depth: int = 4
    embedding_dim: int = 128
    projection_dim: int = 64
    normalization: str = "group"

    def __post_init__(self) -> None:
        if self.base_channels < 4:
            raise ValueError("base_channels must be at least 4.")
        if not 2 <= self.depth <= 5:
            raise ValueError("depth must be between 2 and 5.")
        if self.embedding_dim < 2 or self.projection_dim < 2:
            raise ValueError("embedding_dim and projection_dim must be at least 2.")
        if self.normalization not in {"group", "batch", "none"}:
            raise ValueError("normalization must be 'group', 'batch', or 'none'.")


def build_ood_input(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
) -> np.ndarray:
    """Create obstacle, start, goal channels without heuristic/guidance data."""

    obstacle_map = validate_map(grid)
    height, width = obstacle_map.shape
    features = np.zeros((3, height, width), dtype=np.float32)
    features[0] = obstacle_map.astype(np.float32)
    for channel, label, coordinate in (
        (1, "start", start),
        (2, "goal", goal),
    ):
        if len(coordinate) != 2:
            raise ValueError(f"{label} must be a (row, column) coordinate.")
        row, col = coordinate
        if (
            isinstance(row, (bool, np.bool_))
            or isinstance(col, (bool, np.bool_))
            or not isinstance(row, (int, np.integer))
            or not isinstance(col, (int, np.integer))
        ):
            raise ValueError(f"{label} coordinates must be integers.")
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} coordinate {coordinate!r} is outside the map.")
        if obstacle_map[row, col] != 0:
            raise ValueError(f"{label} must be on a traversable cell.")
        features[channel, row, col] = 1.0
    return features


if torch is not None:

    class _FeatureBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, norm: str) -> None:
            super().__init__()
            groups = gcd(output_channels, 8)
            if output_channels // groups < 2:
                groups = max(groups // 2, 1)
            if norm == "group":
                normalization: nn.Module = nn.GroupNorm(groups, output_channels)
            elif norm == "batch":
                normalization = nn.BatchNorm2d(output_channels)
            else:
                normalization = nn.Identity()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                normalization,
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels)
                if norm == "group"
                else nn.BatchNorm2d(output_channels)
                if norm == "batch"
                else nn.Identity(),
                nn.SiLU(inplace=True),
            )

        def forward(self, inputs: Tensor) -> Tensor:
            return self.layers(inputs)


    class OODFeatureExtractor(nn.Module):
        """CNN encoder with fixed-dimensional map embeddings and a projector.

        `forward()` returns the pre-projection embedding used by Mahalanobis
        statistics. `project()` is only used for self-supervised contrastive
        training; detector statistics never use the contrastive projection head.
        """

        input_channels = 3

        def __init__(self, config: OODFeatureConfig = OODFeatureConfig()) -> None:
            super().__init__()
            self.config = config
            widths = [config.base_channels * min(2**i, 8) for i in range(config.depth)]
            self.blocks = nn.ModuleList()
            current_channels = self.input_channels
            for width in widths:
                self.blocks.append(_FeatureBlock(current_channels, width, config.normalization))
                current_channels = width
            self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.embedding = nn.Sequential(
                nn.Linear(widths[-1], config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.SiLU(),
            )
            self.projector = nn.Sequential(
                nn.Linear(config.embedding_dim, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.SiLU(),
                nn.Linear(config.embedding_dim, config.projection_dim),
            )

        def forward(self, inputs: Tensor) -> Tensor:
            if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
                raise ValueError("Feature input must have shape (batch, 3, height, width).")
            features = inputs
            for index, block in enumerate(self.blocks):
                features = block(features)
                if index < len(self.blocks) - 1:
                    features = self.pool(features)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, output_size=1).flatten(1)
            return self.embedding(pooled)

        def project(self, embeddings: Tensor) -> Tensor:
            projection = self.projector(embeddings)
            return torch.nn.functional.normalize(projection, p=2, dim=1)

else:

    class OODFeatureExtractor:  # type: ignore[no-redef]
        """Placeholder for environments without PyTorch."""

        def __init__(self, config: OODFeatureConfig = OODFeatureConfig()) -> None:
            del config
            raise ImportError("OOD feature extraction requires PyTorch.")
