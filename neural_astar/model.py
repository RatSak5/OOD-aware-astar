"""U-Net/VGG16-style encoder that predicts Neural A* guidance maps."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:

    class _VGGBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, convolutions: int) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for index in range(convolutions):
                layers.extend(
                    (
                        nn.Conv2d(
                            input_channels if index == 0 else output_channels,
                            output_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.GroupNorm(num_groups=8, num_channels=output_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            self.layers = nn.Sequential(*layers)

        def forward(self, features: Tensor) -> Tensor:
            return self.layers(features)


    class _DecoderBlock(nn.Module):
        def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
            super().__init__()
            self.block = _VGGBlock(input_channels + skip_channels, output_channels, 2)

        def forward(self, features: Tensor, skip: Tensor) -> Tensor:
            features = torch.nn.functional.interpolate(
                features,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return self.block(torch.cat((features, skip), dim=1))


    class NeuralAStarUNet(nn.Module):
        """Fully convolutional two-channel U-Net with VGG16 encoder blocks.

        The five encoder blocks use the VGG16 convolution counts and channels
        (64, 128, 256, 512, 512). GroupNorm with eight groups follows every
        convolution in the encoder and decoder blocks. The project-local
        decoder mirrors the encoder levels with bilinear resizing and a final
        sigmoid produces Phi in [0, 1]. Ceil-mode pooling preserves very small
        and odd-sized maps. Convolution and normalization parameters use
        PyTorch defaults; pretrained VGG16 weights are not assumed or downloaded.
        """

        def __init__(self, input_channels: int = 2) -> None:
            super().__init__()
            if input_channels != 2:
                raise ValueError("The paper's explicit-map encoder has two input channels.")
            self.encoder1 = _VGGBlock(input_channels, 64, 2)
            self.encoder2 = _VGGBlock(64, 128, 2)
            self.encoder3 = _VGGBlock(128, 256, 3)
            self.encoder4 = _VGGBlock(256, 512, 3)
            self.encoder5 = _VGGBlock(512, 512, 3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.decoder4 = _DecoderBlock(512, 512, 512)
            self.decoder3 = _DecoderBlock(512, 256, 256)
            self.decoder2 = _DecoderBlock(256, 128, 128)
            self.decoder1 = _DecoderBlock(128, 64, 64)
            self.guidance_head = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, features: Tensor) -> Tensor:
            if features.ndim != 4 or features.shape[1] != 2:
                raise ValueError("Input must have shape (batch, 2, height, width).")
            first = self.encoder1(features)
            second = self.encoder2(self.pool(first))
            third = self.encoder3(self.pool(second))
            fourth = self.encoder4(self.pool(third))
            fifth = self.encoder5(self.pool(fourth))
            decoded = self.decoder4(fifth, fourth)
            decoded = self.decoder3(decoded, third)
            decoded = self.decoder2(decoded, second)
            decoded = self.decoder1(decoded, first)
            return torch.sigmoid(self.guidance_head(decoded))

else:

    class NeuralAStarUNet:  # type: ignore[no-redef]
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, input_channels: int = 2) -> None:
            del input_channels
            raise ImportError(
                "NeuralAStarUNet requires PyTorch. Install PyTorch to construct the CNN."
            )


def make_guidance_tensor(model_output: Tensor) -> Tensor:
    """Validate and return a model's ``(batch, 1, H, W)`` guidance output."""

    if torch is None:
        raise ImportError("Guidance tensors require PyTorch.")
    if model_output.ndim != 4 or model_output.shape[1] != 1:
        raise ValueError("Model output must have shape (batch, 1, height, width).")
    if not bool(torch.isfinite(model_output).all().item()):
        raise ValueError("Model guidance output must be finite.")
    if bool(((model_output < 0.0) | (model_output > 1.0)).any().item()):
        raise ValueError("Sigmoid guidance values must lie in [0, 1].")
    return model_output
