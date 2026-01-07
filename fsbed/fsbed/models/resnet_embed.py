"""ResNet-like embedding network for frame-level features."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNetEmbedding(nn.Module):
    """
    ResNet-like embedding network for spectrogram input.

    Produces frame-level embeddings while preserving time resolution.

    Input: [batch, 1, n_mels, time]
    Output: [batch, time, embed_dim]
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 256,
        channels: list[int] = [64, 128, 256, 256],
        dropout: float = 0.1,
    ):
        """
        Initialize ResNet embedding network.

        Args:
            n_mels: Number of mel frequency bins.
            embed_dim: Output embedding dimension.
            channels: Channel sizes for each block.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim

        # Initial convolution
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

        # Residual blocks
        # Pool along frequency, preserve time
        self.layer1 = self._make_layer(channels[0], channels[0], 2, stride=(2, 1), dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=(2, 1), dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=(2, 1), dropout=dropout)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=(2, 1), dropout=dropout)

        # Global average pool along frequency
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        # Final projection to embedding dim
        self.fc = nn.Linear(channels[3], embed_dim)

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.1,
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != (1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample, dropout)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, None, dropout))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, config: ModelConfig, n_mels: int = 128) -> "ResNetEmbedding":
        """Create from ModelConfig."""
        return cls(
            n_mels=n_mels,
            embed_dim=config.embed_dim,
            channels=config.channels,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 1, n_mels, time].

        Returns:
            Embeddings [batch, time, embed_dim].
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool along frequency
        x = self.avgpool(x)  # [batch, channels, 1, time]
        x = x.squeeze(2)  # [batch, channels, time]
        x = x.permute(0, 2, 1)  # [batch, time, channels]

        # Project to embedding dim
        x = self.fc(x)  # [batch, time, embed_dim]

        return x

    def get_output_time_ratio(self) -> float:
        """
        Get the ratio of output time frames to input time frames.

        Due to pooling, output has fewer time frames than input.
        """
        # Initial conv: stride 2, maxpool doesn't stride time
        # Layers use stride (2, 1) so time is mostly preserved
        return 0.5  # Approximate


class FullModel(nn.Module):
    """
    Complete model with embedding backbone and classification head.

    Convenience wrapper for training.
    """

    def __init__(
        self,
        backbone: ResNetEmbedding,
        head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram [batch, 1, n_mels, time].

        Returns:
            Logits [batch, time, num_classes].
        """
        embeddings = self.backbone(x)
        logits = self.head(embeddings)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings without classification."""
        return self.backbone(x)
