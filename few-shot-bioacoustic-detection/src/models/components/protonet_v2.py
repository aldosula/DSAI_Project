"""
ProtoNetV2 - Enhanced Prototypical Network with CBAM Attention.

This is a NEW architecture that does NOT modify the baseline.
It adds CBAM attention after each ResNet block for improved
feature selection in few-shot bioacoustic event detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
from pathlib import Path
import re

from src.models.components.attention_modules import CBAM


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, with_bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=with_bias,
    )


class BasicBlockV2(nn.Module):
    """ResNet BasicBlock with CBAM attention.
    
    Enhanced version that applies channel and spatial attention
    after the residual connection.
    """
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        drop_rate: float = 0.0,
        drop_block: bool = False,
        block_size: int = 1,
        with_bias: bool = False,
        non_linearity: str = "leaky_relu",
        use_attention: bool = True,
        attention_reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, with_bias=with_bias)
        self.bn1 = nn.BatchNorm2d(planes)

        if non_linearity == "leaky_relu":
            self.relu = nn.LeakyReLU(0.1)
        else:
            self.relu = nn.ReLU()

        self.conv2 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        
        # CBAM Attention
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(planes, reduction_ratio=attention_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_batches_tracked += 1

        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        # Apply CBAM attention after residual connection
        if self.use_attention:
            out = self.cbam(out)
        
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNetV2(nn.Module):
    """ResNet V2 with CBAM Attention.
    
    Enhanced backbone for few-shot bioacoustic event detection.
    Uses CBAM attention after each residual block to focus on
    relevant time-frequency patterns.
    """
    
    def __init__(
        self,
        features,
        block=BasicBlockV2,
        keep_prob: float = 1.0,
        avg_pool: bool = True,
        drop_rate: float = 0.1,
        linear_drop_rate: float = 0.5,
        dropblock_size: int = 5,
        use_attention: bool = True,
        attention_reduction: int = 16,
    ):
        super().__init__()
        
        drop_rate = features.drop_rate
        with_bias = features.with_bias
        
        # V2 specific: attention settings
        self.use_attention = use_attention
        self.attention_reduction = attention_reduction

        self.inplanes = 1
        self.linear_drop_rate = linear_drop_rate
        
        # Build layers with attention
        self.layer1 = self._make_layer(block, 64, stride=2, features=features)
        self.layer2 = self._make_layer(block, 128, stride=2, features=features)
        self.layer3 = self._make_layer(
            block,
            64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )
        self.layer4 = self._make_layer(
            block,
            64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )
        
        self.keep_prob = keep_prob
        self.features = features
        self.keep_avg_pool = avg_pool
        self.drop_rate = drop_rate
        
        self.pool_avg = nn.AdaptiveAvgPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )
        self.pool_max = nn.AdaptiveMaxPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=features.non_linearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, 
        block, 
        planes: int, 
        stride: int = 1, 
        drop_block: bool = False, 
        block_size: int = 1, 
        features=None
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                features.drop_rate,
                drop_block,
                block_size,
                features.with_bias,
                features.non_linearity,
                use_attention=self.use_attention,
                attention_reduction=self.attention_reduction,
            )
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (num_samples, seq_len, mel_bins) = x.shape

        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.features.layer_4:
            x = self.layer4(x)
        x = self.pool_avg(x)

        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    def get_n_params(model):
        return sum(p.numel() for p in model.parameters())
    
    # Test with dummy config
    class DummyFeatures:
        drop_rate = 0.1
        with_bias = False
        non_linearity = "leaky_relu"
        time_max_pool_dim = 4
        embedding_dim = 2048
        layer_4 = False
    
    features = DummyFeatures()
    model = ResNetV2(features)
    print(f"ResNetV2 Parameters: {get_n_params(model):,}")
    
    input_tensor = torch.randn((3, 17, 128))
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
