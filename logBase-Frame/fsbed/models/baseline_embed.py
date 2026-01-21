"""Baseline embedding network

Architecture: 2 BasicBlocks + 2 CNN layers.
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """
    BasicBlock from ResNet with skip connection.
    
    Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> (+skip) -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        return out


class CNNBlock(nn.Module):
    """Simple CNN block: Conv 3x3 -> BN -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class BaselineEmbedding(nn.Module):
    """
    Baseline embedding network.
    
    Architecture:
    - BasicBlock1: 1 -> 64 channels
    - BasicBlock2: 64 -> 64 channels  
    - CNNBlock3: 64 -> 64 channels
    - CNNBlock4: 64 -> 64 channels
    - Global Average Pooling
    - FC: -> embed_dim (1024 in paper)
    
    Input: [batch, 1, n_mels, time]
    Output: [batch, time, embed_dim]
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 1024,
        channels: int = 64,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        # 2 BasicBlocks
        self.block1 = BasicBlock(1, channels, stride=2)
        self.block2 = BasicBlock(channels, channels, stride=2)
        
        # 2 CNN blocks
        self.block3 = CNNBlock(channels, channels, stride=2)
        self.block4 = CNNBlock(channels, channels, stride=1)  
        
        # Global average pooling over frequency
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # FC layer to embed_dim
        self.fc = nn.Linear(channels, embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    @classmethod
    def from_config(cls, config: ModelConfig, n_mels: int = 128) -> "BaselineEmbedding":
        """Create from config."""
        return cls(
            n_mels=n_mels,
            embed_dim=config.embed_dim,
            channels=64,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, 1, n_mels, time]
            
        Returns:
            Frame-level embeddings [batch, time, embed_dim]
        """
        # Conv blocks - downsample frequency, preserve time
        x = self.block1(x)  # [B, 64, freq/2, time]
        x = self.block2(x)  # [B, 64, freq/4, time]
        x = self.block3(x)  # [B, 64, freq/8, time]
        x = self.block4(x)  # [B, 64, freq/8, time]
        
        # Global pool over frequency
        x = self.global_pool(x)  # [B, 64, 1, time]
        x = x.squeeze(2)  # [B, 64, time]
        x = x.permute(0, 2, 1)  # [B, time, 64]
        
        # Project to embed_dim
        x = self.fc(x)  # [B, time, embed_dim]
        
        return x
    
    def get_layer_groups(self) -> dict:
        """Get layer groups for differential learning rates during fine-tuning."""
        return {
            "early": [self.block1, self.block2],
            "late": [self.block3, self.block4],
            "fc": [self.fc],
        }


class BaselineFullModel(nn.Module):
    """
    Full model with backbone and classification head.
    
    Used for both pretraining (num_classes=20) and inference (num_classes=2).
    """
    
    def __init__(
        self,
        backbone: BaselineEmbedding,
        num_classes: int = 20,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, 1, n_mels, time]
            
        Returns:
            Logits [batch, num_classes] (pooled over time)
        """
        # Get frame embeddings
        embeddings = self.backbone(x)  # [B, time, embed_dim]
        
        # Pool over time for classification
        embeddings = embeddings.mean(dim=1)  # [B, embed_dim]
        
        # Classify
        logits = self.classifier(embeddings)  # [B, num_classes]
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings without classification."""
        return self.backbone(x)
    
    def replace_classifier(self, num_classes: int) -> None:
        """Replace classifier for different number of classes."""
        embed_dim = self.backbone.embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        logger.info(f"Replaced classifier with {num_classes} classes")
