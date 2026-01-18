"""
Attention Modules for ProtoNetV2.

Implements CBAM (Convolutional Block Attention Module) for 
channel and spatial attention in spectrogram feature extraction.

Reference: https://arxiv.org/abs/1807.06521
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module.
    
    Focuses on "what" features are important by computing
    per-channel attention weights using global pooling.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck MLP
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Channel-attended tensor of shape [B, C, H, W]
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module.
    
    Focuses on "where" in the time-frequency representation
    by computing per-location attention weights.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Convolution kernel size for spatial attention
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Spatially-attended tensor of shape [B, C, H, W]
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Combines channel and spatial attention sequentially.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Attended tensor of shape [B, C, H, W]
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
