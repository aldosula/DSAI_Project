"""
ProtoNetV3 - BEATs Pretrained Backbone for Few-Shot Bioacoustic Detection.

Uses Microsoft's BEATs (Audio Pre-Training with Acoustic Tokenizers) 
pretrained on AudioSet for cross-species generalization.

Reference: https://arxiv.org/abs/2212.09058
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os

from src.models.components.BEATs import BEATs, BEATsConfig


class ResNetV3(nn.Module):
    """BEATs-based backbone for few-shot bioacoustic event detection.
    
    Uses pretrained BEATs model from Microsoft for feature extraction.
    BEATs is trained on AudioSet with 527 sound categories.
    """
    
    def __init__(
        self,
        features,
        weights_path: str = None,
        freeze_backbone: bool = False,
        output_dim: int = 2048,
    ):
        """
        Args:
            features: Feature config from train.yaml
            weights_path: Path to pretrained BEATs checkpoint
            freeze_backbone: Whether to freeze BEATs weights
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        self.weights_path = weights_path
        self.freeze_backbone = freeze_backbone
        
        # Lazy loading flags
        self._beats_loaded = False
        self.adapter_mode = False
        self.beats = None
        self.beats_cfg = None
        
        # Check if weights exist - if not, use adapter mode
        if weights_path is None or not os.path.exists(weights_path):
            print("WARNING: BEATs weights not found. Using adapter mode.")
            print(f"Expected path: {weights_path}")
            self._init_adapter_mode()
            self._beats_loaded = True
        else:
            # BEATs will be loaded lazily on first forward pass
            self.adapter_mode = False
            print(f"BEATs weights will be loaded from: {weights_path}")
        
        # Project to output dimension
        self.output_proj = nn.Linear(768, output_dim)  # BEATs outputs 768-dim
        
    def _init_adapter_mode(self):
        """Initialize without pretrained weights - use simple CNN."""
        print("Using adapter mode (no pretrained BEATs)")
        self.adapter_mode = True
        
        # Simple CNN to match BEATs output style
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 768, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def _load_beats(self, device):
        """Lazy load BEATs weights on first forward pass."""
        if self._beats_loaded:
            return
            
        print(f"Loading BEATs from: {self.weights_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.weights_path, map_location=device)
        
        # Extract config from checkpoint
        cfg = BEATsConfig(checkpoint.get('cfg', None))
        self.beats_cfg = cfg
        
        # Create BEATs model
        self.beats = BEATs(cfg)
        
        # Load weights
        self.beats.load_state_dict(checkpoint['model'])
        self.beats = self.beats.to(device)
        print("BEATs weights loaded successfully!")
        
        if self.freeze_backbone:
            print("Freezing BEATs backbone...")
            for param in self.beats.parameters():
                param.requires_grad = False
        
        self._beats_loaded = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram [batch, time, freq] (e.g., [B, 17, 128])
        
        Returns:
            Embeddings [batch, output_dim]
        """
        batch_size, time_frames, freq_bins = x.shape
        
        if self.adapter_mode:
            # Use simple CNN adapter
            x = x.unsqueeze(1)  # [B, 1, T, F]
            x = self.adapter(x)
            x = x.view(batch_size, -1)  # [B, 768]
        else:
            # Lazy load BEATs on first forward pass
            self._load_beats(x.device)
            # Use BEATs
            # BEATs expects fbank features [B, T, 128]
            # Normalize like BEATs preprocessing
            fbank_mean = 15.41663
            fbank_std = 6.55582
            x = (x - fbank_mean) / (2 * fbank_std)
            
            # Pad time dimension if too short (BEATs patch_embedding uses 16x16 kernel)
            if time_frames < 16:
                pad_size = 16 - time_frames
                x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
            
            # Add channel dim and run through BEATs
            x = x.unsqueeze(1)  # [B, 1, T, F]
            features = self.beats.patch_embedding(x)
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.transpose(1, 2)
            features = self.beats.layer_norm(features)
            
            if self.beats.post_extract_proj is not None:
                features = self.beats.post_extract_proj(features)
            
            features = self.beats.dropout_input(features)
            
            # Run through transformer encoder
            x, _ = self.beats.encoder(features, padding_mask=None)
            
            # Mean pooling over time
            x = x.mean(dim=1)  # [B, 768]
        
        # Project to output dimension
        x = self.output_proj(x)
        
        return x


if __name__ == "__main__":
    """Test the model."""
    class DummyFeatures:
        drop_rate = 0.1
        with_bias = False
        non_linearity = "leaky_relu"
        time_max_pool_dim = 4
        embedding_dim = 2048
        layer_4 = False
    
    features = DummyFeatures()
    
    # Test adapter mode (no weights)
    model = ResNetV3(features, weights_path=None)
    print(f"ResNetV3 (adapter mode) Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    input_tensor = torch.randn((3, 17, 128))
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("SUCCESS!")
