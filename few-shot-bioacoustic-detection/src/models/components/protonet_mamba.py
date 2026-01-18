"""ProtoNetMamba: Multi-task model based on Lu et al. DCASE2024 paper.

Architecture:
- 4-layer CNN encoder (Conv→BN→ReLU→MaxPool)
- Mamba encoder for sequence modeling (optional, requires mamba-ssm)
- Multi-task training: SED branch + FBC branch (Foreground/Background Classification)

Paper: "Few-Shot Bioacoustic Event Detection with NetMamba Encoder"
F-measure: 56.4% on DCASE2024 Task 5 Validation Set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def conv_block(in_channels, out_channels, stride=2):
    """Standard CNN block: Conv 3x3 → BN → ReLU → MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(stride),
    )


class MambaBlock(nn.Module):
    """Simplified Mamba-style block for sequence modeling.
    
    If mamba-ssm is not installed, falls back to a 1D CNN + gating mechanism
    that approximates the Mamba behavior.
    """
    
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        # Input projection
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal 1D conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=False
        )
        
        # Output projection  
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Gating
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Project and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Conv1d (causal)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :x.shape[2]]  # Causal: truncate
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Gating
        x = self.act(x) * z
        
        # Output projection
        x = self.out_proj(x)
        
        return x + residual


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for sequence modeling."""
    
    def __init__(self, d_model, n_layers=2, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ProtoNetMamba(nn.Module):
    """ProtoNet with Mamba encoder and multi-task training (Lu et al.).
    
    Architecture:
        Input (B, T, n_mels) → 4-layer CNN → (B, T', 64) 
                            → Mamba Encoder → (B, T', 64)
                            → Embedding head → (B, embed_dim)
    
    Multi-task branches:
        - SED branch: FC(embed_dim, num_classes) for sound event detection
        - FBC branch: FC(embed_dim, 2) for foreground/background classification
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 2048,
        num_classes: int = 40,  # SED classes
        use_mamba: bool = True,
        mamba_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_mamba = use_mamba
        
        # 4-layer CNN encoder (paper Table 2)
        self.encoder = nn.Sequential(
            conv_block(1, 64, stride=2),    # Layer 1: 1→64
            conv_block(64, 128, stride=2),  # Layer 2: 64→128  
            conv_block(128, 128, stride=2), # Layer 3: 128→128
            conv_block(128, 64, stride=2),  # Layer 4: 128→64
        )
        
        # Calculate output size after CNN
        # Input: (B, 1, T, 128) → after 4 layers with stride 2: (B, 64, T//16, 8)
        self.cnn_output_dim = 64 * 8  # 512
        
        # Mamba encoder for sequence modeling
        if use_mamba:
            self.mamba = MambaEncoder(
                d_model=self.cnn_output_dim,
                n_layers=mamba_layers,
            )
        
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(self.cnn_output_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Multi-task heads
        # SED branch: Sound Event Detection (num_classes classification)
        self.sed_head = nn.Linear(embed_dim, num_classes)
        
        # FBC branch: Foreground/Background Classification (binary)
        self.fbc_head = nn.Linear(embed_dim, 2)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_embeddings(self, x):
        """Extract embeddings without classification heads.
        
        Args:
            x: (batch, seq_len, n_mels) or (batch, 1, seq_len, n_mels)
            
        Returns:
            embeddings: (batch, embed_dim)
        """
        # Handle input shape
        if x.dim() == 3:
            batch, seq_len, n_mels = x.shape
            x = x.view(batch, 1, seq_len, n_mels)
        
        # CNN encoder
        x = self.encoder(x)  # (B, 64, T', 8)
        
        # Reshape for sequence modeling
        batch, channels, t_out, freq = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, T', 64, 8)
        x = x.reshape(batch, t_out, channels * freq)  # (B, T', 512)
        
        # Mamba encoder
        if self.use_mamba:
            x = self.mamba(x)  # (B, T', 512)
        
        # Global pooling over time
        x = x.mean(dim=1)  # (B, 512)
        
        # Project to embedding space
        embeddings = self.embed_proj(x)  # (B, embed_dim)
        
        return embeddings
    
    def forward(self, x, return_fbc=False):
        """Forward pass for training.
        
        Args:
            x: (batch, seq_len, n_mels)
            return_fbc: If True, also return FBC logits for multi-task training
            
        Returns:
            embeddings: (batch, embed_dim) if return_fbc=False
            (embeddings, sed_logits, fbc_logits) if return_fbc=True
        """
        embeddings = self.get_embeddings(x)
        
        if return_fbc:
            sed_logits = self.sed_head(embeddings)
            fbc_logits = self.fbc_head(embeddings)
            return embeddings, sed_logits, fbc_logits
        
        return embeddings
    
    def compute_multitask_loss(self, x, sed_labels, fbc_labels, sed_weight=1.0, fbc_weight=1.0):
        """Compute combined SED + FBC loss for multi-task training.
        
        Args:
            x: (batch, seq_len, n_mels)
            sed_labels: (batch,) class labels for SED
            fbc_labels: (batch,) binary labels for FBC (0=background, 1=foreground)
            sed_weight: Weight for SED loss
            fbc_weight: Weight for FBC loss
            
        Returns:
            total_loss, sed_loss, fbc_loss
        """
        embeddings, sed_logits, fbc_logits = self.forward(x, return_fbc=True)
        
        sed_loss = F.cross_entropy(sed_logits, sed_labels)
        fbc_loss = F.cross_entropy(fbc_logits, fbc_labels)
        
        total_loss = sed_weight * sed_loss + fbc_weight * fbc_loss
        
        return total_loss, sed_loss, fbc_loss


class ResNetMamba(nn.Module):
    """ResNet backbone + Mamba encoder.
    
    This extends the existing ResNet architecture with Mamba blocks,
    while maintaining compatibility with the baseline training pipeline.
    """
    
    def __init__(self, features, use_mamba=True, mamba_layers=2):
        super().__init__()
        
        # Import and use the existing ResNet architecture
        from src.models.components.protonet import BasicBlock
        
        drop_rate = features.drop_rate
        self.inplanes = 1
        self.features = features
        
        # ResNet layers (same as baseline)
        self.layer1 = self._make_layer(BasicBlock, 64, stride=2, features=features)
        self.layer2 = self._make_layer(BasicBlock, 128, stride=2, features=features)
        self.layer3 = self._make_layer(
            BasicBlock, 64, stride=2, drop_block=True, block_size=5, features=features
        )
        
        if features.layer_4:
            self.layer4 = self._make_layer(
                BasicBlock, 64, stride=2, drop_block=True, block_size=5, features=features
            )
        
        # Pooling
        self.pool_avg = nn.AdaptiveAvgPool2d(
            (features.time_max_pool_dim, int(features.embedding_dim / (features.time_max_pool_dim * 64)))
        )
        
        # Calculate Mamba input size after layer3 (or layer4)
        # After layer3, shape is (B, 64, H, W) where W depends on input mel bins
        # For 128 mels and 3 layers with stride 2: W = 128/2/2/2 = 16
        # We reshape to (B, H, C*W) for Mamba, so d_model = C * W = 64 * 16 = 1024
        self.mamba_dim = 1024  # 64 channels * 16 width after layer3
        
        # Mamba encoder
        self.use_mamba = use_mamba
        if use_mamba:
            self.mamba = MambaEncoder(
                d_model=self.mamba_dim,
                n_layers=mamba_layers,
            )
        
        # FBC head for multi-task training
        self.fbc_head = nn.Linear(features.embedding_dim, 2)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=features.non_linearity)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, stride=1, drop_block=False, block_size=1, features=None):
        from src.models.components.protonet import BasicBlock
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layer = block(
            self.inplanes, planes, stride, downsample,
            features.drop_rate, drop_block, block_size,
            features.with_bias, features.non_linearity,
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(layer)
    
    def forward(self, x, return_fbc=False):
        """Forward pass.
        
        Args:
            x: (batch, seq_len, n_mels)
            return_fbc: If True, also return FBC logits
            
        Returns:
            embeddings or (embeddings, fbc_logits)
        """
        batch, seq_len, n_mels = x.shape
        x = x.view(-1, 1, seq_len, n_mels)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.features.layer_4:
            x = self.layer4(x)
        
        # Apply Mamba before pooling (on temporal dimension)
        if self.use_mamba:
            b, c, h, w = x.shape
            # Reshape: (B, C, H, W) → (B, H, C*W) for sequence modeling over time
            x_seq = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
            x_seq = self.mamba(x_seq)
            x = x_seq.reshape(b, h, c, w).permute(0, 2, 1, 3)
        
        # Pooling
        x = self.pool_avg(x)
        embeddings = x.view(x.size(0), -1)
        
        if return_fbc:
            fbc_logits = self.fbc_head(embeddings)
            return embeddings, fbc_logits
        
        return embeddings


if __name__ == "__main__":
    # Test the model
    model = ProtoNetMamba(n_mels=128, embed_dim=2048, num_classes=40)
    print(f"ProtoNetMamba parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 17, 128)  # (batch, seq_len, n_mels)
    embeddings = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Test multi-task
    embeddings, sed_logits, fbc_logits = model(x, return_fbc=True)
    print(f"SED logits shape: {sed_logits.shape}")
    print(f"FBC logits shape: {fbc_logits.shape}")
