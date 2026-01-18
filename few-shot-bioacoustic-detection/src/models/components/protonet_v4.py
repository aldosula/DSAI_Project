"""
ProtoNetV4 - Enhanced ResNet with Squeeze-and-Excitation Blocks.

Improvements over baseline:
1. SE (Squeeze-and-Excitation) blocks for channel attention
2. Deeper network (5 layers)
3. Multi-scale pooling (avg + max)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, with_bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=with_bias)


class BasicBlockV4(nn.Module):
    """Enhanced BasicBlock with SE attention."""
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        drop_rate: float = 0.0,
        with_bias: bool = False,
        non_linearity: str = "leaky_relu",
        se_reduction: int = 16,
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
        
        # SE block
        self.se = SqueezeExcitation(planes, se_reduction)
        
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        # Apply SE attention
        out = self.se(out)
        
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNetV4(nn.Module):
    """Enhanced ResNet with SE blocks and wider channels."""
    
    def __init__(
        self,
        features,
        block=BasicBlockV4,
        se_reduction: int = 16,
    ):
        super().__init__()
        
        drop_rate = features.drop_rate
        with_bias = features.with_bias
        
        self.inplanes = 1
        self.se_reduction = se_reduction
        
        # 4 layers with SE blocks (wider channels for more capacity)
        self.layer1 = self._make_layer(block, 64, stride=2, features=features)
        self.layer2 = self._make_layer(block, 128, stride=2, features=features)
        self.layer3 = self._make_layer(block, 256, stride=2, features=features)  # Wider
        self.layer4 = self._make_layer(block, 128, stride=2, features=features)
        
        self.features = features
        self.drop_rate = drop_rate
        
        # Multi-scale pooling
        self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_max = nn.AdaptiveMaxPool2d((1, 1))
        
        # Output projection (128 channels * 2 pooling = 256 -> 2048)
        self.output_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, features.embedding_dim),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=features.non_linearity)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, stride: int = 1, features=None) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
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
                features.with_bias,
                features.non_linearity,
                self.se_reduction,
            )
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, mel_bins = x.shape

        # Pad short sequences to minimum 16 frames (required for 4 maxpool layers)
        if seq_len < 16:
            pad_size = 16 - seq_len
            x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
            seq_len = 16

        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Multi-scale pooling
        x_avg = self.pool_avg(x).view(batch_size, -1)
        x_max = self.pool_max(x).view(batch_size, -1)
        x = torch.cat([x_avg, x_max], dim=1)  # [B, 128]
        
        # Project to embedding dimension
        x = self.output_proj(x)

        return x


if __name__ == "__main__":
    class DummyFeatures:
        drop_rate = 0.1
        with_bias = False
        non_linearity = "leaky_relu"
        embedding_dim = 2048
    
    features = DummyFeatures()
    model = ResNetV4(features)
    print(f"ResNetV4 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    input_tensor = torch.randn((3, 17, 128))
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("SUCCESS!")
