import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class EmbeddingNet(nn.Module):
    def __init__(self, backbone='mobilenet_v3_small', weights=None, input_channels=1):
        super(EmbeddingNet, self).__init__()
        
        if backbone == 'mobilenet_v3_small':
            # Use weights parameter instead of deprecated pretrained
            self.backbone = mobilenet_v3_small(weights=weights)
            # Modify first layer for input channels (usually 1 for spectrogram)
            if input_channels != 3:
                old_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    input_channels, old_conv.out_channels, 
                    kernel_size=old_conv.kernel_size, 
                    stride=old_conv.stride, 
                    padding=old_conv.padding, 
                    bias=False
                )
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            self.out_dim = 576 # MobileNetV3-Small output
        else:
            # Simple 4-layer CNN baseline
            self.backbone = nn.Sequential(
                self._conv_block(input_channels, 64),
                self._conv_block(64, 64),
                self._conv_block(64, 64),
                self._conv_block(64, 64),
                nn.Flatten()
            )
            self.out_dim = 64
            
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.backbone(x)

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone='mobilenet_v3_small', weights=MobileNet_V3_Small_Weights.DEFAULT):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = EmbeddingNet(backbone, weights)
        
    def forward(self, support_set, query_set, n_way, n_support, n_query):
        """
        support_set: (n_way * n_support, C, H, W)
        query_set: (n_way * n_query, C, H, W)
        """
        # Embed support and query
        z_support = self.encoder(support_set) # (N*K, D)
        z_query = self.encoder(query_set)     # (N*Q, D)
        
        # Reshape support to (N, K, D) and compute prototypes
        z_support = z_support.view(n_way, n_support, -1)
        prototypes = z_support.mean(dim=1) # (N, D)
        
        # Compute distances
        # dists: (N*Q, N)
        dists = self._euclidean_dist(z_query, prototypes)
        
        return -dists # Logits (negative distance)

    def _euclidean_dist(self, x, y):
        # x: (M, D)
        # y: (N, D)
        # Returns: (M, N)
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
