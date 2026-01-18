"""Loss functions for training."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces loss for well-classified examples, focusing on hard examples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights [num_classes]. If None, no weighting.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits [batch, num_classes] or [batch, time, num_classes].
            targets: Class indices [batch] or [batch, time].

        Returns:
            Loss value.
        """
        ce_loss = F.cross_entropy(
            inputs.reshape(-1, inputs.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined cross-entropy and optional contrastive loss.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize combined loss.

        Args:
            ce_weight: Weight for cross-entropy loss.
            focal_gamma: Focal loss gamma (0 = standard CE).
            class_weights: Optional class weights.
        """
        super().__init__()
        self.ce_weight = ce_weight

        if focal_gamma > 0:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            logits: Model logits.
            targets: Ground truth labels.

        Returns:
            Total loss.
        """
        # Reshape for cross entropy
        if logits.dim() == 3:
            # [batch, time, classes] -> [batch*time, classes]
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)

        return self.ce_weight * self.ce_loss(logits, targets)


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for embedding learning.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Normalized embeddings [batch, embed_dim].
            labels: Class labels [batch].

        Returns:
            Contrastive loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Exclude self-similarity
        self_mask = torch.eye(mask.shape[0], device=mask.device)
        mask = mask - self_mask

        # Compute log softmax
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss
