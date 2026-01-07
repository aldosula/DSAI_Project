"""Classification heads and prototype initialization."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BinaryHead(nn.Module):
    """
    Binary classification head.

    Maps embeddings to {NEG, POS} logits.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize binary classification head.

        Args:
            embed_dim: Input embedding dimension.
            hidden_dim: Optional hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Input embeddings [batch, time, embed_dim] or [batch, embed_dim].

        Returns:
            Logits [batch, time, 2] or [batch, 2].
        """
        return self.classifier(embeddings)

    def get_probs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get positive class probabilities."""
        logits = self.forward(embeddings)
        probs = F.softmax(logits, dim=-1)
        return probs[..., 1]  # Positive class probability


class PrototypeInit:
    """
    Initialize classification head using prototypes.

    Computes positive and negative prototypes from embeddings
    and initializes linear weights accordingly.
    """

    @staticmethod
    def compute_prototype(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype as mean of embeddings.

        Args:
            embeddings: Embeddings [num_samples, embed_dim].

        Returns:
            Prototype [embed_dim].
        """
        return embeddings.mean(dim=0)

    @staticmethod
    def init_head_from_prototypes(
        head: BinaryHead,
        pos_prototype: torch.Tensor,
        neg_prototype: torch.Tensor,
    ) -> None:
        """
        Initialize head weights from prototypes.

        Uses prototype difference as decision boundary.

        Args:
            head: BinaryHead to initialize.
            pos_prototype: Positive class prototype [embed_dim].
            neg_prototype: Negative class prototype [embed_dim].
        """
        # Get the final linear layer
        if isinstance(head.classifier, nn.Sequential):
            linear = head.classifier[-1]
        else:
            linear = head.classifier

        with torch.no_grad():
            # Weight initialization:
            # w_neg = neg_prototype
            # w_pos = pos_prototype
            linear.weight[0] = neg_prototype
            linear.weight[1] = pos_prototype

            # Bias: centered between prototypes
            linear.bias[0] = -torch.dot(neg_prototype, neg_prototype) / 2
            linear.bias[1] = -torch.dot(pos_prototype, pos_prototype) / 2

        logger.info("Initialized head from prototypes")

    @staticmethod
    def compute_and_init(
        head: BinaryHead,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor,
    ) -> None:
        """
        Compute prototypes and initialize head.

        Args:
            head: BinaryHead to initialize.
            pos_embeddings: Positive embeddings [num_pos, embed_dim].
            neg_embeddings: Negative embeddings [num_neg, embed_dim].
        """
        pos_proto = PrototypeInit.compute_prototype(pos_embeddings)
        neg_proto = PrototypeInit.compute_prototype(neg_embeddings)
        PrototypeInit.init_head_from_prototypes(head, pos_proto, neg_proto)


class MultiClassHead(nn.Module):
    """
    Multi-class classification head for training.

    Used during backbone pretraining on training set classes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
