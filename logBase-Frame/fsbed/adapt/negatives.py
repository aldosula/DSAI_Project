"""Negative sampling and hard-negative mining."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ..data.annotations import Annotation

logger = logging.getLogger(__name__)


def get_negative_mask(
    num_frames: int,
    positive_events: list[Annotation],
    hop_length: int,
    sample_rate: int,
    margin_frames: int = 5,
) -> torch.Tensor:
    """
    Create binary mask marking negative (background) frames.

    Args:
        num_frames: Total number of frames.
        positive_events: List of positive events.
        hop_length: Hop length in samples.
        sample_rate: Sample rate in Hz.
        margin_frames: Margin around positive events to exclude.

    Returns:
        Boolean mask [num_frames] where True = negative.
    """
    mask = torch.ones(num_frames, dtype=torch.bool)

    for event in positive_events:
        start_frame = int(event.starttime * sample_rate / hop_length)
        end_frame = int(event.endtime * sample_rate / hop_length)

        # Add margin
        start_frame = max(0, start_frame - margin_frames)
        end_frame = min(num_frames, end_frame + margin_frames)

        mask[start_frame:end_frame] = False

    return mask


def sample_negatives(
    embeddings: torch.Tensor,
    negative_mask: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Randomly sample negative embeddings.

    Args:
        embeddings: All frame embeddings [num_frames, embed_dim].
        negative_mask: Boolean mask [num_frames] where True = negative.
        num_samples: Number of negatives to sample.

    Returns:
        Sampled negative embeddings [num_samples, embed_dim].
    """
    negative_indices = torch.where(negative_mask)[0]

    if len(negative_indices) == 0:
        logger.warning("No negative frames available")
        return embeddings[:num_samples]

    if len(negative_indices) < num_samples:
        # Sample with replacement
        indices = negative_indices[
            torch.randint(len(negative_indices), (num_samples,))
        ]
    else:
        # Sample without replacement
        perm = torch.randperm(len(negative_indices))[:num_samples]
        indices = negative_indices[perm]

    return embeddings[indices]


def hard_negative_mining(
    embeddings: torch.Tensor,
    negative_mask: torch.Tensor,
    pos_prototype: torch.Tensor,
    k: int = 1500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mine hard negatives: negatives most similar to positive prototype.

    Args:
        embeddings: All frame embeddings [num_frames, embed_dim].
        negative_mask: Boolean mask [num_frames] where True = negative.
        pos_prototype: Positive class prototype [embed_dim].
        k: Number of hard negatives to select.

    Returns:
        Tuple of (hard_negative_embeddings, hard_negative_indices).
    """
    negative_indices = torch.where(negative_mask)[0]

    if len(negative_indices) == 0:
        logger.warning("No negative frames for hard mining")
        return embeddings[:k], torch.arange(k)

    # Get negative embeddings
    neg_embeddings = embeddings[negative_indices]

    # Compute cosine similarity to positive prototype
    neg_normalized = F.normalize(neg_embeddings, dim=1)
    pos_normalized = F.normalize(pos_prototype.unsqueeze(0), dim=1)
    similarities = torch.matmul(neg_normalized, pos_normalized.T).squeeze()

    # Select top-k most similar (hardest negatives)
    k = min(k, len(similarities))
    _, hard_indices = torch.topk(similarities, k)

    hard_neg_embeddings = neg_embeddings[hard_indices]
    hard_neg_frame_indices = negative_indices[hard_indices]

    logger.debug(f"Mined {k} hard negatives with similarity range: "
                 f"{similarities[hard_indices].min():.3f} - {similarities[hard_indices].max():.3f}")

    return hard_neg_embeddings, hard_neg_frame_indices


def get_balanced_negatives(
    embeddings: torch.Tensor,
    negative_mask: torch.Tensor,
    pos_prototype: torch.Tensor,
    num_hard: int = 1000,
    num_random: int = 500,
) -> torch.Tensor:
    """
    Get a mix of hard and random negatives.

    Args:
        embeddings: All frame embeddings.
        negative_mask: Negative frame mask.
        pos_prototype: Positive prototype.
        num_hard: Number of hard negatives.
        num_random: Number of random negatives.

    Returns:
        Combined negative embeddings.
    """
    hard_negs, _ = hard_negative_mining(embeddings, negative_mask, pos_prototype, num_hard)
    random_negs = sample_negatives(embeddings, negative_mask, num_random)

    return torch.cat([hard_negs, random_negs], dim=0)
