"""Pseudo-labeling for semi-supervised adaptation."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def select_pseudo_positives(
    probs: torch.Tensor,
    threshold: float = 0.95,
    max_per_iter: int = 50,
    min_spacing_frames: int = 10,
    existing_positive_frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Select high-confidence pseudo-positive frames.

    Uses temporal NMS to avoid selecting adjacent frames.

    Args:
        probs: Positive class probabilities [num_frames].
        threshold: Confidence threshold.
        max_per_iter: Maximum pseudo-labels per iteration.
        min_spacing_frames: Minimum spacing between selected frames.
        existing_positive_frames: Already labeled positive frames to exclude.

    Returns:
        Indices of selected pseudo-positive frames.
    """
    # Find high-confidence frames
    high_conf_mask = probs >= threshold

    # Exclude existing positives
    if existing_positive_frames is not None:
        for idx in existing_positive_frames:
            start = max(0, idx - min_spacing_frames)
            end = min(len(probs), idx + min_spacing_frames + 1)
            high_conf_mask[start:end] = False

    high_conf_indices = torch.where(high_conf_mask)[0]

    if len(high_conf_indices) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by probability (descending)
    high_conf_probs = probs[high_conf_indices]
    sorted_order = torch.argsort(high_conf_probs, descending=True)
    sorted_indices = high_conf_indices[sorted_order]

    # Temporal NMS: select frames with minimum spacing
    selected = []
    selected_set = set()

    for idx in sorted_indices.tolist():
        # Check spacing with already selected
        is_valid = True
        for sel_idx in selected:
            if abs(idx - sel_idx) < min_spacing_frames:
                is_valid = False
                break

        if is_valid:
            selected.append(idx)
            if len(selected) >= max_per_iter:
                break

    logger.debug(f"Selected {len(selected)} pseudo-positives from {len(high_conf_indices)} candidates")

    return torch.tensor(selected, dtype=torch.long)


def update_prototype(
    current_prototype: torch.Tensor,
    new_embeddings: torch.Tensor,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Update prototype with new embeddings using momentum.

    Args:
        current_prototype: Current prototype [embed_dim].
        new_embeddings: New embeddings [num_new, embed_dim].
        momentum: Momentum for update (higher = more weight on current).

    Returns:
        Updated prototype.
    """
    new_proto = new_embeddings.mean(dim=0)
    updated = momentum * current_prototype + (1 - momentum) * new_proto
    return updated


def pseudo_label_iteration(
    embeddings: torch.Tensor,
    probs: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    threshold: float = 0.95,
    max_add: int = 50,
    min_spacing_frames: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Perform one pseudo-labeling iteration.

    Args:
        embeddings: All frame embeddings [num_frames, embed_dim].
        probs: Current positive probabilities [num_frames].
        positive_indices: Current positive frame indices.
        negative_indices: Current negative frame indices.
        threshold: Pseudo-label threshold.
        max_add: Max new positives per iteration.
        min_spacing_frames: Min spacing between positives.

    Returns:
        Tuple of (updated_positive_indices, updated_negative_indices, num_added).
    """
    # Find new pseudo-positives
    pseudo_pos = select_pseudo_positives(
        probs,
        threshold=threshold,
        max_per_iter=max_add,
        min_spacing_frames=min_spacing_frames,
        existing_positive_frames=positive_indices,
    )

    if len(pseudo_pos) == 0:
        return positive_indices, negative_indices, 0

    # Add to positives
    updated_pos = torch.cat([positive_indices, pseudo_pos])

    # Remove from negatives (with margin)
    updated_neg_mask = torch.ones(len(negative_indices), dtype=torch.bool)
    for new_pos in pseudo_pos:
        for i, neg_idx in enumerate(negative_indices):
            if abs(neg_idx.item() - new_pos.item()) < min_spacing_frames:
                updated_neg_mask[i] = False

    updated_neg = negative_indices[updated_neg_mask]

    return updated_pos, updated_neg, len(pseudo_pos)
