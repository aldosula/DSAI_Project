"""Per-file adaptation with fine-tuning and pseudo-labeling."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..config import AdaptConfig, AudioConfig
from ..models.heads import BinaryHead, PrototypeInit
from .negatives import get_negative_mask, hard_negative_mining, sample_negatives
from .pseudo_label import select_pseudo_positives

logger = logging.getLogger(__name__)


def per_file_adaptation(
    backbone: nn.Module,
    head: BinaryHead,
    embeddings: torch.Tensor,
    positive_frames: torch.Tensor,
    num_frames: int,
    adapt_config: AdaptConfig,
    audio_config: AudioConfig,
    device: torch.device,
) -> BinaryHead:
    """
    Perform per-file adaptation with fine-tuning and pseudo-labeling.

    Args:
        backbone: Embedding backbone (frozen or partially trainable).
        head: Binary classification head.
        embeddings: All frame embeddings [num_frames, embed_dim].
        positive_frames: Indices of support positive frames.
        num_frames: Total number of frames.
        adapt_config: Adaptation configuration.
        audio_config: Audio configuration.
        device: Compute device.

    Returns:
        Adapted binary head.
    """
    logger.info(f"Starting per-file adaptation with {len(positive_frames)} support positives")

    # Create negative mask
    margin_frames = int(0.1 * audio_config.sample_rate / audio_config.hop_length)
    negative_mask = torch.ones(num_frames, dtype=torch.bool, device=device)
    for idx in positive_frames:
        start = max(0, idx - margin_frames)
        end = min(num_frames, idx + margin_frames + 1)
        negative_mask[start:end] = False

    # Get positive embeddings
    pos_embeddings = embeddings[positive_frames]

    # Initialize prototype
    pos_prototype = pos_embeddings.mean(dim=0)

    # Mine hard negatives
    hard_neg_emb, hard_neg_idx = hard_negative_mining(
        embeddings,
        negative_mask,
        pos_prototype,
        k=adapt_config.hard_neg_k,
    )

    # Initialize head from prototypes
    neg_prototype = hard_neg_emb.mean(dim=0)
    PrototypeInit.init_head_from_prototypes(head, pos_prototype, neg_prototype)

    # Prepare optimizer (only head parameters)
    params_to_train = list(head.parameters())
    optimizer = optim.Adam(params_to_train, lr=adapt_config.finetune_lr)

    # Current positive/negative sets
    current_pos_idx = positive_frames.clone()
    current_neg_idx = hard_neg_idx.clone()

    # Pseudo-labeling iterations
    for iteration in range(adapt_config.iterations):
        logger.debug(f"Adaptation iteration {iteration + 1}/{adapt_config.iterations}")

        # Get current training data
        pos_emb = embeddings[current_pos_idx]
        neg_emb = embeddings[current_neg_idx] if len(current_neg_idx) > 0 else hard_neg_emb

        # Balance batch sizes
        n_pos = len(pos_emb)
        n_neg = min(len(neg_emb), n_pos * 3)  # 3:1 neg:pos ratio

        if n_neg > 0:
            neg_sample_idx = torch.randperm(len(neg_emb))[:n_neg]
            neg_emb_batch = neg_emb[neg_sample_idx]
        else:
            neg_emb_batch = neg_emb

        # Create batch
        batch_emb = torch.cat([pos_emb, neg_emb_batch], dim=0)
        batch_labels = torch.cat([
            torch.ones(n_pos, dtype=torch.long, device=device),
            torch.zeros(n_neg, dtype=torch.long, device=device),
        ])

        # Shuffle
        perm = torch.randperm(len(batch_emb))
        batch_emb = batch_emb[perm]
        batch_labels = batch_labels[perm]

        # Training step
        head.train()
        optimizer.zero_grad()
        logits = head(batch_emb)
        loss = F.cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()

        logger.debug(f"  Loss: {loss.item():.4f}")

        # Pseudo-labeling
        if iteration < adapt_config.iterations - 1:
            head.eval()
            with torch.no_grad():
                all_probs = head.get_probs(embeddings)

            # Select pseudo-positives
            min_spacing = int(adapt_config.pseudo_min_spacing_s * audio_config.sample_rate / audio_config.hop_length)
            pseudo_pos = select_pseudo_positives(
                all_probs,
                threshold=adapt_config.pseudo_threshold,
                max_per_iter=adapt_config.pseudo_max_per_iter,
                min_spacing_frames=min_spacing,
                existing_positive_frames=current_pos_idx,
            )

            if len(pseudo_pos) > 0:
                pseudo_pos = pseudo_pos.to(device)
                current_pos_idx = torch.cat([current_pos_idx, pseudo_pos])

                # Update negative mask
                for idx in pseudo_pos:
                    start = max(0, idx - margin_frames)
                    end = min(num_frames, idx + margin_frames + 1)
                    negative_mask[start:end] = False

                # Re-mine hard negatives
                pos_prototype = embeddings[current_pos_idx].mean(dim=0)
                hard_neg_emb, current_neg_idx = hard_negative_mining(
                    embeddings,
                    negative_mask,
                    pos_prototype,
                    k=adapt_config.hard_neg_k,
                )

                logger.info(f"  Added {len(pseudo_pos)} pseudo-positives, total: {len(current_pos_idx)}")

    head.eval()
    logger.info(f"Adaptation complete. Final positives: {len(current_pos_idx)}")

    return head
