"""Two-step fine-tuning 

Implements: 
1. Step 1: Joint training of FC(1024,20) + FC(1024,2)
   - FC(1024,20): Multi-class head for regularization (keeps backbone features sharp)
   - FC(1024,2): Binary head for target species detection
2. Step 2: Pseudo-labeling with high-confidence predictions
3. Repeat for N iterations
"""

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..config import AdaptConfig, AudioConfig, Config
from ..models.heads import BinaryHead

logger = logging.getLogger(__name__)


class DualHeadAdapter(nn.Module):
    """
    Dual-head adapter for Step 1 fine-tuning.
    
    Combines:
    - FC(embed_dim, 20): Multi-class head from pretraining (regularization)
    - FC(embed_dim, 2): Binary head for new species detection
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_pretrain_classes: int = 20,
        pretrained_classifier: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-class head (loaded from pretrained or initialized)
        if pretrained_classifier is not None:
            self.multiclass_head = pretrained_classifier
        else:
            self.multiclass_head = nn.Linear(embed_dim, num_pretrain_classes)
        
        # Binary head for new species
        self.binary_head = nn.Linear(embed_dim, 2)
        
        # Initialize binary head
        nn.init.kaiming_normal_(self.binary_head.weight)
        nn.init.constant_(self.binary_head.bias, 0)
    
    def forward_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Binary classification forward."""
        return self.binary_head(x)
    
    def forward_multiclass(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-class classification forward."""
        return self.multiclass_head(x)
    
    def get_binary_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of positive class."""
        logits = self.binary_head(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # Probability of class 1 (positive)
    
    def init_binary_from_prototypes(
        self,
        pos_proto: torch.Tensor,
        neg_proto: torch.Tensor,
    ):
        """Initialize binary head weights from prototypes."""
        with torch.no_grad():
            self.binary_head.weight[0] = neg_proto
            self.binary_head.weight[1] = pos_proto
            self.binary_head.bias[0] = -torch.dot(neg_proto, neg_proto) / 2
            self.binary_head.bias[1] = -torch.dot(pos_proto, pos_proto) / 2
        logger.info("Initialized binary head from prototypes")


def get_inter_segment_negatives(
    num_frames: int,
    support_frames: List[int],
    margin_frames: int = 10,
) -> torch.Tensor:
    """
    Get negative frame indices from regions between support events.

    Args:
        num_frames: Total number of frames.
        support_frames: Sorted list of support event center frames.
        margin_frames: Margin around support events to exclude.
        
    Returns:
        Tensor of negative frame indices.
    """
    if len(support_frames) < 2:
        # Fallback: use frames outside support
        mask = torch.ones(num_frames, dtype=torch.bool)
        for frame in support_frames:
            start = max(0, frame - margin_frames)
            end = min(num_frames, frame + margin_frames + 1)
            mask[start:end] = False
        return torch.where(mask)[0]
    
    # Get segments between consecutive support events
    negative_indices = []
    sorted_frames = sorted(support_frames)
    
    for i in range(len(sorted_frames) - 1):
        start = sorted_frames[i] + margin_frames
        end = sorted_frames[i + 1] - margin_frames
        
        if end > start:
            # Sample center of this segment
            center = (start + end) // 2
            # Take a few frames around center
            for offset in range(-5, 6):
                idx = center + offset
                if start <= idx < end:
                    negative_indices.append(idx)
    
    return torch.tensor(negative_indices)


def two_step_finetune(
    backbone: nn.Module,
    embeddings: torch.Tensor,
    support_frames: torch.Tensor,
    num_frames: int,
    config: Config,
    device: torch.device,
    pretrained_classifier: Optional[nn.Linear] = None,
) -> Tuple[DualHeadAdapter, torch.Tensor]:
    """
    Two-step fine-tuning per file
    
    Step 1: Joint training of dual heads
    - Binary head: Pos (support) vs Neg (inter-segment)
    - Multi-class head: Regularization to maintain feature quality
    
    Step 2: Pseudo-labeling
    - High-confidence predictions added to positive set
    - Repeat for N iterations
    
    Args:
        backbone: Embedding backbone (frozen during fine-tuning).
        embeddings: All frame embeddings [num_frames, embed_dim].
        support_frames: Indices of support positive frames.
        num_frames: Total number of frames.
        config: Configuration.
        device: Compute device.
        pretrained_classifier: Optional pretrained 20-class head.
        
    Returns:
        Tuple of (DualHeadAdapter, frame probabilities).
    """
    adapt_config = config.adapt
    audio_config = config.audio
    
    # Create dual-head adapter
    embed_dim = embeddings.shape[-1]
    adapter = DualHeadAdapter(
        embed_dim=embed_dim,
        num_pretrain_classes=20,
        pretrained_classifier=pretrained_classifier,
    ).to(device)
    
    # Margin in frames
    margin_frames = int(0.1 * audio_config.sample_rate / audio_config.hop_length)
    
    # Get initial positive frames from support
    pos_frames = support_frames.clone()
    
    # Get support end frame (where query region starts)
    support_frames_list = support_frames.tolist()
    support_end_frame = max(support_frames_list) + margin_frames
    

    neg_frames = get_inter_segment_negatives(
        num_frames, 
        support_frames_list,
        margin_frames,
    ).to(device)
    
    # If inter-segment negatives are insufficient, supplement with query region negatives
    if len(neg_frames) < 50:
        logger.info(f"Only {len(neg_frames)} inter-segment negatives, supplementing with query region")
        query_length = num_frames - support_end_frame
        if query_length > 0:
            query_indices = torch.arange(support_end_frame, num_frames, device=device)
            target_additional = min(100 - len(neg_frames), len(query_indices))
            if target_additional > 0:
                perm = torch.randperm(len(query_indices), device=device)
                additional_neg = query_indices[perm[:target_additional]]
                neg_frames = torch.cat([neg_frames, additional_neg])
        
    # Check if we have any negatives
    if len(neg_frames) == 0:
        logger.warning("No negatives found! Using random frames as negatives")
        all_indices = torch.arange(num_frames, device=device)
        mask = torch.ones(num_frames, dtype=torch.bool, device=device)
        mask[support_frames] = False
        candidates = all_indices[mask]
        if len(candidates) > 0:
             perm = torch.randperm(len(candidates), device=device)
             neg_frames = candidates[perm[:min(100, len(candidates))]]
    
    # Initialize with prototypes
    pos_emb = embeddings[pos_frames]
    if len(neg_frames) > 0:
        neg_emb = embeddings[neg_frames]
    else:
        neg_emb = torch.zeros_like(pos_emb[:1])
    
    # Initialize binary head from prototypes
    pos_proto = pos_emb.mean(dim=0)
    neg_proto = neg_emb.mean(dim=0)
    adapter.init_binary_from_prototypes(pos_proto, neg_proto)
    
    # Optimizer
    optimizer = optim.Adam([
        {'params': adapter.multiclass_head.parameters(), 'lr': 0.0001},
        {'params': adapter.binary_head.parameters(), 'lr': 0.001},
    ])
    
    iterations = getattr(adapt_config, 'iterations', 3)
    pseudo_threshold = getattr(adapt_config, 'pseudo_threshold', 0.9)
    pseudo_max = getattr(adapt_config, 'pseudo_max_per_iter', 50)
    
    # Loss weights for dual-head training
    binary_weight = 1.0
    multiclass_weight = 0.1  # Regularization term
    
    for iteration in range(iterations):
        # Current positive and negative embeddings
        current_pos_emb = embeddings[pos_frames]
        current_neg_emb = embeddings[neg_frames] if len(neg_frames) > 0 else neg_emb
        
        # Balance batch
        n_pos = len(current_pos_emb)
        n_neg = min(len(current_neg_emb), n_pos * 3)
        
        if n_neg > 0:
            neg_sample_idx = torch.randperm(len(current_neg_emb))[:n_neg]
            current_neg_emb = current_neg_emb[neg_sample_idx]
        
        # Create batch for binary classification
        batch_emb = torch.cat([current_pos_emb, current_neg_emb], dim=0)
        binary_labels = torch.cat([
            torch.ones(n_pos, dtype=torch.long, device=device),
            torch.zeros(n_neg, dtype=torch.long, device=device),
        ])
        
        # Shuffle
        perm = torch.randperm(len(batch_emb))
        batch_emb = batch_emb[perm]
        binary_labels = binary_labels[perm]
        
        # Fine-tune with dual heads
        adapter.train()
        optimizer.zero_grad()
        
        # Binary loss (main objective)
        binary_logits = adapter.forward_binary(batch_emb)
        binary_loss = F.cross_entropy(binary_logits, binary_labels)
        
        # Multi-class loss (regularization)
        multiclass_logits = adapter.forward_multiclass(batch_emb)
        multiclass_probs = F.softmax(multiclass_logits, dim=-1)
        entropy = -(multiclass_probs * torch.log(multiclass_probs + 1e-8)).sum(dim=-1).mean()
        multiclass_loss = -entropy
        
        # Total loss
        total_loss = binary_weight * binary_loss + multiclass_weight * multiclass_loss
        total_loss.backward()
        optimizer.step()
        
        logger.debug(f"Iter {iteration + 1}: binary_loss={binary_loss.item():.4f}, "
                    f"multiclass_reg={multiclass_loss.item():.4f}")
        
        # Pseudo-labeling (except last iteration)
        if iteration < iterations - 1:
            adapter.eval()
            with torch.no_grad():
                all_probs = adapter.get_binary_probs(embeddings)
            
            # Find high-confidence positives
            high_conf_mask = all_probs > pseudo_threshold
            
            # Exclude existing positives and their margins
            for idx in pos_frames:
                start = max(0, idx - margin_frames)
                end = min(num_frames, idx + margin_frames + 1)
                high_conf_mask[start:end] = False
            
            # Select pseudo-positives
            pseudo_indices = torch.where(high_conf_mask)[0]
            
            if len(pseudo_indices) > pseudo_max:
                # Keep highest confidence
                pseudo_probs = all_probs[pseudo_indices]
                top_k = torch.topk(pseudo_probs, pseudo_max).indices
                pseudo_indices = pseudo_indices[top_k]
            
            if len(pseudo_indices) > 0:
                pos_frames = torch.cat([pos_frames, pseudo_indices])
                logger.debug(f"Added {len(pseudo_indices)} pseudo-positives, total={len(pos_frames)}")
    
    # Final predictions
    adapter.eval()
    with torch.no_grad():
        probs = adapter.get_binary_probs(embeddings)
    
    logger.info(f"Dual-head fine-tuning complete. Final positives: {len(pos_frames)}")
    
    return adapter, probs


# Keep backward compatibility alias
def _init_head(head, pos_proto: torch.Tensor, neg_proto: torch.Tensor):
    """Initialize head weights from prototypes (deprecated, use DualHeadAdapter)."""
    if hasattr(head, 'init_binary_from_prototypes'):
        head.init_binary_from_prototypes(pos_proto, neg_proto)
    else:
        linear = head.classifier if not isinstance(head.classifier, nn.Sequential) else head.classifier[-1]
        with torch.no_grad():
            linear.weight[0] = neg_proto
            linear.weight[1] = pos_proto
            linear.bias[0] = -torch.dot(neg_proto, neg_proto) / 2
            linear.bias[1] = -torch.dot(pos_proto, pos_proto) / 2
        logger.info("Initialized head from prototypes")

