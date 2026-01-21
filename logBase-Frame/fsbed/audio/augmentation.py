"""Data augmentation for audio spectrograms.

Implements Mixup and SpecAugment.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch


def mixup(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup data augmentation.
    
    Interpolates inputs and targets from two samples.
    
    Args:
        x1: First input tensor.
        x2: Second input tensor.
        y1: First target (one-hot or soft labels).
        y2: Second target.
        alpha: Beta distribution parameter.
        
    Returns:
        Tuple of (mixed_x, mixed_y).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    
    return mixed_x, mixed_y


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup to a batch by shuffling and mixing.
    
    Args:
        x: Batch of inputs [batch, ...].
        y: Batch of targets [batch, num_classes] (one-hot).
        alpha: Beta distribution parameter.
        
    Returns:
        Tuple of (mixed_x, mixed_y).
    """
    batch_size = x.size(0)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


class SpecAugment:
    """
    SpecAugment data augmentation for spectrograms.
    
    Applies frequency and time masking.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum frequency mask width (F).
            time_mask_param: Maximum time mask width (T).
            num_freq_masks: Number of frequency masks to apply.
            num_time_masks: Number of time masks to apply.
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.
        
        Args:
            spectrogram: Input tensor [batch, channels, freq, time] or [channels, freq, time].
            
        Returns:
            Augmented spectrogram.
        """
        # Handle both batched and single inputs
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch, channels, freq_bins, time_steps = spectrogram.shape
        augmented = spectrogram.clone()
        
        for b in range(batch):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = random.randint(0, min(self.freq_mask_param, freq_bins - 1))
                f0 = random.randint(0, freq_bins - f)
                augmented[b, :, f0:f0 + f, :] = 0
            
            # Time masking
            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, time_steps - 1))
                t0 = random.randint(0, time_steps - t)
                augmented[b, :, :, t0:t0 + t] = 0
        
        if squeeze:
            augmented = augmented.squeeze(0)
        
        return augmented


def apply_augmentation(
    features: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float = 0.4,
    spec_augment: Optional[SpecAugment] = None,
    apply_mixup: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply full augmentation pipeline to a batch.
    
    Args:
        features: Batch of spectrograms [batch, channels, freq, time].
        labels: Batch of labels [batch] (will be converted to soft labels for mixup).
        mixup_alpha: Mixup alpha parameter.
        spec_augment: SpecAugment instance (or None to skip).
        apply_mixup: Whether to apply mixup.
        
    Returns:
        Tuple of (augmented_features, soft_labels).
    """
    batch_size = features.size(0)
    
    # Apply SpecAugment first (per-sample)
    if spec_augment is not None:
        features = spec_augment(features)
    
    # Convert labels to one-hot for mixup
    if labels.dim() == 1:
        num_classes = labels.max().item() + 1
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        labels = one_hot
    
    # Apply mixup
    if apply_mixup and mixup_alpha > 0:
        features, labels = mixup_batch(features, labels, mixup_alpha)
    
    return features, labels
