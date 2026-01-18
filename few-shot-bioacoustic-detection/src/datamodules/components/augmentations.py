"""
Data augmentation utilities for few-shot bioacoustic event detection.

Provides configurable augmentations:
- SpecAugment: Frequency and time masking
- TimeShift: Random circular shift along time axis
- Mixup: Blend two samples (applied at batch level)
"""

import numpy as np
from typing import Optional, Tuple


class SpecAugment:
    """SpecAugment: frequency and time masking for spectrograms.
    
   
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to spectrogram.
        
        Args:
            x: Spectrogram of shape [time, freq] or [freq, time]
        
        Returns:
            Augmented spectrogram
        """
        x = x.copy()
        time_dim, freq_dim = x.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, freq_dim))
            f0 = np.random.randint(0, freq_dim - f)
            x[:, f0:f0 + f] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, time_dim))
            t0 = np.random.randint(0, time_dim - t)
            x[t0:t0 + t, :] = 0
        
        return x


class TimeShift:
    """Random circular shift along time axis."""
    
    def __init__(self, max_shift_ratio: float = 0.2):
        """
        Args:
            max_shift_ratio: Maximum shift as ratio of total length
        """
        self.max_shift_ratio = max_shift_ratio
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply time shift to spectrogram.
        
        Args:
            x: Spectrogram of shape [time, freq]
        
        Returns:
            Time-shifted spectrogram
        """
        time_dim = x.shape[0]
        max_shift = int(time_dim * self.max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=0)


class Mixup:
    """Mixup augmentation for few-shot learning.
    
    Blends two samples: x_mixed = lambda * x1 + (1 - lambda) * x2
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def __call__(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Mix two samples.
        
        Args:
            x1: First sample
            x2: Second sample
        
        Returns:
            Mixed sample and mixing coefficient lambda
        """
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed, lam


class AugmentationPipeline:
    """Composable augmentation pipeline with configurable toggles."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Augmentation config dict with enabled flags
        """
        self.enabled = config.get("enabled", True)
        
        # Initialize augmentations based on config
        self.spec_augment = None
        self.time_shift = None
        self.mixup = None
        
        if self.enabled:
            spec_cfg = config.get("spec_augment", {})
            if spec_cfg.get("enabled", False):
                self.spec_augment = SpecAugment(
                    freq_mask_param=spec_cfg.get("freq_mask_param", 10),
                    time_mask_param=spec_cfg.get("time_mask_param", 20),
                    num_freq_masks=spec_cfg.get("num_freq_masks", 2),
                    num_time_masks=spec_cfg.get("num_time_masks", 2),
                )
            
            shift_cfg = config.get("time_shift", {})
            if shift_cfg.get("enabled", False):
                self.time_shift = TimeShift(
                    max_shift_ratio=shift_cfg.get("max_shift_ratio", 0.2)
                )
            
            mixup_cfg = config.get("mixup", {})
            if mixup_cfg.get("enabled", False):
                self.mixup = Mixup(alpha=mixup_cfg.get("alpha", 0.4))
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to a single sample.
        
        Args:
            x: Input spectrogram
        
        Returns:
            Augmented spectrogram
        """
        if not self.enabled:
            return x
        
        if self.time_shift is not None:
            x = self.time_shift(x)
        
        if self.spec_augment is not None:
            x = self.spec_augment(x)
        
        return x
    
    def apply_mixup(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Apply mixup if enabled.
        
        Returns:
            Mixed sample and lambda (1.0 if mixup disabled)
        """
        if self.mixup is not None:
            return self.mixup(x1, x2)
        return x1, 1.0
