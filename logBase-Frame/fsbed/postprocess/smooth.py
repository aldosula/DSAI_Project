"""Probability smoothing utilities."""

import logging
from typing import Union

import torch
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def smooth_probs(
    probs: Union[torch.Tensor, np.ndarray],
    method: str = "median",
    kernel_size: int = 5,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Smooth probability sequence.

    Args:
        probs: Probability sequence [num_frames] or [batch, num_frames].
        method: 'median' or 'moving_avg'.
        kernel_size: Smoothing kernel size (should be odd).

    Returns:
        Smoothed probabilities.
    """
    is_tensor = isinstance(probs, torch.Tensor)
    if is_tensor:
        device = probs.device
        probs_np = probs.cpu().numpy()
    else:
        probs_np = probs

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == "median":
        smoothed = _median_filter(probs_np, kernel_size)
    elif method == "moving_avg":
        smoothed = _moving_average(probs_np, kernel_size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    if is_tensor:
        return torch.from_numpy(smoothed).to(device)
    return smoothed


def _median_filter(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median filter to 1D or 2D array."""
    if x.ndim == 1:
        return ndimage.median_filter(x, size=kernel_size, mode="nearest")
    else:
        # Apply along last dimension
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            result[i] = ndimage.median_filter(x[i], size=kernel_size, mode="nearest")
        return result


def _moving_average(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply moving average filter."""
    kernel = np.ones(kernel_size) / kernel_size

    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    else:
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            result[i] = np.convolve(x[i], kernel, mode="same")
        return result


def kernel_size_from_seconds(
    seconds: float,
    hop_length: int,
    sample_rate: int,
) -> int:
    """
    Convert smoothing window in seconds to kernel size in frames.

    Args:
        seconds: Window duration in seconds.
        hop_length: Hop length in samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Kernel size (odd integer).
    """
    frames = int(seconds * sample_rate / hop_length)
    # Ensure odd
    if frames % 2 == 0:
        frames += 1
    return max(3, frames)  # Minimum 3
