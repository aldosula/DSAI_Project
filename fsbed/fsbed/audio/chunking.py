"""Audio chunking and framing utilities."""

import logging
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


def create_frames(
    features: torch.Tensor,
    frame_size: int,
    hop_size: int,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """
    Create overlapping frames from feature tensor.

    Args:
        features: Feature tensor [batch, channels, freq, time] or [batch, freq, time].
        frame_size: Number of time steps per frame.
        hop_size: Hop between frames.
        pad_mode: Padding mode for incomplete frames.

    Returns:
        Framed features [batch, num_frames, channels, freq, frame_size].
    """
    # Ensure 4D
    if features.dim() == 3:
        features = features.unsqueeze(1)

    batch, channels, freq, time = features.shape

    # Pad to ensure complete frames
    remainder = (time - frame_size) % hop_size
    if remainder != 0:
        pad_amount = hop_size - remainder
        features = torch.nn.functional.pad(features, (0, pad_amount), mode=pad_mode)
        time = features.shape[-1]

    # Calculate number of frames
    num_frames = (time - frame_size) // hop_size + 1

    # Create frames using unfold
    frames = features.unfold(dimension=-1, size=frame_size, step=hop_size)
    # Shape: [batch, channels, freq, num_frames, frame_size]

    # Rearrange to [batch, num_frames, channels, freq, frame_size]
    frames = frames.permute(0, 3, 1, 2, 4)

    return frames


def stitch_frames(
    frames: torch.Tensor,
    hop_size: int,
    original_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Stitch overlapping frames back together.

    Uses averaging for overlapping regions.

    Args:
        frames: Framed tensor [batch, num_frames, ..., frame_size].
        hop_size: Hop between frames.
        original_length: Original time dimension length for trimming.

    Returns:
        Stitched tensor [batch, ..., time].
    """
    batch, num_frames = frames.shape[:2]
    frame_size = frames.shape[-1]
    inner_shape = frames.shape[2:-1]

    # Calculate output length
    output_length = (num_frames - 1) * hop_size + frame_size

    # Initialize output and count tensors
    output = torch.zeros(batch, *inner_shape, output_length, device=frames.device)
    counts = torch.zeros(output_length, device=frames.device)

    # Accumulate frames
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        output[..., start:end] += frames[:, i]
        counts[start:end] += 1

    # Average overlapping regions
    counts = counts.clamp(min=1)
    output = output / counts

    # Trim to original length
    if original_length is not None:
        output = output[..., :original_length]

    return output


def iter_audio_chunks(
    waveform: torch.Tensor,
    chunk_samples: int,
    overlap_samples: int,
) -> Iterator[tuple[torch.Tensor, int]]:
    """
    Iterate over audio chunks.

    Args:
        waveform: Audio tensor [channels, samples].
        chunk_samples: Samples per chunk.
        overlap_samples: Overlap samples between chunks.

    Yields:
        Tuples of (chunk_waveform, start_sample).
    """
    total_samples = waveform.shape[-1]
    hop_samples = chunk_samples - overlap_samples

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[..., start:end]

        # Pad last chunk if needed
        if chunk.shape[-1] < chunk_samples:
            pad_amount = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount))

        yield chunk, start
        start += hop_samples


def time_to_frame_idx(
    time_s: float,
    hop_length: int,
    sample_rate: int,
) -> int:
    """
    Convert time in seconds to frame index.

    Args:
        time_s: Time in seconds.
        hop_length: Hop length in samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Frame index.
    """
    return int(time_s * sample_rate / hop_length)


def frame_idx_to_time(
    frame_idx: int,
    hop_length: int,
    sample_rate: int,
) -> float:
    """
    Convert frame index to time in seconds.

    Args:
        frame_idx: Frame index.
        hop_length: Hop length in samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Time in seconds.
    """
    return frame_idx * hop_length / sample_rate
