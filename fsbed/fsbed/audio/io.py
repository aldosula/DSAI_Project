"""Audio I/O utilities."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate. If None, keep original.
        mono: If True, convert to mono.

    Returns:
        Tuple of (waveform, sample_rate).
        Waveform shape: [channels, samples] or [1, samples] if mono.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If audio loading fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}") from e

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Audio tensor [channels, samples].
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio tensor.
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def get_audio_info(path: Union[str, Path]) -> tuple[int, int]:
    """
    Get audio file info using soundfile.

    Args:
        path: Path to audio file.

    Returns:
        Tuple of (num_frames, sample_rate).
    """
    info = sf.info(str(path))
    return info.frames, info.samplerate


def get_audio_duration(path: Union[str, Path]) -> float:
    """
    Get audio duration in seconds without loading full file.

    Args:
        path: Path to audio file.

    Returns:
        Duration in seconds.
    """
    num_frames, sample_rate = get_audio_info(path)
    return num_frames / sample_rate


def load_audio_segment(
    path: Union[str, Path],
    start_s: float,
    end_s: float,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Load a segment of audio file.

    Args:
        path: Path to audio file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        target_sr: Target sample rate.
        mono: If True, convert to mono.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    path = Path(path)
    
    # Get audio info using soundfile
    num_frames, orig_sr = get_audio_info(path)

    # Calculate frame offsets
    frame_offset = int(start_s * orig_sr)
    num_frames_to_read = int((end_s - start_s) * orig_sr)

    # Clamp to file bounds
    frame_offset = max(0, min(frame_offset, num_frames - 1))
    num_frames_to_read = min(num_frames_to_read, num_frames - frame_offset)

    try:
        waveform, sr = torchaudio.load(
            str(path),
            frame_offset=frame_offset,
            num_frames=num_frames_to_read,
        )
    except Exception as e:
        logger.warning(f"Failed to load segment from {path}: {e}")
        # Return zeros on failure
        duration_samples = int((end_s - start_s) * (target_sr or orig_sr))
        return torch.zeros(1, max(1, duration_samples)), target_sr or orig_sr

    # Handle empty waveform
    if waveform.numel() == 0:
        duration_samples = int((end_s - start_s) * (target_sr or orig_sr))
        return torch.zeros(1, max(1, duration_samples)), target_sr or orig_sr

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr

