"""Inference dataset for processing audio files."""

import logging
from pathlib import Path
from typing import Iterator, Optional, Union

import torch

from ..audio.io import load_audio, get_audio_duration
from ..audio.features import LogMelExtractor
from ..audio.chunking import time_to_frame_idx, frame_idx_to_time
from ..config import AudioConfig, InferenceConfig

logger = logging.getLogger(__name__)


class InferenceDataset:
    """
    Dataset for inference on a single audio file.

    Processes audio in chunks and yields frame-level features.
    """

    def __init__(
        self,
        audio_path: Union[str, Path],
        audio_config: AudioConfig,
        inference_config: InferenceConfig,
    ):
        """
        Initialize inference dataset.

        Args:
            audio_path: Path to audio file.
            audio_config: Audio processing configuration.
            inference_config: Inference configuration.
        """
        self.audio_path = Path(audio_path)
        self.audio_config = audio_config
        self.inference_config = inference_config

        self.feature_extractor = LogMelExtractor.from_config(audio_config)

        # Get audio info
        self.duration_s = get_audio_duration(self.audio_path)

        # Calculate chunk parameters
        self.chunk_samples = int(
            inference_config.chunk_duration_s * audio_config.sample_rate
        )
        self.overlap_samples = int(
            inference_config.chunk_overlap_s * audio_config.sample_rate
        )
        self.hop_samples = self.chunk_samples - self.overlap_samples

    def __len__(self) -> int:
        """Number of chunks."""
        total_samples = int(self.duration_s * self.audio_config.sample_rate)
        if total_samples <= self.chunk_samples:
            return 1
        return (total_samples - self.chunk_samples) // self.hop_samples + 1

    def iter_chunks(self) -> Iterator[tuple[torch.Tensor, float, float]]:
        """
        Iterate over audio chunks.

        Yields:
            Tuples of (features, start_time, end_time).
            Features shape: [1, n_mels, time_frames]
        """
        # Load full audio
        waveform, sr = load_audio(
            self.audio_path,
            target_sr=self.audio_config.sample_rate,
            mono=True,
        )

        total_samples = waveform.shape[-1]
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + self.chunk_samples, total_samples)
            chunk = waveform[..., start_sample:end_sample]

            # Pad if needed
            if chunk.shape[-1] < self.chunk_samples:
                pad_amount = self.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_amount))

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(chunk.unsqueeze(0))

            start_time = start_sample / self.audio_config.sample_rate
            end_time = end_sample / self.audio_config.sample_rate

            yield features.squeeze(0), start_time, end_time

            start_sample += self.hop_samples

    def get_full_features(self) -> tuple[torch.Tensor, int]:
        """
        Get features for entire file (use for short files).

        Returns:
            Tuple of (features, num_frames).
            Features shape: [1, n_mels, total_frames]
        """
        waveform, sr = load_audio(
            self.audio_path,
            target_sr=self.audio_config.sample_rate,
            mono=True,
        )

        with torch.no_grad():
            features = self.feature_extractor(waveform.unsqueeze(0))

        num_frames = features.shape[-1]
        return features.squeeze(0), num_frames

    def time_to_frame(self, time_s: float) -> int:
        """Convert time to frame index."""
        return time_to_frame_idx(
            time_s,
            self.audio_config.hop_length,
            self.audio_config.sample_rate,
        )

    def frame_to_time(self, frame: int) -> float:
        """Convert frame index to time."""
        return frame_idx_to_time(
            frame,
            self.audio_config.hop_length,
            self.audio_config.sample_rate,
        )
