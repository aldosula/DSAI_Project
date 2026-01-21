"""Tests for feature extraction."""

import numpy as np
import pytest
import torch

from fsbed.audio.features import LogMelExtractor, extract_logmel
from fsbed.audio.chunking import (
    create_frames,
    time_to_frame_idx,
    frame_idx_to_time,
)
from fsbed.config import AudioConfig


class TestLogMelExtractor:
    def test_output_shape(self):
        extractor = LogMelExtractor(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )

        # 1 second of audio
        waveform = torch.randn(2, 1, 22050)  # [batch, channels, samples]
        output = extractor(waveform)

        assert output.dim() == 4  # [batch, 1, n_mels, time]
        assert output.shape[0] == 2
        assert output.shape[1] == 1
        assert output.shape[2] == 128  # n_mels

    def test_mono_input(self):
        extractor = LogMelExtractor(n_mels=64)
        waveform = torch.randn(1, 22050)  # [batch, samples] - no channel dim
        output = extractor(waveform)

        assert output.dim() == 4
        assert output.shape[2] == 64

    def test_from_config(self):
        config = AudioConfig(n_mels=80, n_fft=512)
        extractor = LogMelExtractor.from_config(config)

        assert extractor.n_mels == 80
        assert extractor.n_fft == 512

    def test_output_frames_calculation(self):
        extractor = LogMelExtractor(n_fft=1024, hop_length=256)
        frames = extractor.get_output_frames(22050)
        assert frames > 0


class TestExtractLogmel:
    def test_convenience_function(self):
        config = AudioConfig()
        waveform = torch.randn(1, 22050)

        features = extract_logmel(waveform, config)
        assert features.dim() == 4


class TestFrameConversions:
    def test_time_to_frame(self):
        # At 22050 Hz with hop_length=256
        frame = time_to_frame_idx(1.0, hop_length=256, sample_rate=22050)
        # 1.0s * 22050 / 256 ≈ 86
        assert frame == 86

    def test_frame_to_time(self):
        time = frame_idx_to_time(86, hop_length=256, sample_rate=22050)
        # 86 * 256 / 22050 ≈ 0.998
        assert abs(time - 1.0) < 0.01

    def test_roundtrip(self):
        original_time = 2.5
        hop_length = 256
        sample_rate = 22050

        frame = time_to_frame_idx(original_time, hop_length, sample_rate)
        recovered_time = frame_idx_to_time(frame, hop_length, sample_rate)

        assert abs(recovered_time - original_time) < hop_length / sample_rate


class TestCreateFrames:
    def test_frame_creation(self):
        # [batch, channels, freq, time]
        features = torch.randn(2, 1, 128, 100)
        frames = create_frames(features, frame_size=10, hop_size=5)

        assert frames.dim() == 5  # [batch, num_frames, channels, freq, frame_size]
        assert frames.shape[0] == 2
        assert frames.shape[-1] == 10  # frame_size

    def test_3d_input(self):
        # [batch, freq, time] - no channel dim
        features = torch.randn(2, 128, 100)
        frames = create_frames(features, frame_size=10, hop_size=5)

        assert frames.dim() == 5
