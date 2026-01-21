"""Feature extraction utilities."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchaudio

from ..config import AudioConfig

logger = logging.getLogger(__name__)


class LogMelExtractor(nn.Module):
    """
    Log-Mel spectrogram extractor.

    Converts raw audio waveforms to log-mel spectrograms.
    Output shape: [batch, 1, n_mels, time_frames]
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 128,
        fmin: float = 50.0,
        fmax: Optional[float] = None,
        log_offset: float = 1e-6,
    ):
        """
        Initialize LogMelExtractor.

        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Hop between frames.
            win_length: Window length.
            n_mels: Number of mel bins.
            fmin: Minimum frequency.
            fmax: Maximum frequency (default: sample_rate / 2).
            log_offset: Offset for log scaling to avoid log(0).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        self.log_offset = log_offset

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            power=2.0,
        )

    @classmethod
    def from_config(cls, config: AudioConfig) -> "LogMelExtractor":
        """Create extractor from AudioConfig."""
        return cls(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            log_offset=config.log_offset,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram from waveform.

        Args:
            waveform: Audio tensor [batch, channels, samples] or [batch, samples].

        Returns:
            Log-mel spectrogram [batch, 1, n_mels, time_frames].
        """
        # Ensure 3D: [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Take first channel if multi-channel
        if waveform.shape[1] > 1:
            waveform = waveform[:, :1, :]

        # Squeeze channel dim for mel_spec
        waveform = waveform.squeeze(1)  # [batch, samples]

        # Compute mel spectrogram
        mel = self.mel_spec(waveform)  # [batch, n_mels, time]

        # Apply log scaling
        log_mel = torch.log(mel + self.log_offset)

        # Add channel dimension
        log_mel = log_mel.unsqueeze(1)  # [batch, 1, n_mels, time]

        return log_mel

    def get_output_frames(self, num_samples: int) -> int:
        """Calculate number of output frames for given input samples."""
        return (num_samples - self.n_fft) // self.hop_length + 1


def extract_logmel(
    waveform: torch.Tensor,
    config: AudioConfig,
) -> torch.Tensor:
    """
    Convenience function to extract log-mel features.

    Args:
        waveform: Audio tensor [channels, samples] or [batch, channels, samples].
        config: Audio configuration.

    Returns:
        Log-mel spectrogram.
    """
    extractor = LogMelExtractor.from_config(config)

    # Ensure batch dimension
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        features = extractor(waveform)

    return features


class PCENExtractor(nn.Module):
    """
    Per-Channel Energy Normalization (PCEN) extractor.
    
    PCEN(t, f) = (E(t, f) / (M(t, f) + eps)) ^ alpha + delta
    M(t, f) = (1 - s) * M(t-1, f) + s * E(t, f)
    
 
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 128,
        fmin: float = 50.0,
        fmax: Optional[float] = None,
        # PCEN parameters
        time_constant_s: float = 0.06,  #
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        # Base Mel Spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,  
        )
        
        self.log_mel = LogMelExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        
        # PCEN params
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps
        
        # Calculate smoothing coefficient s
        # s = sqrt(1 + (T / hop_time)^2) - T / hop_time ? No.
        # Commonly s = (sqrt(1 + 4*T^2) - 1) / (2*T^2) ?
        # Simple IIR: s = ???
        # Librosa uses b = [s], a = [1, s-1]
        
        import numpy as np
        
        hop_s = hop_length / sample_rate
        # Standard RC low-pass filter coefficient
        self.s = 1 - np.exp(-hop_s / time_constant_s)
        
    @classmethod
    def from_config(cls, config: AudioConfig) -> "PCENExtractor":
        return cls(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            # Add config params for PCEN if they exist, else default
            time_constant_s=getattr(config, 'pcen_time_constant', 0.06),
            alpha=getattr(config, 'pcen_alpha', 0.98),
            delta=getattr(config, 'pcen_delta', 2.0),
            r=getattr(config, 'pcen_r', 0.5),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute PCEN features [batch, 1, n_mels, time]."""
        # Ensure 3D: [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            
        # First channel only
        if waveform.shape[1] > 1:
            waveform = waveform[:, :1, :]
            
        # Mel spectrogram (Magnitude)
        mel = self.mel_spec(waveform.squeeze(1))  # [batch, n_mels, time]
        
        # PCEN
        
        device = mel.device
        dtype = mel.dtype
        
        # Apply recurrence (single pole IIR low-pass)
        
        batch_size, n_mels, time_steps = mel.shape
        M = torch.zeros_like(mel)
        m_prev = torch.zeros(batch_size, n_mels, device=device, dtype=dtype)
        
        s = self.s
        
        # Parallelizing across freq and batch
        # Iterate over time
        mel_t = mel.permute(2, 0, 1) # [time, batch, n_mels]
        M_t_list = []
        
        for t in range(time_steps):
            E_t = mel_t[t]
            current_m = (1 - s) * m_prev + s * E_t
            M_t_list.append(current_m)
            m_prev = current_m
            
        M = torch.stack(M_t_list).permute(1, 2, 0) # [batch, n_mels, time]
        
        smooth = (M + self.eps).pow(self.r)
        pcen = (mel / smooth + self.delta).pow(self.alpha) - self.delta**self.alpha
        
        return pcen.unsqueeze(1)
