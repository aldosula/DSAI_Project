import torch
import torchaudio
import numpy as np

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        win_length=512,
        hop_length=256,
        n_mels=128,
        f_min=50,
        f_max=11000,
        use_pcen=False
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_pcen = use_pcen
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0 if use_pcen else 2.0
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def pcen(self, x, eps=1e-6, alpha=0.98, delta=2.0, r=0.5):
        # x shape: (batch, n_mels, time)
        # Simple PCEN implementation
        # M(t, f) = (E(t, f) / (eps + M_smooth(t, f))**alpha + delta)**r - delta**r
        # We'll use a simplified version or rely on external lib if needed, 
        # but for now let's implement a basic version or stick to log-mel if PCEN is complex to vectorize efficiently in pure torch without state.
        # Torchaudio doesn't have PCEN built-in transform yet in older versions, but let's check.
        # Actually, let's stick to Log-Mel for the baseline and add PCEN if needed or manually implement.
        # For now, we will just use Log-Mel as default.
        return x

    def forward(self, waveform):
        # waveform: (batch, samples) or (samples,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel_spec = self.mel_spectrogram(waveform)
        
        if self.use_pcen:
            # Placeholder for PCEN, falling back to Log-Mel for now to ensure stability first
            # Real PCEN requires temporal smoothing which is stateful or requires unfolding.
            # We'll stick to Log-Mel for the first iteration.
            features = self.amplitude_to_db(mel_spec) 
        else:
            features = self.amplitude_to_db(mel_spec)
            
        return features
