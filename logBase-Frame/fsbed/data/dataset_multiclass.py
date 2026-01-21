"""Multi-class training dataset for 20-class pretraining.

This dataset extracts species labels from the training set for 
multi-class classification pretraining (as per Liu et al. paper).
"""

import logging
import random
from pathlib import Path
from typing import Optional, Union, List, Tuple

import torch
from torch.utils.data import Dataset

from ..audio.io import load_audio_segment
from ..audio.features import LogMelExtractor
from ..audio.augmentation import SpecAugment
from ..config import AudioConfig
from .dcase_paths import discover_dataset

import pandas as pd

logger = logging.getLogger(__name__)


class MultiClassDataset(Dataset):
    """
    Multi-class dataset for pretraining.
    
    Extracts species labels from training set and creates a 
    multi-class classification task.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        audio_config: AudioConfig,
        segment_duration_s: float = 0.2,
        samples_per_epoch: int = 16000,
        augment: bool = True,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
    ):
        """
        Initialize multi-class training dataset.
        
        Args:
            root_dir: Training data root directory.
            audio_config: Audio processing configuration.
            segment_duration_s: Duration of each segment.
            samples_per_epoch: Number of samples per epoch.
            augment: Whether to apply SpecAugment.
            freq_mask_param: SpecAugment frequency mask parameter.
            time_mask_param: SpecAugment time mask parameter.
        """
        self.root_dir = Path(root_dir)
        self.audio_config = audio_config
        self.segment_duration_s = segment_duration_s
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        
        self.feature_extractor = LogMelExtractor.from_config(audio_config)
        
        if augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
            )
        else:
            self.spec_augment = None
        
        # Index all data
        self._index_data()
    
    def _index_data(self) -> None:
        """Index all training data and extract species classes."""
        self.file_pairs = discover_dataset(self.root_dir)
        
        # Collect all samples per species class
        self.class_samples: dict[str, List[Tuple[Path, float, float]]] = {}
        
        for audio_path, csv_path in self.file_pairs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")
                continue
            
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            
            # Find species columns (not Audiofilename, Starttime, Endtime, Q)
            meta_cols = {'Audiofilename', 'Starttime', 'Endtime', 'Q', 
                        'audiofilename', 'starttime', 'endtime', 'q'}
            species_cols = [c for c in df.columns if c not in meta_cols]
            
            if not species_cols:
                continue
            
            # Extract samples for each species
            for _, row in df.iterrows():
                try:
                    start = float(row.get('Starttime', row.get('starttime', 0)))
                    end = float(row.get('Endtime', row.get('endtime', 0)))
                except (ValueError, TypeError):
                    continue
                
                for species in species_cols:
                    label = row.get(species, 'UNK')
                    if label == 'POS':
                        if species not in self.class_samples:
                            self.class_samples[species] = []
                        self.class_samples[species].append((audio_path, start, end))
        
        # Create class to index mapping
        self.classes = sorted(self.class_samples.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Flatten all samples with their class indices
        self.all_samples: List[Tuple[Path, float, float, int]] = []
        for cls, samples in self.class_samples.items():
            cls_idx = self.class_to_idx[cls]
            for audio_path, start, end in samples:
                self.all_samples.append((audio_path, start, end, cls_idx))
        
        logger.info(f"Indexed {len(self.all_samples)} samples across {self.num_classes} classes")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (features, class_label).
        """
        # Random sample
        audio_path, start, end, cls_idx = random.choice(self.all_samples)
        
        # Adjust sampling position
        event_dur = end - start
        if event_dur > self.segment_duration_s:
            # Random crop within event
            max_offset = event_dur - self.segment_duration_s
            offset = random.random() * max_offset
            start_sample = start + offset
        else:
            # Center on event
            center = (start + end) / 2
            start_sample = center - self.segment_duration_s / 2
            start_sample = max(0, start_sample)
        
        end_sample = start_sample + self.segment_duration_s
        
        # Load and extract features
        features = self._load_segment_features(audio_path, start_sample, end_sample)
        
        # Apply augmentation
        if self.augment and self.spec_augment is not None:
            features = self.spec_augment(features)
        
        return features, cls_idx
    
    def _load_segment_features(
        self,
        audio_path: Path,
        start_s: float,
        end_s: float,
    ) -> torch.Tensor:
        """Load audio segment and extract features."""
        waveform, sr = load_audio_segment(
            audio_path,
            start_s,
            end_s,
            target_sr=self.audio_config.sample_rate,
            mono=True,
        )
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(waveform.unsqueeze(0))
        
        # Return [1, n_mels, time]
        return features.squeeze(0)
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced training."""
        counts = torch.zeros(self.num_classes)
        for _, _, _, cls_idx in self.all_samples:
            counts[cls_idx] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1)
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def get_num_classes(self) -> int:
        """Get number of species classes."""
        return self.num_classes
