"""Cached multi-class training dataset using pre-computed features.

Loads pre-computed logmel features from .npy files for 10-20x faster training.
Uses proper index-based access to support train/val splitting.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..audio.augmentation import SpecAugment
from ..config import AudioConfig
from .dcase_paths import discover_dataset

import pandas as pd

logger = logging.getLogger(__name__)


class CachedMultiClassDataset(Dataset):
    """
    Multi-class dataset using pre-computed logmel features.
    
    Uses proper index-based access to support train/val splitting.
    Each sample is a (audio_path, start_frame, end_frame, class_idx) tuple.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        audio_config: AudioConfig,
        segment_frames: int = 17,  # ~0.2s at hop=256, sr=22050
        augment: bool = True,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        feature_suffix: str = "_logmel.npy",
        sample_indices: Optional[List[int]] = None,  # For subsetting (train/val split)
    ):
        """
        Initialize cached multi-class training dataset.
        
        Args:
            root_dir: Training data root directory.
            audio_config: Audio processing configuration.
            segment_frames: Number of frames per segment.
            augment: Whether to apply SpecAugment.
            freq_mask_param: SpecAugment frequency mask parameter.
            time_mask_param: SpecAugment time mask parameter.
            feature_suffix: Suffix for cached feature files.
            sample_indices: If provided, only use these sample indices (for train/val split).
        """
        self.root_dir = Path(root_dir)
        self.audio_config = audio_config
        self.segment_frames = segment_frames
        self.augment = augment
        self.feature_suffix = feature_suffix
        
        if augment and (freq_mask_param > 0 or time_mask_param > 0):
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
            )
        else:
            self.spec_augment = None
        
        # Cache for loaded features
        self._feature_cache = {}
        
        # Index all data
        self._index_data()
        
        # Apply subset if provided
        if sample_indices is not None:
            self.all_samples = [self.all_samples[i] for i in sample_indices]
    
    def _load_features(self, audio_path: Path) -> np.ndarray:
        """Load cached features for an audio file."""
        key = str(audio_path)
        if key in self._feature_cache:
            return self._feature_cache[key]
        
        # Find corresponding .npy file
        feature_path = audio_path.with_suffix('').parent / f"{audio_path.stem}{self.feature_suffix}"
        
        if not feature_path.exists():
            logger.warning(f"Feature file not found: {feature_path}")
            return None
        
        features = np.load(feature_path)  # [time, n_mels]
        
        # Cache in memory (be careful with memory for large datasets)
        if len(self._feature_cache) < 200:  # Limit cache size
            self._feature_cache[key] = features
        
        return features
    
    def _index_data(self) -> None:
        """Index all training data and extract species classes."""
        self.file_pairs = discover_dataset(self.root_dir)
        
        # Collect all samples per species class
        self.class_samples: dict[str, List[Tuple[Path, int, int]]] = {}
        
        # Frame duration in seconds
        frame_duration = self.audio_config.hop_length / self.audio_config.sample_rate
        
        for audio_path, csv_path in self.file_pairs:
            # Check if features exist
            feature_path = audio_path.with_suffix('').parent / f"{audio_path.stem}{self.feature_suffix}"
            if not feature_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")
                continue
            
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            
            # Find species columns
            meta_cols = {'Audiofilename', 'Starttime', 'Endtime', 'Q', 
                        'audiofilename', 'starttime', 'endtime', 'q'}
            species_cols = [c for c in df.columns if c not in meta_cols]
            
            if not species_cols:
                continue
            
            # Extract samples for each species
            for _, row in df.iterrows():
                try:
                    start_s = float(row.get('Starttime', row.get('starttime', 0)))
                    end_s = float(row.get('Endtime', row.get('endtime', 0)))
                except (ValueError, TypeError):
                    continue
                
                # Convert to frame indices
                start_frame = int(start_s / frame_duration)
                end_frame = int(end_s / frame_duration)
                
                for species in species_cols:
                    label = row.get(species, 'UNK')
                    if label == 'POS':
                        if species not in self.class_samples:
                            self.class_samples[species] = []
                        self.class_samples[species].append((audio_path, start_frame, end_frame))
        
        # Create class to index mapping
        self.classes = sorted(self.class_samples.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Flatten all samples with their class indices
        self.all_samples: List[Tuple[Path, int, int, int]] = []
        for cls, samples in self.class_samples.items():
            cls_idx = self.class_to_idx[cls]
            for audio_path, start_frame, end_frame in samples:
                self.all_samples.append((audio_path, start_frame, end_frame, cls_idx))
        
        # Shuffle samples for better mixing of classes
        random.shuffle(self.all_samples)
        
        logger.info(f"Indexed {len(self.all_samples)} samples across {self.num_classes} classes (cached)")
    
    def __len__(self) -> int:
        """Return actual number of samples (not artificial epoch size)."""
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a training sample by index."""
        # Use the index to get a specific sample (not random!)
        audio_path, start_frame, end_frame, cls_idx = self.all_samples[idx]
        
        # Load cached features
        features = self._load_features(audio_path)
        if features is None:
            # Fallback
            return torch.zeros(1, self.audio_config.n_mels, self.segment_frames), cls_idx
        
        total_frames = features.shape[0]
        
        # Sample position within event (random crop for augmentation during training)
        event_frames = end_frame - start_frame
        if event_frames > self.segment_frames:
            # Random crop within event
            max_offset = event_frames - self.segment_frames
            offset = random.randint(0, max_offset)
            sample_start = start_frame + offset
        else:
            # Center on event
            center = (start_frame + end_frame) // 2
            sample_start = center - self.segment_frames // 2
        
        # Clamp to valid range
        sample_start = max(0, min(sample_start, total_frames - self.segment_frames))
        sample_end = sample_start + self.segment_frames
        
        # Handle edge case where segment is too short
        if sample_end > total_frames:
            sample_end = total_frames
            sample_start = max(0, sample_end - self.segment_frames)
        
        # Extract segment
        segment = features[sample_start:sample_end]  # [time, n_mels]
        
        # Pad if necessary
        if segment.shape[0] < self.segment_frames:
            pad_len = self.segment_frames - segment.shape[0]
            segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='constant')
        
        segment = segment.T  # [n_mels, time]
        segment = torch.from_numpy(segment).float().unsqueeze(0)  # [1, n_mels, time]
        
        # Normalize to mean=0, std=1 (per-sample normalization)
        # This is critical for good gradient flow
        mean = segment.mean()
        std = segment.std() + 1e-8  # Avoid division by zero
        segment = (segment - mean) / std
        
        # Apply augmentation (only during training)
        if self.augment and self.spec_augment is not None:
            segment = self.spec_augment(segment)
        
        return segment, cls_idx
    
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
    
    def create_train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple['CachedMultiClassDataset', 'CachedMultiClassDataset']:
        """
        Create train and validation datasets with proper splitting.
        
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        # Get total samples
        n_samples = len(self.all_samples)
        indices = list(range(n_samples))
        
        # Shuffle with seed for reproducibility
        rng = random.Random(seed)
        rng.shuffle(indices)
        
        # Split
        val_size = int(n_samples * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Create new datasets with subsets
        train_dataset = CachedMultiClassDataset(
            root_dir=self.root_dir,
            audio_config=self.audio_config,
            segment_frames=self.segment_frames,
            augment=True,  # Training uses augmentation
            freq_mask_param=self.spec_augment.freq_mask_param if self.spec_augment else 0,
            time_mask_param=self.spec_augment.time_mask_param if self.spec_augment else 0,
            feature_suffix=self.feature_suffix,
            sample_indices=train_indices,
        )
        
        val_dataset = CachedMultiClassDataset(
            root_dir=self.root_dir,
            audio_config=self.audio_config,
            segment_frames=self.segment_frames,
            augment=False,  # Validation does NOT use augmentation
            freq_mask_param=0,
            time_mask_param=0,
            feature_suffix=self.feature_suffix,
            sample_indices=val_indices,
        )
        
        # Copy class info to subsets
        train_dataset.classes = self.classes
        train_dataset.class_to_idx = self.class_to_idx
        train_dataset.num_classes = self.num_classes
        
        val_dataset.classes = self.classes
        val_dataset.class_to_idx = self.class_to_idx
        val_dataset.num_classes = self.num_classes
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_dataset, val_dataset
