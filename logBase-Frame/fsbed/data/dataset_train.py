"""Training dataset for offline backbone training."""

import logging
import random
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from ..audio.io import load_audio_segment
from ..audio.features import LogMelExtractor
from ..config import AudioConfig
from .annotations import read_annotation_csv, Annotation
from .dcase_paths import discover_dataset

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """
    Dataset for training the embedding backbone.

    Samples positive and negative segments from training files.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        audio_config: AudioConfig,
        segment_duration_s: float = 0.2,
        samples_per_epoch: int = 10000,
        pos_neg_ratio: float = 0.5,
        ignore_unk: bool = True,
        cache_features: bool = False,
    ):
        """
        Initialize training dataset.

        Args:
            root_dir: Training data root directory.
            audio_config: Audio processing configuration.
            segment_duration_s: Duration of each segment.
            samples_per_epoch: Number of samples per epoch.
            pos_neg_ratio: Ratio of positive samples (0.5 = balanced).
            ignore_unk: If True, ignore UNK labels.
            cache_features: If True, cache features to disk.
        """
        self.root_dir = Path(root_dir)
        self.audio_config = audio_config
        self.segment_duration_s = segment_duration_s
        self.samples_per_epoch = samples_per_epoch
        self.pos_neg_ratio = pos_neg_ratio
        self.ignore_unk = ignore_unk

        self.feature_extractor = LogMelExtractor.from_config(audio_config)

        # Index all files and annotations
        self._index_data()

    def _index_data(self) -> None:
        """Index all training data."""
        self.file_pairs = discover_dataset(self.root_dir)

        # Collect all positive events per class
        self.class_positives: dict[str, list[tuple[Path, Annotation]]] = {}
        # Collect negative regions (gaps between positives)
        self.negatives: list[tuple[Path, float, float]] = []

        for audio_path, csv_path in self.file_pairs:
            annotations = read_annotation_csv(csv_path)

            # Sort by start time
            annotations = sorted(annotations, key=lambda a: a.starttime)

            for ann in annotations:
                if ann.label not in self.class_positives:
                    self.class_positives[ann.label] = []
                self.class_positives[ann.label].append((audio_path, ann))

            # Extract negative regions (gaps between events)
            if annotations:
                # Before first event
                if annotations[0].starttime > self.segment_duration_s:
                    self.negatives.append(
                        (audio_path, 0.0, annotations[0].starttime - 0.05)
                    )

                # Between events
                for i in range(len(annotations) - 1):
                    gap_start = annotations[i].endtime + 0.05
                    gap_end = annotations[i + 1].starttime - 0.05
                    if gap_end - gap_start >= self.segment_duration_s:
                        self.negatives.append((audio_path, gap_start, gap_end))

        # Flatten positives for sampling
        self.all_positives: list[tuple[Path, Annotation]] = []
        for cls, items in self.class_positives.items():
            self.all_positives.extend(items)

        logger.info(
            f"Indexed {len(self.all_positives)} positive events, "
            f"{len(self.negatives)} negative regions, "
            f"{len(self.class_positives)} classes"
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get a training sample.

        Returns:
            Tuple of (features, label) where label is 0=NEG, 1=POS.
        """
        # Decide if positive or negative
        is_positive = random.random() < self.pos_neg_ratio

        if is_positive and self.all_positives:
            return self._get_positive_sample()
        else:
            return self._get_negative_sample()

    def _get_positive_sample(self) -> tuple[torch.Tensor, int]:
        """Sample a positive event."""
        audio_path, ann = random.choice(self.all_positives)

        # Sample within or around the event
        event_dur = ann.duration
        if event_dur > self.segment_duration_s:
            # Random crop within event
            max_offset = event_dur - self.segment_duration_s
            offset = random.random() * max_offset
            start = ann.starttime + offset
        else:
            # Center on event
            start = ann.center - self.segment_duration_s / 2
            start = max(0, start)

        end = start + self.segment_duration_s

        # Load and extract features
        features = self._load_segment_features(audio_path, start, end)
        return features, 1

    def _get_negative_sample(self) -> tuple[torch.Tensor, int]:
        """Sample a negative segment."""
        if not self.negatives:
            # Fallback: return zeros
            return torch.zeros(1, self.audio_config.n_mels, 10), 0

        audio_path, neg_start, neg_end = random.choice(self.negatives)

        # Random position within negative region
        max_start = neg_end - self.segment_duration_s
        if max_start <= neg_start:
            start = neg_start
        else:
            start = neg_start + random.random() * (max_start - neg_start)

        end = start + self.segment_duration_s

        features = self._load_segment_features(audio_path, start, end)
        return features, 0

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
        n_pos = len(self.all_positives)
        n_neg = len(self.negatives) * 10  # Estimate negative samples
        total = n_pos + n_neg

        weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
        weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0

        return torch.tensor([weight_neg, weight_pos])
