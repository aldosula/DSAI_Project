"""Configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PathsConfig:
    """Paths configuration."""

    train_dir: str = "/data/msc-proj/Training_Set"
    val_dir: str = "/data/msc-proj/Validation_Set_DSAI_2025_2026"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 11025.0
    log_offset: float = 1e-6

    def __post_init__(self):
        """Ensure proper types after initialization."""
        # Handle case where log_offset might be a string (from YAML)
        if isinstance(self.log_offset, str):
            self.log_offset = float(self.log_offset)

    @property
    def frame_duration_s(self) -> float:
        """Duration of one frame in seconds."""
        return self.hop_length / self.sample_rate

    def time_to_frame(self, time_s: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_s * self.sample_rate / self.hop_length)

    def frame_to_time(self, frame: int) -> float:
        """Convert frame index to time in seconds."""
        return frame * self.hop_length / self.sample_rate


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    embed_dim: int = 256
    num_blocks: int = 4
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 256])
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    warmup_epochs: int = 5
    early_stop_patience: int = 15
    num_workers: int = 4
    class_balance: bool = True
    
    # Augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    feature_type: str = "logmel"
    freq_mask_param: int = 27
    time_mask_param: int = 100


@dataclass
class FewshotConfig:
    """Few-shot settings."""

    n_support: int = 5
    support_context_s: float = 0.025


@dataclass
class AdaptConfig:
    """Per-file adaptation configuration."""

    enabled: bool = True
    iterations: int = 3
    finetune_lr: float = 0.0001
    finetune_layers: list[str] = field(default_factory=lambda: ["head"])
    hard_neg_k: int = 1500
    pseudo_threshold: float = 0.95
    pseudo_max_per_iter: int = 50
    pseudo_min_spacing_s: float = 0.5


@dataclass
class InferenceConfig:
    """Inference configuration."""

    chunk_duration_s: float = 30.0
    chunk_overlap_s: float = 2.0
    frame_hop_s: float = 0.01


@dataclass
class PostprocessConfig:
    """Post-processing configuration."""

    smooth_method: str = "median"
    smooth_kernel_s: float = 0.1
    threshold: float = 0.5
    adaptive_threshold: bool = False
    min_event_s: float = 0.06
    max_event_s: Optional[float] = None
    merge_gap_s: float = 0.1
    output_after_support_end: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    onset_tolerance_s: float = 0.2
    offset_tolerance_s: float = 0.2
    min_iou: float = 0.0


@dataclass
class Config:
    """Master configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fewshot: FewshotConfig = field(default_factory=FewshotConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    deterministic: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return Config(
            paths=PathsConfig(**data.get("paths", {})),
            audio=AudioConfig(**data.get("audio", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            fewshot=FewshotConfig(**data.get("fewshot", {})),
            adapt=AdaptConfig(**data.get("adapt", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            postprocess=PostprocessConfig(**data.get("postprocess", {})),
            eval=EvalConfig(**data.get("eval", {})),
            seed=data.get("seed", 42),
            deterministic=data.get("deterministic", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _to_dict(self)
