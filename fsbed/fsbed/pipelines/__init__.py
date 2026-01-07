"""Pipelines subpackage."""

from .infer_one import infer_single_file
from .infer_batch import infer_batch

# Lazy import for train to avoid tensorboard dependency when not training
def __getattr__(name):
    if name == "train_backbone":
        from .train import train_backbone
        return train_backbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["train_backbone", "infer_single_file", "infer_batch"]
