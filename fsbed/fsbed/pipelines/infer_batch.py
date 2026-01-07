"""Batch inference pipeline."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..data.dcase_paths import discover_dataset
from ..data.annotations import write_annotation_csv, Annotation
from ..postprocess.events import Event
from .infer_one import infer_single_file

logger = logging.getLogger(__name__)


def infer_batch(
    input_dir: Path,
    output_dir: Path,
    backbone: nn.Module,
    config: Config,
    device: torch.device,
) -> dict[str, list[Event]]:
    """
    Run inference on all files in a directory.

    Args:
        input_dir: Input directory with audio and annotation files.
        output_dir: Output directory for predictions.
        backbone: Trained embedding backbone.
        config: Configuration.
        device: Compute device.

    Returns:
        Dictionary mapping audio filenames to detected events.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    file_pairs = discover_dataset(input_dir)
    logger.info(f"Found {len(file_pairs)} files to process")

    all_results = {}

    for audio_path, annotation_path in tqdm(file_pairs, desc="Processing files"):
        try:
            events = infer_single_file(
                audio_path,
                annotation_path,
                backbone,
                config,
                device,
            )

            # Convert to annotations
            annotations = [
                Annotation(
                    audiofilename=audio_path.name,
                    starttime=e.start_time,
                    endtime=e.end_time,
                    label="POS",
                )
                for e in events
            ]

            # Save predictions
            output_csv = output_dir / f"{audio_path.stem}_predictions.csv"
            write_annotation_csv(annotations, output_csv)

            all_results[audio_path.name] = events

        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {e}")
            # Clear GPU state to prevent cascading failures
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            continue
        
        # Clear GPU cache between files to free memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"Processed {len(all_results)} files")
    return all_results


def load_model_for_inference(
    checkpoint_path: Path,
    config: Config,
    device: torch.device,
) -> nn.Module:
    """
    Load trained model for inference.

    Args:
        checkpoint_path: Path to checkpoint.
        config: Configuration.
        device: Compute device.

    Returns:
        Loaded backbone model.
    """
    from ..models.resnet_embed import ResNetEmbedding

    backbone = ResNetEmbedding.from_config(config.model, n_mels=config.audio.n_mels)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle full model checkpoint (backbone + head)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Extract backbone weights if full model
    backbone_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_dict[k.replace("backbone.", "")] = v
        elif not k.startswith("head."):
            backbone_dict[k] = v

    if backbone_dict:
        backbone.load_state_dict(backbone_dict)
    else:
        backbone.load_state_dict(state_dict)

    backbone = backbone.to(device)
    backbone.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return backbone
