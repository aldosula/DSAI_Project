"""Dataset path discovery utilities."""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def discover_dataset(
    root_dir: Union[str, Path],
    audio_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> list[tuple[Path, Path]]:
    """
    Discover audio-annotation pairs in a dataset directory.

    Assumes each audio file has a corresponding .csv with same basename.

    Args:
        root_dir: Dataset root directory.
        audio_extensions: Audio file extensions to search for.

    Returns:
        List of (audio_path, annotation_path) tuples.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    pairs = []

    # Search recursively for audio files
    for ext in audio_extensions:
        for audio_path in root.rglob(f"*{ext}"):
            # Look for corresponding CSV
            csv_path = audio_path.with_suffix(".csv")
            if csv_path.exists():
                pairs.append((audio_path, csv_path))

    logger.info(f"Found {len(pairs)} audio-annotation pairs in {root}")
    return pairs


def get_audio_annotation_pairs(
    root_dir: Union[str, Path],
) -> list[tuple[Path, Path]]:
    """
    Get all audio-annotation pairs from a directory.

    Args:
        root_dir: Dataset root directory.

    Returns:
        List of (audio_path, annotation_path) tuples.
    """
    return discover_dataset(root_dir)


def get_class_directories(root_dir: Union[str, Path]) -> list[Path]:
    """
    Get class directories in training set.

    Training set is organized as:
    Training_Set/
        BV/
            audio1.wav
            audio1.csv
        HT/
            ...

    Args:
        root_dir: Training set root.

    Returns:
        List of class directory paths.
    """
    root = Path(root_dir)
    return [d for d in root.iterdir() if d.is_dir()]


def get_audio_files_for_class(
    class_dir: Union[str, Path],
    audio_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> list[Path]:
    """
    Get audio files in a class directory.

    Args:
        class_dir: Class directory path.
        audio_extensions: Audio file extensions.

    Returns:
        List of audio file paths.
    """
    class_path = Path(class_dir)
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(class_path.glob(f"*{ext}"))
    return sorted(audio_files)
