"""DCASE annotation parsing and writing utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Single annotation event."""

    audiofilename: str
    starttime: float
    endtime: float
    label: str = "POS"

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        return self.endtime - self.starttime

    @property
    def center(self) -> float:
        """Event center time in seconds."""
        return (self.starttime + self.endtime) / 2


def read_annotation_csv(
    path: Union[str, Path],
    label_filter: Optional[str] = None,
) -> list[Annotation]:
    """
    Read DCASE-style annotation CSV.

    Handles two formats:
    1. Simple: Audiofilename, Starttime, Endtime
    2. Multi-class: Audiofilename, Starttime, Endtime, Class1, Class2, ...
       where each class column has POS/NEG/UNK values.

    Args:
        path: Path to CSV file.
        label_filter: If set, only return annotations with this label.
                     For multi-class, only return rows where this class is POS.

    Returns:
        List of Annotation objects.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Annotation file not found: {path}")
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read CSV {path}: {e}")
        return []

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    required = {"Audiofilename", "Starttime", "Endtime"}
    if not required.issubset(set(df.columns)):
        # Try lowercase
        df.columns = [c.lower() for c in df.columns]
        required_lower = {"audiofilename", "starttime", "endtime"}
        if not required_lower.issubset(set(df.columns)):
            logger.error(f"Missing required columns in {path}")
            return []

    # Standardize to title case
    col_map = {c.lower(): c for c in ["Audiofilename", "Starttime", "Endtime"]}
    df = df.rename(columns=lambda c: col_map.get(c.lower(), c))

    annotations = []

    # Detect format
    meta_cols = {"Audiofilename", "Starttime", "Endtime"}
    class_cols = [c for c in df.columns if c not in meta_cols and c != "Q"]

    if class_cols:
        # Multi-class format
        for _, row in df.iterrows():
            for cls in class_cols:
                if row.get(cls) == "POS":
                    if label_filter is None or cls == label_filter:
                        annotations.append(
                            Annotation(
                                audiofilename=str(row["Audiofilename"]),
                                starttime=float(row["Starttime"]),
                                endtime=float(row["Endtime"]),
                                label=cls,
                            )
                        )
    else:
        # Simple format (assume all are POS)
        for _, row in df.iterrows():
            label = row.get("Q", "POS") if "Q" in df.columns else "POS"
            if label_filter is None or label == label_filter:
                annotations.append(
                    Annotation(
                        audiofilename=str(row["Audiofilename"]),
                        starttime=float(row["Starttime"]),
                        endtime=float(row["Endtime"]),
                        label=label,
                    )
                )

    return annotations


def read_support_events(
    path: Union[str, Path],
    n_support: int = 5,
) -> list[Annotation]:
    """
    Read the first N support events from annotation file.

    Args:
        path: Path to annotation CSV.
        n_support: Number of support events to read.

    Returns:
        List of first N POS annotations, sorted by start time.
    """
    annotations = read_annotation_csv(path, label_filter="POS")

    # Sort by start time
    annotations = sorted(annotations, key=lambda a: a.starttime)

    return annotations[:n_support]


def get_support_end_time(annotations: list[Annotation]) -> float:
    """Get the end time of the last support event."""
    if not annotations:
        return 0.0
    return max(a.endtime for a in annotations)


def write_annotation_csv(
    annotations: list[Annotation],
    path: Union[str, Path],
) -> None:
    """
    Write annotations to DCASE-style CSV.

    Args:
        annotations: List of Annotation objects.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for ann in annotations:
        data.append({
            "Audiofilename": ann.audiofilename,
            "Starttime": ann.starttime,
            "Endtime": ann.endtime,
        })

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(annotations)} annotations to {path}")
