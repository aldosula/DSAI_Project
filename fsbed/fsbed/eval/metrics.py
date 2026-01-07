"""Event-based evaluation metrics."""

import logging
from dataclasses import dataclass
from typing import Optional

from ..postprocess.events import Event
from ..data.annotations import Annotation

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of event matching."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def compute_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    Compute Intersection over Union of two events.

    Args:
        pred_start, pred_end: Predicted event boundaries.
        gt_start, gt_end: Ground truth event boundaries.

    Returns:
        IoU value in [0, 1].
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def match_events_iou(
    predictions: list[Event],
    ground_truth: list[Annotation],
    min_iou: float = 0.3,
) -> MatchResult:
    """
    Match predictions to ground truth using IoU.

    Args:
        predictions: Predicted events.
        ground_truth: Ground truth annotations.
        min_iou: Minimum IoU for a match.

    Returns:
        MatchResult with TP/FP/FN counts.
    """
    matched_gt = set()
    matched_pred = set()

    # Sort by IoU to prioritize best matches
    matches = []
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            iou = compute_iou(pred.start_time, pred.end_time, gt.starttime, gt.endtime)
            if iou >= min_iou:
                matches.append((iou, i, j))

    # Greedy matching: highest IoU first
    matches.sort(reverse=True)
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)

    return MatchResult(tp, fp, fn)


def match_events_tolerance(
    predictions: list[Event],
    ground_truth: list[Annotation],
    onset_tolerance_s: float = 0.2,
    offset_tolerance_s: float = 0.2,
) -> MatchResult:
    """
    Match predictions to ground truth using onset/offset tolerance.

    Args:
        predictions: Predicted events.
        ground_truth: Ground truth annotations.
        onset_tolerance_s: Maximum onset time difference.
        offset_tolerance_s: Maximum offset time difference.

    Returns:
        MatchResult with TP/FP/FN counts.
    """
    matched_gt = set()
    matched_pred = set()

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue

            onset_diff = abs(pred.start_time - gt.starttime)
            offset_diff = abs(pred.end_time - gt.endtime)

            if onset_diff <= onset_tolerance_s and offset_diff <= offset_tolerance_s:
                matched_pred.add(i)
                matched_gt.add(j)
                break

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)

    return MatchResult(tp, fp, fn)


def match_events(
    predictions: list[Event],
    ground_truth: list[Annotation],
    onset_tolerance_s: float = 0.2,
    offset_tolerance_s: float = 0.2,
    min_iou: float = 0.0,
) -> MatchResult:
    """
    Match predictions to ground truth.

    Uses IoU matching if min_iou > 0, otherwise uses tolerance matching.

    Args:
        predictions: Predicted events.
        ground_truth: Ground truth annotations.
        onset_tolerance_s: Onset tolerance for tolerance matching.
        offset_tolerance_s: Offset tolerance for tolerance matching.
        min_iou: Minimum IoU for IoU matching (0 = use tolerance).

    Returns:
        MatchResult with metrics.
    """
    if min_iou > 0:
        return match_events_iou(predictions, ground_truth, min_iou)
    return match_events_tolerance(predictions, ground_truth, onset_tolerance_s, offset_tolerance_s)


def event_based_f1(
    predictions: list[Event],
    ground_truth: list[Annotation],
    onset_tolerance_s: float = 0.2,
    offset_tolerance_s: float = 0.2,
    min_iou: float = 0.0,
) -> tuple[float, float, float]:
    """
    Compute event-based precision, recall, and F1.

    Args:
        predictions: Predicted events.
        ground_truth: Ground truth annotations.
        onset_tolerance_s: Onset tolerance.
        offset_tolerance_s: Offset tolerance.
        min_iou: Minimum IoU (0 = use tolerance).

    Returns:
        Tuple of (precision, recall, f1).
    """
    result = match_events(
        predictions,
        ground_truth,
        onset_tolerance_s,
        offset_tolerance_s,
        min_iou,
    )
    return result.precision, result.recall, result.f1
