"""Event extraction and post-processing."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Detected event."""

    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def probs_to_events(
    probs: np.ndarray | torch.Tensor,
    threshold: float,
    frame_duration_s: float,
) -> list[Event]:
    """
    Convert frame probabilities to events.

    Args:
        probs: Frame probabilities [num_frames].
        threshold: Detection threshold.
        frame_duration_s: Duration of one frame in seconds.

    Returns:
        List of detected events.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    # Threshold to binary
    binary = probs >= threshold

    # Find event boundaries
    events = []
    in_event = False
    start_frame = 0

    for i, is_pos in enumerate(binary):
        if is_pos and not in_event:
            # Start of event
            in_event = True
            start_frame = i
        elif not is_pos and in_event:
            # End of event
            in_event = False
            start_time = start_frame * frame_duration_s
            end_time = i * frame_duration_s
            confidence = float(probs[start_frame:i].mean())
            events.append(Event(start_time, end_time, confidence))

    # Handle event at end
    if in_event:
        start_time = start_frame * frame_duration_s
        end_time = len(probs) * frame_duration_s
        confidence = float(probs[start_frame:].mean())
        events.append(Event(start_time, end_time, confidence))

    return events


def merge_events(
    events: list[Event],
    merge_gap_s: float,
) -> list[Event]:
    """
    Merge events separated by small gaps.

    Args:
        events: List of events (sorted by start time).
        merge_gap_s: Maximum gap to merge.

    Returns:
        Merged events.
    """
    if len(events) <= 1:
        return events

    merged = [events[0]]

    for event in events[1:]:
        prev = merged[-1]
        gap = event.start_time - prev.end_time

        if gap <= merge_gap_s:
            # Merge: extend previous event
            merged[-1] = Event(
                start_time=prev.start_time,
                end_time=event.end_time,
                confidence=(prev.confidence + event.confidence) / 2,
            )
        else:
            merged.append(event)

    return merged


def filter_events_by_duration(
    events: list[Event],
    min_duration_s: float,
    max_duration_s: Optional[float] = None,
) -> list[Event]:
    """
    Filter events by duration.

    Args:
        events: List of events.
        min_duration_s: Minimum event duration.
        max_duration_s: Maximum event duration (optional).

    Returns:
        Filtered events.
    """
    filtered = []
    for event in events:
        if event.duration < min_duration_s:
            continue
        if max_duration_s is not None and event.duration > max_duration_s:
            continue
        filtered.append(event)

    return filtered


def filter_events_after_time(
    events: list[Event],
    min_start_time: float,
) -> list[Event]:
    """
    Filter events to only those starting after a given time.

    Used to ignore events before end of support set.

    Args:
        events: List of events.
        min_start_time: Minimum start time.

    Returns:
        Filtered events.
    """
    return [e for e in events if e.start_time >= min_start_time]


def postprocess_predictions(
    probs: np.ndarray | torch.Tensor,
    frame_duration_s: float,
    threshold: float = 0.5,
    merge_gap_s: float = 0.1,
    min_event_s: float = 0.06,
    max_event_s: Optional[float] = None,
    min_start_time: Optional[float] = None,
) -> list[Event]:
    """
    Full post-processing pipeline.

    Args:
        probs: Frame probabilities.
        frame_duration_s: Frame duration.
        threshold: Detection threshold.
        merge_gap_s: Gap for merging.
        min_event_s: Minimum event duration.
        max_event_s: Maximum event duration.
        min_start_time: Minimum start time (for ignoring before support end).

    Returns:
        Final list of events.
    """
    # Extract events
    events = probs_to_events(probs, threshold, frame_duration_s)

    # Merge close events
    events = merge_events(events, merge_gap_s)

    # Filter by duration
    events = filter_events_by_duration(events, min_event_s, max_event_s)

    # Filter by start time
    if min_start_time is not None:
        events = filter_events_after_time(events, min_start_time)

    return events
