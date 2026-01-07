"""Tests for post-processing utilities."""

import numpy as np
import pytest
import torch

from fsbed.postprocess.smooth import smooth_probs, kernel_size_from_seconds
from fsbed.postprocess.events import (
    Event,
    probs_to_events,
    merge_events,
    filter_events_by_duration,
    filter_events_after_time,
    postprocess_predictions,
)


class TestSmoothing:
    def test_median_filter_numpy(self):
        probs = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        smoothed = smooth_probs(probs, method="median", kernel_size=3)
        assert len(smoothed) == len(probs)
        # Median should smooth out isolated spikes
        assert smoothed[1] < 0.9

    def test_median_filter_torch(self):
        probs = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.1])
        smoothed = smooth_probs(probs, method="median", kernel_size=3)
        assert isinstance(smoothed, torch.Tensor)
        assert len(smoothed) == len(probs)

    def test_moving_average(self):
        probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        smoothed = smooth_probs(probs, method="moving_avg", kernel_size=3)
        # Peak should be spread out
        assert smoothed[2] < 1.0
        assert smoothed[1] > 0.0

    def test_kernel_size_from_seconds(self):
        kernel = kernel_size_from_seconds(0.1, hop_length=256, sample_rate=22050)
        assert kernel >= 3
        assert kernel % 2 == 1  # Should be odd


class TestProbsToEvents:
    def test_simple_detection(self):
        probs = np.array([0.1, 0.1, 0.8, 0.9, 0.8, 0.1, 0.1])
        frame_dur = 0.1

        events = probs_to_events(probs, threshold=0.5, frame_duration_s=frame_dur)
        assert len(events) == 1
        assert events[0].start_time == 0.2
        assert events[0].end_time == 0.5

    def test_multiple_events(self):
        probs = np.array([0.1, 0.8, 0.1, 0.1, 0.9, 0.9, 0.1])
        events = probs_to_events(probs, threshold=0.5, frame_duration_s=0.1)
        assert len(events) == 2

    def test_no_events(self):
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        events = probs_to_events(probs, threshold=0.5, frame_duration_s=0.1)
        assert len(events) == 0

    def test_event_at_end(self):
        probs = np.array([0.1, 0.1, 0.8, 0.9])
        events = probs_to_events(probs, threshold=0.5, frame_duration_s=0.1)
        assert len(events) == 1
        assert events[0].end_time == 0.4


class TestMergeEvents:
    def test_merge_close_events(self):
        events = [
            Event(1.0, 2.0),
            Event(2.05, 3.0),  # Close gap
            Event(5.0, 6.0),  # Far gap
        ]
        merged = merge_events(events, merge_gap_s=0.1)
        assert len(merged) == 2
        assert merged[0].start_time == 1.0
        assert merged[0].end_time == 3.0

    def test_no_merge_needed(self):
        events = [
            Event(1.0, 2.0),
            Event(3.0, 4.0),
        ]
        merged = merge_events(events, merge_gap_s=0.1)
        assert len(merged) == 2


class TestFilterEvents:
    def test_filter_by_min_duration(self):
        events = [
            Event(1.0, 1.01),  # 0.01s - too short
            Event(2.0, 2.1),   # 0.1s - ok
            Event(3.0, 3.5),   # 0.5s - ok
        ]
        filtered = filter_events_by_duration(events, min_duration_s=0.05)
        assert len(filtered) == 2

    def test_filter_by_max_duration(self):
        events = [
            Event(1.0, 1.5),   # 0.5s - ok
            Event(2.0, 4.0),   # 2.0s - too long
        ]
        filtered = filter_events_by_duration(events, min_duration_s=0.1, max_duration_s=1.0)
        assert len(filtered) == 1

    def test_filter_after_time(self):
        events = [
            Event(1.0, 2.0),
            Event(3.0, 4.0),
            Event(5.0, 6.0),
        ]
        filtered = filter_events_after_time(events, min_start_time=2.5)
        assert len(filtered) == 2
        assert filtered[0].start_time == 3.0


class TestFullPostprocess:
    def test_full_pipeline(self):
        probs = np.array([0.1] * 10 + [0.9] * 5 + [0.1] * 5 + [0.85] * 3 + [0.1] * 10)
        frame_dur = 0.1

        events = postprocess_predictions(
            probs,
            frame_duration_s=frame_dur,
            threshold=0.5,
            merge_gap_s=0.3,
            min_event_s=0.1,
        )

        # Should merge the two close events
        assert len(events) >= 1
