"""Tests for annotation parsing."""

import tempfile
from pathlib import Path

import pytest

from fsbed.data.annotations import (
    Annotation,
    read_annotation_csv,
    write_annotation_csv,
    read_support_events,
    get_support_end_time,
)


class TestAnnotation:
    def test_duration(self):
        ann = Annotation("test.wav", 1.0, 2.5, "POS")
        assert ann.duration == 1.5

    def test_center(self):
        ann = Annotation("test.wav", 1.0, 3.0, "POS")
        assert ann.center == 2.0


class TestReadAnnotationCSV:
    def test_simple_format(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Audiofilename,Starttime,Endtime\n"
            "audio.wav,1.0,2.0\n"
            "audio.wav,3.0,4.0\n"
        )

        annotations = read_annotation_csv(csv_path)
        assert len(annotations) == 2
        assert annotations[0].starttime == 1.0
        assert annotations[1].endtime == 4.0

    def test_missing_file(self):
        annotations = read_annotation_csv("/nonexistent/path.csv")
        assert annotations == []


class TestWriteAnnotationCSV:
    def test_write_read_roundtrip(self, tmp_path):
        annotations = [
            Annotation("a.wav", 1.0, 2.0, "POS"),
            Annotation("a.wav", 3.5, 4.5, "POS"),
        ]

        csv_path = tmp_path / "output.csv"
        write_annotation_csv(annotations, csv_path)

        # Read back
        loaded = read_annotation_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0].starttime == 1.0
        assert loaded[1].endtime == 4.5


class TestSupportEvents:
    def test_read_support_events(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Audiofilename,Starttime,Endtime,Q\n"
            "audio.wav,5.0,6.0,POS\n"
            "audio.wav,1.0,2.0,POS\n"
            "audio.wav,3.0,4.0,POS\n"
            "audio.wav,7.0,8.0,POS\n"
            "audio.wav,9.0,10.0,POS\n"
            "audio.wav,11.0,12.0,POS\n"
        )

        support = read_support_events(csv_path, n_support=5)
        assert len(support) == 5
        # Should be sorted by start time
        assert support[0].starttime == 1.0
        assert support[4].starttime == 9.0

    def test_get_support_end_time(self):
        annotations = [
            Annotation("a.wav", 1.0, 2.0),
            Annotation("a.wav", 3.0, 5.0),
            Annotation("a.wav", 6.0, 7.5),
        ]
        end_time = get_support_end_time(annotations)
        assert end_time == 7.5

    def test_empty_support(self):
        assert get_support_end_time([]) == 0.0
