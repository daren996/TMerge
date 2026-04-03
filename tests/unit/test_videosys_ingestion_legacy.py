from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from videosys.ingestion import fields
from videosys.ingestion.data import ObjectDetectionResult
from videosys.ingestion.formatting import DetToTrackResult
from videosys.ingestion.io.loaders import MOTDetLoader, MOTGTLoader, MOTResultLoader


@dataclass
class RecordingCollector:
    emitted: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, tables: dict[str, Any] | None = None) -> None:
        self.emitted.append(dict(tables or {}))


def test_mot_det_loader_reads_sequence_folder(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    det_dir.mkdir()
    (det_dir / "det.txt").write_text("1,7,10,20,30,40,0.95\n2,8,1,2,3,4,0.50\n", encoding="utf-8")

    loader = MOTDetLoader(tmp_path)
    loader.collector = RecordingCollector()
    loader.prepare()

    tables = {fields.DATA_FRAME_ID: 1}
    loader.process(tables)

    detections = tables[fields.DATA_OBJECT_DETECTION]
    assert len(detections) == 1
    np.testing.assert_array_equal(detections[0].bbox, np.array([10.0, 20.0, 40.0, 60.0]))
    assert detections[0].confidence == 0.95
    assert loader.collector.emitted[0][fields.DATA_OBJECT_DETECTION] == detections


def test_mot_result_loader_reads_track_file(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("3,12,4,5,6,7,0.75\n", encoding="utf-8")

    loader = MOTResultLoader(result_file)
    loader.collector = RecordingCollector()
    loader.prepare()

    tables = {fields.DATA_FRAME_ID: 3}
    loader.process(tables)

    tracks = tables[fields.DATA_OBJECT_TRACK]
    assert len(tracks) == 1
    assert tracks[0].uid == 12
    np.testing.assert_array_equal(tracks[0].bbox, np.array([4.0, 5.0, 10.0, 12.0]))
    assert tracks[0].payload.confidence == 1.0


def test_mot_gt_loader_uses_ground_truth_field(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("5,99,2,4,6,8,1.0\n", encoding="utf-8")

    loader = MOTGTLoader(gt_file)
    loader.collector = RecordingCollector()
    loader.prepare()

    tables = {fields.DATA_FRAME_ID: 5}
    loader.process(tables)

    assert fields.DATA_OBJECT_TRACK_GT in tables
    assert fields.DATA_OBJECT_TRACK not in tables


def test_det_to_track_result_filters_by_threshold() -> None:
    operator = DetToTrackResult(threshold=0.6)
    operator.collector = RecordingCollector()

    tables = {
        fields.DATA_OBJECT_DETECTION: [
            ObjectDetectionResult(bbox=np.array([1, 2, 3, 4]), label=1, confidence=0.9),
            ObjectDetectionResult(bbox=np.array([5, 6, 7, 8]), label=2, confidence=0.4),
        ]
    }
    operator.process(tables)

    tracks = tables[fields.DATA_OBJECT_TRACK]
    assert len(tracks) == 1
    assert tracks[0].label == 1
    assert tracks[0].confidence == 0.9
    np.testing.assert_array_equal(tracks[0].bbox, np.array([1, 2, 3, 4]))
