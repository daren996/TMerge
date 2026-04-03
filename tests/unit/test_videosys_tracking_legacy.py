from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from videosys.ingestion import fields
from videosys.ingestion.base import Context
from videosys.ingestion.data import ImageFolderMeta, ObjectDetectionResult, ObjectTrackingResult
from videosys.ingestion.io.tracking import MOTResultSink, TrackResultFolderSink
from videosys.ingestion.observer.reporter import ProgressReporter


@dataclass
class RecordingCollector:
    emitted: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, tables: dict[str, Any] | None = None) -> None:
        self.emitted.append(dict(tables or {}))


def _build_track() -> ObjectTrackingResult:
    payload = ObjectDetectionResult(
        bbox=np.array([10.0, 20.0, 30.0, 50.0]),
        label=3,
        confidence=0.75,
    )
    return ObjectTrackingResult(
        uid=7,
        label=3,
        bbox=np.array([10.0, 20.0, 30.0, 50.0]),
        confidence=0.9,
        payload=payload,
    )


def test_track_result_folder_sink_writes_one_file_per_frame(tmp_path: Path) -> None:
    sink = TrackResultFolderSink(tmp_path)
    sink.collector = RecordingCollector()
    sink.prepare()

    tables = {fields.DATA_FRAME_ID: 1, fields.DATA_OBJECT_TRACK: [_build_track()]}
    sink.process(tables)

    output = (tmp_path / "1.txt").read_text(encoding="utf-8").strip()
    assert output == "7;[10. 20. 30. 50.];3;0.75"


def test_mot_result_sink_writes_mot_rows(tmp_path: Path) -> None:
    sink = MOTResultSink(tmp_path / "mot.txt", add_type_column=True)
    sink.collector = RecordingCollector()
    sink.prepare()

    tables = {fields.DATA_FRAME_ID: 4, fields.DATA_OBJECT_TRACK: [_build_track()]}
    sink.process(tables)

    assert (tmp_path / "mot.txt").read_text(encoding="utf-8").strip() == (
        "4,7,10.00,20.00,20.00,30.00,0.9,3,-1,-1,-1"
    )


def test_progress_reporter_writes_report_file_and_emits_tables(
    tmp_path: Path,
    capsys: Any,
) -> None:
    report_path = tmp_path / "progress.log"
    reporter = ProgressReporter(report_interval=1, save_file=report_path)
    reporter.collector = RecordingCollector()
    reporter.context = Context()
    reporter.context.put(fields.META_IMAGE, ImageFolderMeta(total=2, path=str(tmp_path)))

    reporter.prepare()
    reporter.process({fields.DATA_FRAME_ID: 1})
    reporter.cleanup()

    report_text = report_path.read_text(encoding="utf-8")
    captured = capsys.readouterr().out
    assert "begin to process" in report_text
    assert "processing: 1/2" in report_text
    assert "done. time elapsed" in report_text
    assert "begin to process" in captured
    assert reporter.collector.emitted == [{fields.DATA_FRAME_ID: 1}]
