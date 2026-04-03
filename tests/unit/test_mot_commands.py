from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tmerge.evaluation import mot as mot_eval
from tmerge.io.mot import load_mot_tracks
from tmerge.visualization.mot import export_mot_video


def test_load_mot_tracks_parses_rows(tmp_path: Path) -> None:
    result_path = tmp_path / "result.txt"
    result_path.write_text("1,5,10,20,30,40,0.9,-1,-1,-1\n", encoding="utf-8")

    tracks = load_mot_tracks(result_path)

    assert 1 in tracks
    assert tracks[1][0].uid == 5
    assert tracks[1][0].bbox.tolist() == [10.0, 20.0, 40.0, 60.0]


def test_export_mot_video_writes_video(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "000001.png"), frame)
    cv2.imwrite(str(input_dir / "000002.png"), frame)

    result_path = tmp_path / "result.txt"
    result_path.write_text(
        "1,1,1,1,4,4,0.9,-1,-1,-1\n2,1,2,2,4,4,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.mp4"

    frames_written = export_mot_video(input_dir, result_path, output_path, fps=10)

    assert frames_written == 2
    assert output_path.exists()


def test_evaluate_mot_results_uses_motmetrics(monkeypatch, tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt.txt"
    result = tmp_path / "result.txt"
    ground_truth.write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")
    result.write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")

    class FakeAccumulator:
        def __init__(self):
            self.updates = []

        def update(self, oids, hids, dists, frameid):
            self.updates.append((oids.tolist(), hids.tolist(), frameid))

    class FakeMetricsHandler:
        formatters = {}

        def compute(self, accumulator, metrics, name):
            return {"acc": 1}

    fake_mm = type(
        "FakeMM",
        (),
        {
            "mot": type("FakeMot", (), {"MOTAccumulator": FakeAccumulator}),
            "distances": type("FakeDist", (), {"iou_matrix": lambda *_args: np.empty((1, 1)), "norm2squared_matrix": lambda *_args: np.empty((1, 1))}),
            "metrics": type(
                "FakeMetrics",
                (),
                {
                    "create": staticmethod(lambda: FakeMetricsHandler()),
                    "motchallenge_metrics": ["mota"],
                },
            ),
            "io": type("FakeIo", (), {"render_summary": staticmethod(lambda *_args, **_kwargs: "summary"), "motchallenge_metric_names": {}}),
        },
    )

    monkeypatch.setattr(mot_eval, "_require_motmetrics", lambda: fake_mm)

    summary = mot_eval.evaluate_mot_results(ground_truth, result)

    assert summary == "summary"

