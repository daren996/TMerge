from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.common.track_filtering import (
    filter_bboxes_on_borders,
    filter_short_tracks,
    filter_small_bboxes,
    read_first_frame_shape,
    summarize_track_results,
    write_tracking_results,
)


def _build_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"FrameId": 1, "Id": 1, "X": 10, "Y": 10, "Width": 30, "Height": 40, "Confidence": 0.9, "ClassId": 1, "Visibility": 1},
            {"FrameId": 2, "Id": 1, "X": 12, "Y": 11, "Width": 30, "Height": 40, "Confidence": 0.8, "ClassId": 1, "Visibility": 1},
            {"FrameId": 1, "Id": 2, "X": 0, "Y": 5, "Width": 10, "Height": 20, "Confidence": 0.7, "ClassId": 1, "Visibility": 1},
        ]
    )


def test_filter_short_tracks_keeps_only_long_enough_tracks() -> None:
    results = _build_results_frame()

    filtered = filter_short_tracks(results, min_frames=2)

    assert set(filtered["Id"]) == {1}
    assert len(filtered) == 2


def test_filter_bboxes_on_borders_uses_first_image_shape(tmp_path: Path) -> None:
    image = 255 * np.ones((100, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "000001.png"), image)
    results = _build_results_frame()

    filtered = filter_bboxes_on_borders(results, image_dir=tmp_path)

    assert len(filtered) == 2
    assert set(filtered["Id"]) == {1}


def test_filter_small_bboxes_drops_small_rows() -> None:
    results = _build_results_frame()

    filtered = filter_small_bboxes(results, min_width=20, min_height=30)

    assert len(filtered) == 2
    assert set(filtered["Id"]) == {1}


def test_summarize_and_write_tracking_results(tmp_path: Path) -> None:
    results = _build_results_frame().iloc[:2]

    summary = summarize_track_results(results)
    output_path = tmp_path / "out" / "tracks.txt"
    write_tracking_results(output_path, results)

    assert summary == {"rows": 2, "tracks": 1, "avg_bboxes": 2.0}
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "1,1,10,10,30,40,0.9,1,1,-1",
        "2,1,12,11,30,40,0.8,1,1,-1",
    ]


def test_read_first_frame_shape_returns_height_width(tmp_path: Path) -> None:
    image = 255 * np.ones((80, 60, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "frame.jpg"), image)

    assert read_first_frame_shape(tmp_path) == (80, 60)
