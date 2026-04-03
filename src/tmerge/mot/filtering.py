"""Filtering helpers for MOT result files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2

from tmerge.io.mot import MotRow, load_mot_rows, write_mot_rows


def filter_mot_results(
    result_path: str | Path,
    output_path: str | Path,
    *,
    image_dir: str | Path | None = None,
    min_frames: int = 1,
    min_width: float = 0.0,
    min_height: float = 0.0,
    drop_border_boxes: bool = False,
) -> dict[str, float]:
    rows = load_mot_rows(result_path)
    original_count = len(rows)

    if min_frames > 1:
        counts = Counter(row.track_id for row in rows)
        rows = [row for row in rows if counts[row.track_id] >= min_frames]

    if min_width > 0 or min_height > 0:
        rows = [row for row in rows if row.width >= min_width and row.height >= min_height]

    if drop_border_boxes:
        if image_dir is None:
            raise ValueError("image_dir is required when drop_border_boxes=True")
        frame_height, frame_width = _read_first_frame_shape(image_dir)
        rows = [
            row
            for row in rows
            if row.left > 0
            and row.top > 0
            and row.left + row.width < frame_width
            and row.top + row.height < frame_height
        ]

    write_mot_rows(output_path, rows)
    track_ids = {row.track_id for row in rows}
    return {
        "rows_before": float(original_count),
        "rows_after": float(len(rows)),
        "tracks_after": float(len(track_ids)),
    }


def _read_first_frame_shape(image_dir: str | Path) -> tuple[int, int]:
    path = Path(image_dir).expanduser()
    image_paths = sorted(candidate for candidate in path.iterdir() if candidate.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {path}")
    frame = cv2.imread(str(image_paths[0]))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    height, width = frame.shape[:2]
    return height, width

