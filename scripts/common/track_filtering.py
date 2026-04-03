"""Shared filtering helpers for MOT-style tracking result scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd


TrackingFrame = pd.DataFrame


def load_tracking_results(path: str | Path) -> TrackingFrame:
    """Load MOT-format tracking results through `motmetrics`."""

    try:
        import motmetrics as mm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "motmetrics is required to load tracking results for legacy scripts."
        ) from exc

    results = mm.io.loadtxt(str(path))
    results.reset_index(level=["FrameId", "Id"], inplace=True)
    return results


def filter_short_tracks(results: TrackingFrame, min_frames: int) -> TrackingFrame:
    """Keep only tracks that survive for at least `min_frames` frames."""

    track_lengths = results.groupby("Id")["FrameId"].count().reset_index(name="Count")
    selected_ids = set(track_lengths.loc[track_lengths["Count"] >= min_frames]["Id"])
    return results[results["Id"].isin(selected_ids)]


def filter_bboxes_on_borders(results: TrackingFrame, image_dir: str | Path) -> TrackingFrame:
    """Drop boxes that touch or exceed the image boundary."""

    frame_height, frame_width = read_first_frame_shape(image_dir)
    return results[
        (results["X"] > 0)
        & (results["Y"] > 0)
        & (results["X"] + results["Width"] < frame_width)
        & (results["Y"] + results["Height"] < frame_height)
    ]


def filter_small_bboxes(
    results: TrackingFrame,
    min_width: float = 20,
    min_height: float = 30,
) -> TrackingFrame:
    """Drop boxes smaller than the configured dimensions."""

    return results[(results["Width"] >= min_width) & (results["Height"] >= min_height)]


def summarize_track_results(results: TrackingFrame) -> dict[str, Any]:
    """Return a compact summary of the current filtered result set."""

    track_count = len(results.groupby("Id")) if len(results) else 0
    average_bboxes = round(len(results) / track_count, 4) if track_count else 0.0
    return {
        "rows": len(results),
        "tracks": track_count,
        "avg_bboxes": average_bboxes,
    }


def write_tracking_results(path: str | Path, results: TrackingFrame) -> None:
    """Write a MOT-style result table back to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results.itertuples(index=False):
            handle.write(
                f"{row.FrameId:.0f},{row.Id:.0f},{row.X},{row.Y},{row.Width},{row.Height},"
                f"{row.Confidence},{row.ClassId:.0f},{row.Visibility:.0f},-1\n"
            )


def read_first_frame_shape(image_dir: str | Path) -> tuple[int, int]:
    """Read the first image in a directory and return `(height, width)`."""

    path = Path(image_dir)
    image_paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {path}")
    frame = cv2.imread(str(image_paths[0]))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    height, width = frame.shape[:2]
    return height, width
