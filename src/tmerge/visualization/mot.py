"""Rendering helpers for MOT tracks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tmerge.io.mot import load_mot_tracks
from tmerge.models.types import TrackingResult


def render_track_frame(
    frame: np.ndarray,
    tracks: list[TrackingResult],
    *,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        color = _track_color(track.uid)
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=thickness)
        label = f"{track.uid}/{track.label}\n{track.confidence:.2f}" if track.label >= 0 else f"{track.uid}\n{track.confidence:.2f}"
        _draw_multiline_label(
            output,
            label,
            origin=(x1, max(y1, 0)),
            color=color,
            font_scale=font_scale,
            thickness=max(1, thickness - 1),
        )
    return output


def visualize_mot_results(
    data_path: str | Path,
    result_path: str | Path,
    *,
    start_frame: int = 1,
    thickness: int = 2,
    font_scale: float = 0.6,
    wait_ms: int = 100,
    window_name: str = "tmerge-mot",
) -> int:
    tracks_by_frame = load_mot_tracks(result_path)
    processed = 0
    try:
        for frame_id, frame in iter_frames(data_path):
            if frame_id < start_frame:
                continue
            rendered = render_track_frame(frame, tracks_by_frame.get(frame_id, []), thickness=thickness, font_scale=font_scale)
            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(wait_ms)
            processed += 1
            if key in {27, ord("q")}:
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(window_name)
    return processed


def export_mot_video(
    data_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
    *,
    fps: float | None = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> int:
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tracks_by_frame = load_mot_tracks(result_path)

    writer = None
    frames_written = 0
    source_fps = None
    try:
        for frame_id, frame in iter_frames(data_path):
            rendered = render_track_frame(frame, tracks_by_frame.get(frame_id, []), thickness=thickness, font_scale=font_scale)
            if writer is None:
                frame_fps = fps or source_fps or 30.0
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_file),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    frame_fps,
                    (width, height),
                )
            writer.write(rendered)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    return frames_written


def iter_frames(data_path: str | Path):
    source_path = Path(data_path).expanduser()
    if source_path.is_dir():
        images = sorted(source_path.iterdir(), key=_natural_frame_key)
        for frame_id, image_path in enumerate(images, start=1):
            if not image_path.is_file():
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            yield frame_id, frame
        return

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")
    frame_id = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
    finally:
        capture.release()


def _track_color(uid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(uid)
    rgb = rng.integers(32, 256, size=3, dtype=np.int64)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _draw_multiline_label(
    image: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    x, y = origin
    for index, line in enumerate(text.splitlines()):
        baseline_y = y + int((index + 1) * max(16, 28 * font_scale))
        cv2.putText(
            image,
            line,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness=thickness,
        )


def _natural_frame_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (10**12, path.name)

