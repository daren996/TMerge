"""Simple video export helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def images_to_video(
    data_path: str | Path,
    output_path: str | Path,
    *,
    fps: float = 30.0,
) -> int:
    input_dir = Path(data_path).expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {input_dir}")
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(candidate for candidate in input_dir.iterdir() if candidate.is_file())
    writer = None
    written = 0
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_file),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written

