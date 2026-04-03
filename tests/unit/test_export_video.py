from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tmerge.io.video import images_to_video


def test_images_to_video_creates_mp4(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    frame = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "000001.png"), frame)
    cv2.imwrite(str(input_dir / "000002.png"), frame)
    output_path = tmp_path / "output.mp4"

    written = images_to_video(input_dir, output_path, fps=12)

    assert written == 2
    assert output_path.exists()

