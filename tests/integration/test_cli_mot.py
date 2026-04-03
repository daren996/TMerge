from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tmerge.cli.main import main


def test_cli_mot_export_video(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(input_dir / "000001.png"), frame)
    result_path = tmp_path / "result.txt"
    result_path.write_text("1,1,1,1,4,4,0.9,-1,-1,-1\n", encoding="utf-8")
    output_path = tmp_path / "out.mp4"

    exit_code = main(
        [
            "mot",
            "export-video",
            "--data",
            str(input_dir),
            "--result",
            str(result_path),
            "--output",
            str(output_path),
            "--fps",
            "10",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_cli_export_video(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(input_dir / "000001.png"), frame)
    output_path = tmp_path / "plain.mp4"

    exit_code = main(
        [
            "export",
            "video",
            "--data",
            str(input_dir),
            "--output",
            str(output_path),
            "--fps",
            "12",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_cli_mot_filter(tmp_path: Path) -> None:
    result_path = tmp_path / "result.txt"
    result_path.write_text(
        "1,1,0,0,5,5,0.9,-1,-1,-1\n"
        "2,1,0,0,5,5,0.9,-1,-1,-1\n"
        "1,2,0,0,5,5,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "filtered.txt"

    exit_code = main(
        [
            "mot",
            "filter",
            "--result",
            str(result_path),
            "--output",
            str(output_path),
            "--min-frames",
            "2",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
