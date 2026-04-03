from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tmerge.cli.main import main


def test_cli_runs_image_copy_pipeline(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "000001.png"), image)
    cv2.imwrite(str(input_dir / "000002.png"), image)

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "pipeline:\n"
        "  name: smoke\n"
        "  operators:\n"
        "    - type: source.image_folder\n"
        f"      path: {input_dir}\n"
        "    - type: reporter.progress\n"
        "      report_interval: 1\n"
        "    - type: sink.image_folder\n"
        f"      path: {output_dir}\n",
        encoding="utf-8",
    )

    exit_code = main(["run", "pipeline", str(config)])

    assert exit_code == 0
    assert (output_dir / "000001.png").exists()
    assert (output_dir / "000002.png").exists()
