"""Run configured MOT methods over a batch of KITTI sequences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Final


DETECTOR_CONFIG: Final[Path] = Path("e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py")
CHECKPOINT_URL: Final[str] = (
    "https://download.openmmlab.com/mmtracking/mot/faster_rcnn/"
    "faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth"
)
DEFAULT_VIDEO_NAMES: Final[tuple[str, ...]] = ("0013", "0014", "0015", "0016", "0017", "0019")


@dataclass(frozen=True, slots=True)
class MethodConfig:
    name: str
    config_file: str
    detection_method: str
    extra_args: tuple[str, ...]


METHODS: Final[tuple[MethodConfig, ...]] = (
    MethodConfig(
        name="tracktor",
        config_file="mmt_tracktor_private.py",
        detection_method="faster_rcnn",
        extra_args=(
            "--config",
            str(DETECTOR_CONFIG),
            "--checkpoint",
            CHECKPOINT_URL,
        ),
    ),
)


def _build_command(video_name: str, split: str, method: MethodConfig) -> list[str]:
    dataset_path = Path("../storage/dataset/KITTI") / split / "image_02" / video_name
    output_path = (
        Path("../storage/results/kitti")
        / video_name
        / f"{method.detection_method}-{method.name}-person.txt"
    )
    return [
        sys.executable,
        "e2e/ingestion_runner.py",
        str(Path("e2e/configs/tracking") / method.config_file),
        "--path",
        str(dataset_path),
        "--output",
        str(output_path),
        *method.extra_args,
    ]


def run_all_methods(video_name: str, split: str = "training") -> None:
    for method in METHODS:
        command = _build_command(video_name=video_name, split=split, method=method)
        print(f"working on method: {method.name}")
        print(f"executing command: {' '.join(command)}")
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", help="KITTI video ids to process")
    parser.add_argument(
        "--split",
        default="training",
        help="Dataset split under ../storage/dataset/KITTI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    videos = tuple(args.videos) or DEFAULT_VIDEO_NAMES
    for video_name in videos:
        run_all_methods(video_name, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
