"""Run configured MOT methods over a batch of MOT17 sequences."""

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
DEFAULT_TRAIN_SEQUENCES: Final[tuple[str, ...]] = (
    "MOT17-04-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
)
DEFAULT_TEST_SEQUENCES: Final[tuple[str, ...]] = (
    "MOT17-01-FRCNN",
    "MOT17-06-FRCNN",
    "MOT17-07-FRCNN",
    "MOT17-08-FRCNN",
    "MOT17-12-FRCNN",
    "MOT17-14-FRCNN",
)


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


def _build_command(dataset_name: str, split: str, method: MethodConfig) -> list[str]:
    dataset_path = Path("../storage/dataset/MOT17") / split / dataset_name / "img1"
    output_path = (
        Path("../storage/results/mot17")
        / dataset_name
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


def run_all_methods(dataset_name: str, split: str = "train") -> None:
    for method in METHODS:
        command = _build_command(dataset_name=dataset_name, split=split, method=method)
        print(f"working on method: {method.name}")
        print(f"executing command: {' '.join(command)}")
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset names to process")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "test"),
        help="Dataset split under ../storage/dataset/MOT17",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(args.datasets) or (
        DEFAULT_TRAIN_SEQUENCES if args.split == "train" else DEFAULT_TEST_SEQUENCES
    )
    for dataset_name in datasets:
        run_all_methods(dataset_name, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
