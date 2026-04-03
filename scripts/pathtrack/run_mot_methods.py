"""Run configured MOT methods over batches of PathTrack sequences."""

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
DEFAULT_DATASETS: Final[tuple[str, ...]] = (
    "0Xtp77A4zF4_0_2000",
    "0Xtp77A4zF4_1000_3000",
    "0Xtp77A4zF4_2000_4000",
    "0Xtp77A4zF4_3000_5000",
    "0Xtp77A4zF4_4000_6000",
    "0Xtp77A4zF4_5000_7000",
    "0Xtp77A4zF4_6000_8000",
    "0Xtp77A4zF4_7000_9000",
    "0Xtp77A4zF4_8000_10000",
    "0Xtp77A4zF4_9000_11000",
    "0Xtp77A4zF4_10000_12000",
    "0Xtp77A4zF4_11000_13000",
    "0Xtp77A4zF4_12000_14000",
    "0Xtp77A4zF4_13000_15000",
    "0Xtp77A4zF4_14000_16000",
    "0Xtp77A4zF4_15000_17000",
    "0Xtp77A4zF4_16000_18000",
    "0Xtp77A4zF4_17000_19000",
    "0Xtp77A4zF4_18000_20000",
    "0Xtp77A4zF4_19000_21000",
    "0Xtp77A4zF4_20000_21635",
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
    dataset_path = Path("../storage/dataset/PathTrack") / split / dataset_name / "img1"
    output_path = (
        Path("../storage/results/pathtrack")
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


def run_all_methods(dataset_name: str, split: str = "selected") -> None:
    for method in METHODS:
        command = _build_command(dataset_name=dataset_name, split=split, method=method)
        print(f"working on method: {method.name}")
        print(f"executing command: {' '.join(command)}")
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="PathTrack dataset shards to process")
    parser.add_argument(
        "--split",
        default="selected",
        help="Dataset split under ../storage/dataset/PathTrack",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(args.datasets) or DEFAULT_DATASETS
    for dataset_name in datasets:
        run_all_methods(dataset_name, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
