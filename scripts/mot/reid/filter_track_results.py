"""Filter MOT17 tracking outputs before downstream ReID analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.track_filtering import (
    filter_bboxes_on_borders,
    filter_short_tracks,
    load_tracking_results,
    write_tracking_results,
)


DEFAULT_TRAIN_DATASETS: Final[tuple[str, ...]] = (
    "MOT17-04-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
)
DEFAULT_TEST_DATASETS: Final[tuple[str, ...]] = (
    "MOT17-01-FRCNN",
    "MOT17-06-FRCNN",
    "MOT17-07-FRCNN",
    "MOT17-08-FRCNN",
    "MOT17-12-FRCNN",
    "MOT17-14-FRCNN",
)


def preprocess(
    dataset: str,
    method: str,
    train_test: str = "train",
    obj_frame_threshold: int = 100,
) -> None:
    result_path = Path("../storage/results/mot17") / dataset / f"faster_rcnn-{method}-person.txt"
    image_dir = Path("../storage/dataset/MOT17") / train_test / dataset / "img1"
    output_path = (
        Path("../storage/results/mot17") / dataset / "filtered-tracked" / f"faster_rcnn-{method}-person.txt"
    )

    results = load_tracking_results(result_path)
    original_count = len(results)
    results = filter_short_tracks(results, min_frames=obj_frame_threshold)
    results = filter_bboxes_on_borders(results, image_dir=image_dir)

    print(f"{dataset} origin feats: {original_count}, new feats: {len(results)}")
    write_tracking_results(output_path, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="MOT17 datasets to process")
    parser.add_argument("--method", default="uma", help="Tracking method name in the result filename")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--min-frames", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(args.datasets) or (
        DEFAULT_TRAIN_DATASETS if args.split == "train" else DEFAULT_TEST_DATASETS
    )
    for dataset in datasets:
        preprocess(dataset, args.method, train_test=args.split, obj_frame_threshold=args.min_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
