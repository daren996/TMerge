"""Filter PathTrack tracking outputs before downstream analysis."""

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
    filter_small_bboxes,
    load_tracking_results,
    summarize_track_results,
    write_tracking_results,
)


DEFAULT_DATASETS: Final[tuple[str, ...]] = (
    "0Xtp77A4zF4_0_2000",
    "0Xtp77A4zF4_1000_3000",
    "0Xtp77A4zF4_2000_4000",
    "0Xtp77A4zF4_3000_5000",
    "0Xtp77A4zF4_4000_6000",
)


def preprocess(
    dataset: str,
    method: str,
    train_test: str = "selected",
    obj_frame_threshold: int = 100,
) -> None:
    result_path = Path("../storage/results/pathtrack") / dataset / f"faster_rcnn-{method}-person.txt"
    image_dir = Path("../storage/dataset/PathTrack") / train_test / dataset / "img1"
    output_path = (
        Path("../storage/results/pathtrack")
        / dataset
        / "filtered-tracked"
        / f"faster_rcnn-{method}-person.txt"
    )

    results = load_tracking_results(result_path)
    original_count = len(results)
    results = filter_short_tracks(results, min_frames=obj_frame_threshold)
    results = filter_bboxes_on_borders(results, image_dir=image_dir)
    results = filter_small_bboxes(results)

    summary = summarize_track_results(results)
    print(
        f"{dataset} origin feats: {original_count}, new feats: {summary['rows']}, "
        f"tracks: {summary['tracks']}, avg bboxes: {summary['avg_bboxes']}"
    )
    write_tracking_results(output_path, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="PathTrack dataset shards to process")
    parser.add_argument("--method", default="tracktor", help="Tracking method name in the result filename")
    parser.add_argument("--split", default="selected")
    parser.add_argument("--min-frames", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(args.datasets) or DEFAULT_DATASETS
    for dataset in datasets:
        preprocess(dataset, args.method, train_test=args.split, obj_frame_threshold=args.min_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
