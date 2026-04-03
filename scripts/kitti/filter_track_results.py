"""Filter KITTI tracking outputs before downstream analysis."""

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


DEFAULT_VIDEOS: Final[tuple[str, ...]] = ("0013", "0014", "0015", "0016", "0017", "0019")


def preprocess(
    video_name: str,
    method: str,
    train_test: str = "training",
    obj_frame_threshold: int = 10,
    *,
    filter_borders: bool = False,
    filter_small_boxes: bool = False,
) -> None:
    result_path = Path("../storage/results/kitti") / video_name / f"faster_rcnn-{method}-person.txt"
    image_dir = Path("../storage/dataset/KITTI") / train_test / "image_02" / video_name
    output_path = (
        Path("../storage/results/kitti")
        / video_name
        / "filtered-tracked"
        / f"faster_rcnn-{method}-person.txt"
    )

    results = load_tracking_results(result_path)
    original_count = len(results)
    results = filter_short_tracks(results, min_frames=obj_frame_threshold)
    if filter_borders:
        results = filter_bboxes_on_borders(results, image_dir=image_dir)
    if filter_small_boxes:
        results = filter_small_bboxes(results)

    summary = summarize_track_results(results)
    print(
        f"{video_name} origin feats: {original_count}, new feats: {summary['rows']}, "
        f"tracks: {summary['tracks']}, avg bboxes: {summary['avg_bboxes']}"
    )
    write_tracking_results(output_path, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", help="KITTI video ids to process")
    parser.add_argument("--method", default="tracktor", help="Tracking method name in the result filename")
    parser.add_argument("--split", default="training")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--filter-borders", action="store_true")
    parser.add_argument("--filter-small-boxes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    videos = tuple(args.videos) or DEFAULT_VIDEOS
    for video_name in videos:
        preprocess(
            video_name,
            args.method,
            train_test=args.split,
            obj_frame_threshold=args.min_frames,
            filter_borders=args.filter_borders,
            filter_small_boxes=args.filter_small_boxes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
