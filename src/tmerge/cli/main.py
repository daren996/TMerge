"""Unified command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from tmerge.config.loader import load_yaml_config
from tmerge.config.validation import ConfigValidationError, validate_config
from tmerge.core.factory import build_pipeline
from tmerge.core.runtime import RuntimeContext
from tmerge.export.commands import run_export_video as run_plain_export_video
from tmerge.mot.commands import (
    run_batch_report,
    run_evaluate,
    run_export_video,
    run_filter,
    run_statistics,
    run_visualize,
)
from tmerge.training.reid import run_reid_training
from tmerge.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmerge", description="Modern TMerge command line")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured pipeline")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    pipeline_parser = run_subparsers.add_parser("pipeline", help="Run a pipeline config")
    _add_config_args(pipeline_parser)

    export_parser = subparsers.add_parser("export", help="Run export-oriented pipelines")
    export_subparsers = export_parser.add_subparsers(dest="export_command", required=True)
    export_tracks = export_subparsers.add_parser("tracks", help="Export track results")
    _add_config_args(export_tracks)
    export_features = export_subparsers.add_parser("features", help="Export track features")
    _add_config_args(export_features)
    export_video_parser = export_subparsers.add_parser("video", help="Generate a plain video from an image folder")
    export_video_parser.add_argument("--data", required=True, type=Path)
    export_video_parser.add_argument("--output", required=True, type=Path)
    export_video_parser.add_argument("--fps", type=float, default=30.0)

    train_parser = subparsers.add_parser("train", help="Run model training workflows")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    reid_parser = train_subparsers.add_parser("reid", help="Train a ReID model")
    _add_config_args(reid_parser)

    mot_parser = subparsers.add_parser("mot", help="MOT-specific utilities")
    mot_subparsers = mot_parser.add_subparsers(dest="mot_command", required=True)

    eval_parser = mot_subparsers.add_parser("evaluate", help="Evaluate MOT results against ground truth")
    eval_parser.add_argument("--ground-truth", required=True, dest="ground_truth", type=Path)
    eval_parser.add_argument("--result", required=True, type=Path)
    eval_parser.add_argument("--distance", choices=["iou", "euclidean"], default="iou")
    eval_parser.add_argument("--distance-threshold", type=float, default=0.5)
    eval_parser.add_argument("--gt-min-confidence", type=float, default=1.0)

    visualize_parser = mot_subparsers.add_parser("visualize", help="Visualize MOT results on frames")
    visualize_parser.add_argument("--data", required=True, type=Path)
    visualize_parser.add_argument("--result", required=True, type=Path)
    visualize_parser.add_argument("--start-frame", type=int, default=1)
    visualize_parser.add_argument("--thickness", type=int, default=2)
    visualize_parser.add_argument("--font-scale", type=float, default=0.6)
    visualize_parser.add_argument("--wait-ms", type=int, default=100)
    visualize_parser.add_argument("--window-name", default="tmerge-mot")

    export_video_parser = mot_subparsers.add_parser("export-video", help="Render MOT results to a video file")
    export_video_parser.add_argument("--data", required=True, type=Path)
    export_video_parser.add_argument("--result", required=True, type=Path)
    export_video_parser.add_argument("--output", required=True, type=Path)
    export_video_parser.add_argument("--fps", type=float, default=None)
    export_video_parser.add_argument("--thickness", type=int, default=2)
    export_video_parser.add_argument("--font-scale", type=float, default=0.6)

    filter_parser = mot_subparsers.add_parser("filter", help="Filter MOT results for downstream workflows")
    filter_parser.add_argument("--result", required=True, type=Path)
    filter_parser.add_argument("--output", required=True, type=Path)
    filter_parser.add_argument("--image-dir", type=Path, default=None)
    filter_parser.add_argument("--min-frames", type=int, default=1)
    filter_parser.add_argument("--min-width", type=float, default=0.0)
    filter_parser.add_argument("--min-height", type=float, default=0.0)
    filter_parser.add_argument("--drop-border-boxes", action="store_true")

    statistics_parser = mot_subparsers.add_parser("statistics", help="Summarize track durations for a MOT result file")
    statistics_parser.add_argument("--result", required=True, type=Path)
    statistics_parser.add_argument("--output-csv", type=Path, default=None)
    statistics_parser.add_argument("--output-plot", type=Path, default=None)
    statistics_parser.add_argument("--label", default=None)

    batch_parser = mot_subparsers.add_parser("batch-report", help="Evaluate multiple MOT result files at once")
    batch_parser.add_argument("--ground-truth", required=True, dest="ground_truth", type=Path)
    batch_parser.add_argument("--method", action="append", required=True, default=[], help="Method entry in name=path format")
    batch_parser.add_argument("--output-csv", type=Path, default=None)
    batch_parser.add_argument("--distance", choices=["iou", "euclidean"], default="iou")
    batch_parser.add_argument("--distance-threshold", type=float, default=0.5)

    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="Path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted key=value syntax",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(verbose=args.verbose)
    context = RuntimeContext(logger=logger)

    if args.command == "mot":
        if args.mot_command == "evaluate":
            return run_evaluate(args)
        if args.mot_command == "visualize":
            return run_visualize(args)
        if args.mot_command == "export-video":
            return run_export_video(args)
        if args.mot_command == "filter":
            return run_filter(args)
        if args.mot_command == "statistics":
            return run_statistics(args)
        if args.mot_command == "batch-report":
            return run_batch_report(args)
        parser.error(f"Unknown MOT subcommand: {args.mot_command}")

    if args.command == "export" and args.export_command == "video":
        return run_plain_export_video(args)

    try:
        config = load_yaml_config(args.config, args.overrides)
        validate_config(config)
    except (ValueError, ConfigValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "train" and args.train_command == "reid":
        return run_reid_training(config)

    pipeline = build_pipeline(config, context=context)
    pipeline.run()
    return 0
