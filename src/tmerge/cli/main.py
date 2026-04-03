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

    train_parser = subparsers.add_parser("train", help="Run model training workflows")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    reid_parser = train_subparsers.add_parser("reid", help="Train a ReID model")
    _add_config_args(reid_parser)

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
