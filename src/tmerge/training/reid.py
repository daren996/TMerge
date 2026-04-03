"""Training entrypoints exposed through the new CLI."""

from __future__ import annotations

from typing import Any


def run_reid_training(config: dict[str, Any]) -> int:
    training = config.get("training", {})
    if training.get("backend", "legacy_torchreid") != "legacy_torchreid":
        raise ValueError("Only the 'legacy_torchreid' backend is currently supported")

    try:
        from videosys.train.reid.train_reid import main as legacy_main
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ReID training requires the legacy training modules and optional dependencies."
        ) from exc

    import sys

    args = []
    if "config_file" in training:
        args.extend(["--config-file", str(training["config_file"])])
    args.extend(training.get("extra_args", []))
    previous_argv = sys.argv[:]
    sys.argv = ["tmerge train reid", *args]
    try:
        legacy_main()
    finally:
        sys.argv = previous_argv
    return 0

