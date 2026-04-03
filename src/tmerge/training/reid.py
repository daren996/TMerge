"""Training entrypoints exposed through the new CLI."""

from __future__ import annotations

from typing import Any


def run_reid_training(config: dict[str, Any]) -> int:
    training = config.get("training", {})
    backend = training.get("backend", "tmerge_torchreid")
    if backend not in {"tmerge_torchreid", "legacy_torchreid"}:
        raise ValueError("Only the 'tmerge_torchreid' and 'legacy_torchreid' backends are supported")

    args = []
    if "config_file" in training:
        args.extend(["--config-file", str(training["config_file"])])
    args.extend(training.get("extra_args", []))

    if backend == "legacy_torchreid":
        try:
            from videosys.train.reid.train_reid import main as legacy_main
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Legacy ReID training requires the legacy training modules and optional dependencies."
            ) from exc

        import sys

        previous_argv = sys.argv[:]
        sys.argv = ["tmerge train reid", *args]
        try:
            legacy_main()
        finally:
            sys.argv = previous_argv
        return 0

    from tmerge.training.reid_runner import main as native_main

    return native_main(args)
