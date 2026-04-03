"""Direct export command handlers."""

from __future__ import annotations

from pathlib import Path

from tmerge.io.video import images_to_video


def run_export_video(args) -> int:
    written = images_to_video(args.data, args.output, fps=args.fps)
    print(f"Wrote {written} frames to {Path(args.output).expanduser()}")
    return 0
