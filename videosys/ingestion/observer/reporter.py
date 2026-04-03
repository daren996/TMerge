"""Progress reporting utilities for legacy `videosys` pipelines."""

from __future__ import annotations

from pathlib import Path
import time

from videosys.ingestion import fields
from videosys.ingestion.base import Operator


class ProgressReporter(Operator):
    """Print and optionally persist periodic pipeline progress updates."""

    def __init__(self, report_interval: int = 100, save_file: str | Path | None = None) -> None:
        super().__init__()
        self.report_interval = report_interval
        self.save_file = Path(save_file) if save_file is not None else None
        self.source_type: str | None = None
        self.source_path: str | list[str] | None = None
        self.total_frames: float | None = None
        self.start_time = 0.0

    def prepare(self) -> None:
        if self.context.has(fields.META_VIDEO):
            meta = self.context.get(fields.META_VIDEO)
            self.source_type = "video"
            self.total_frames = float(meta.frame_count)
            self.source_path = meta.path
        elif self.context.has(fields.META_IMAGE):
            meta = self.context.get(fields.META_IMAGE)
            self.source_type = "image folder"
            self.total_frames = float(meta.total)
            self.source_path = meta.path

        if self.save_file is not None:
            self.save_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_file.write_text("", encoding="utf-8")

        self.start_time = time.perf_counter()
        self.report(f"begin to process: [{self.source_type}], path: [{self.source_path}]")

    def process(self, tables: dict[str, object]) -> None:
        frame_id = int(tables[fields.DATA_FRAME_ID])
        if self.report_interval > 0 and frame_id % self.report_interval == 0:
            elapsed = max(time.perf_counter() - self.start_time, 1e-9)
            fps = frame_id / elapsed
            if self.total_frames:
                progress = frame_id / self.total_frames
                message = (
                    f"processing: {frame_id:.0f}/{self.total_frames:.0f}, {progress:.0%}, "
                    f"time elapsed: {elapsed:.2f}s, fps: {fps:.2f}"
                )
            else:
                message = f"processing: {frame_id}, time elapsed: {elapsed:.2f}s, fps: {fps:.2f}"
            self.report(message)
        self.collector.emit(tables)

    def report(self, msg: str) -> None:
        if self.save_file is not None:
            with self.save_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{msg}\n")
        print(msg)

    def cleanup(self) -> None:
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        fps = (self.total_frames / elapsed) if self.total_frames else 0.0
        self.report(f"done. time elapsed: {elapsed:.2f}s. fps: {fps:.2f}")
