"""Tracking result sinks for legacy `videosys` pipelines."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

from videosys.ingestion import fields
from videosys.ingestion.base import Operator

Tables = MutableMapping[str, Any]


class AbstractTrackResultSink(Operator):
    """Base sink for serializing tracking results."""

    def __init__(self, save_at_end: bool = False) -> None:
        super().__init__()
        self._save_at_end = save_at_end
        self._buffers: list[Tables] = []

    def store(self, fid: int, track_results: Sequence[object]) -> None:
        """Persist one frame of tracking results."""

    def process(self, tables: Tables) -> None:
        if not self._save_at_end:
            track_results = tables[fields.DATA_OBJECT_TRACK]
            frame_id = tables[fields.DATA_FRAME_ID]
            self.store(frame_id, track_results)
        else:
            self._buffers.append(dict(tables))
        self.collector.emit(tables)

    def cleanup(self) -> None:
        if self._save_at_end:
            for tables in self._buffers:
                track_results = tables[fields.DATA_OBJECT_TRACK]
                frame_id = tables[fields.DATA_FRAME_ID]
                self.store(frame_id, track_results)


class TrackResultFolderSink(AbstractTrackResultSink):
    """Write one text file per frame into an output folder."""

    def __init__(self, output_folder: str | Path, **kwargs: Any) -> None:
        self.output_folder = Path(output_folder)
        self.file_path = self.output_folder
        super().__init__(**kwargs)

    def prepare(self) -> None:
        self.file_path.mkdir(parents=True, exist_ok=True)

    def store(self, fid: int, track_results: Sequence[object]) -> None:
        output_path = self.file_path / f"{fid}.txt"
        with output_path.open("w", encoding="utf-8") as handle:
            for tracklet in track_results:
                if getattr(tracklet, "payload", None) is None:
                    raise ValueError("TrackResultFolderSink expects track payload metadata to be present.")
                handle.write(
                    f"{tracklet.uid};{tracklet.bbox};{tracklet.payload.label};"
                    f"{tracklet.payload.confidence}\n"
                )


class MOTResultSink(AbstractTrackResultSink):
    """Append tracking results to a MOT-format text file."""

    def __init__(self, path: str | Path, add_type_column: bool = False, **kwargs: Any) -> None:
        self.output_path = Path(path)
        self.add_type_column = add_type_column
        self.file_path = self.output_path
        super().__init__(**kwargs)

    def prepare(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self.file_path = self.output_path

    def store(self, fid: int, track_results: Sequence[object]) -> None:
        with self.file_path.open("a", encoding="utf-8") as handle:
            for tracklet in track_results:
                if self.add_type_column:
                    handle.write(
                        "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{},{}\n".format(
                            fid,
                            tracklet.uid,
                            tracklet.bbox[0],
                            tracklet.bbox[1],
                            tracklet.bbox[2] - tracklet.bbox[0],
                            tracklet.bbox[3] - tracklet.bbox[1],
                            tracklet.confidence,
                            tracklet.label,
                            -1,
                            -1,
                            -1,
                        )
                    )
                else:
                    handle.write(
                        "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{}\n".format(
                            fid,
                            tracklet.uid,
                            tracklet.bbox[0],
                            tracklet.bbox[1],
                            tracklet.bbox[2] - tracklet.bbox[0],
                            tracklet.bbox[3] - tracklet.bbox[1],
                            tracklet.confidence,
                            -1,
                            -1,
                            -1,
                        )
                    )
