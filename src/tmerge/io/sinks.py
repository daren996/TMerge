"""Output operators for frames, MOT exports, and track features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.models.types import FramePacket


@dataclass(slots=True)
class ImageFolderSink(Operator):
    path: Path
    image_field: str = "frame"

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ImageFolderSink":
        return cls(
            path=Path(str(config["path"])).expanduser(),
            image_field=str(config.get("image_field", "frame")),
        )

    def prepare(self, context: RuntimeContext) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        image = packet.frame if self.image_field == "frame" else packet.extras[self.image_field]
        output = self.path / f"{packet.frame_id:06d}.png"
        cv2.imwrite(str(output), image)
        return packet


@dataclass(slots=True)
class VideoSink(Operator):
    path: Path
    fps: float | None = None
    _writer: object = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "VideoSink":
        return cls(
            path=Path(str(config["path"])).expanduser(),
            fps=float(config["fps"]) if config.get("fps") is not None else None,
        )

    def prepare(self, context: RuntimeContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        if self._writer is None:
            fps = self.fps
            if fps is None:
                meta = context.get("input_meta")
                fps = getattr(meta, "fps", None) or 30.0
            height, width = packet.frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        self._writer.write(packet.frame)
        return packet

    def cleanup(self, context: RuntimeContext) -> None:
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()


@dataclass(slots=True)
class MotResultSink(Operator):
    path: Path
    include_label_column: bool = False

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "MotResultSink":
        return cls(
            path=Path(str(config["path"])).expanduser(),
            include_label_column=bool(config.get("include_label_column", False)),
        )

    def prepare(self, context: RuntimeContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        with self.path.open("a", encoding="utf-8") as handle:
            for track in packet.tracks:
                width = float(track.bbox[2] - track.bbox[0])
                height = float(track.bbox[3] - track.bbox[1])
                if self.include_label_column:
                    line = (
                        f"{packet.frame_id},{track.uid},{track.bbox[0]:.2f},{track.bbox[1]:.2f},"
                        f"{width:.2f},{height:.2f},{track.confidence},{track.label},-1,-1,-1\n"
                    )
                else:
                    line = (
                        f"{packet.frame_id},{track.uid},{track.bbox[0]:.2f},{track.bbox[1]:.2f},"
                        f"{width:.2f},{height:.2f},{track.confidence},-1,-1,-1\n"
                    )
                handle.write(line)
        return packet


@dataclass(slots=True)
class TrackFeatureSink(Operator):
    path: Path
    _rows: list[list[object]] = field(init=False, default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "TrackFeatureSink":
        return cls(path=Path(str(config["path"])).expanduser())

    def prepare(self, context: RuntimeContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        track_map = {track.uid: track for track in packet.tracks}
        for track_feature in packet.track_features:
            track = track_map.get(track_feature.uid)
            if track is None:
                continue
            self._rows.append(
                [
                    packet.frame_id,
                    *track_feature.bbox.tolist(),
                    track_feature.uid,
                    track.label,
                    track.confidence,
                    [track_feature.feature],
                ]
            )
        return packet

    def cleanup(self, context: RuntimeContext) -> None:
        if not self._rows:
            return
        frame = pd.DataFrame(
            data=np.array(self._rows, dtype=object),
            columns=["fid", "left", "top", "right", "bottom", "id", "label", "score", "feature"],
        )
        frame.to_pickle(self.path)

