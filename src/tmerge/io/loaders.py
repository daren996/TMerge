"""Frame-aligned file loaders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.io.mot import load_mot_rows
from tmerge.models.types import DetectionResult, FramePacket, TrackingResult


@dataclass(slots=True)
class MotResultLoader(Operator):
    path: Path
    emit_detections: bool = False
    _frame_map: dict[int, list[TrackingResult | DetectionResult]] = field(
        init=False,
        default_factory=dict,
    )

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "MotResultLoader":
        return cls(
            path=Path(str(config["path"])).expanduser(),
            emit_detections=bool(config.get("emit_detections", False)),
        )

    def prepare(self, context: RuntimeContext) -> None:
        rows = load_mot_rows(self.path)
        frame_map: defaultdict[int, list[TrackingResult | DetectionResult]] = defaultdict(list)
        for row in rows:
            if self.emit_detections:
                frame_map[row.frame_id].append(
                    DetectionResult(bbox=row.bbox, label=-1, confidence=row.confidence)
                )
            else:
                frame_map[row.frame_id].append(
                    TrackingResult(
                        uid=row.track_id,
                        label=row.class_id,
                        bbox=row.bbox,
                        confidence=row.confidence,
                    )
                )
        self._frame_map = dict(frame_map)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        loaded = self._frame_map.get(packet.frame_id, [])
        if self.emit_detections:
            packet.detections = list(loaded)
        else:
            packet.tracks = list(loaded)
        return packet

