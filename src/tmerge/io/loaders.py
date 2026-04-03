"""Frame-aligned file loaders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tmerge.core.runtime import Operator, RuntimeContext
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
        if not self.path.exists():
            raise FileNotFoundError(f"MOT result file not found: {self.path}")
        frame_map: defaultdict[int, list[TrackingResult | DetectionResult]] = defaultdict(list)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                left, top, width, height = map(float, parts[2:6])
                score = float(parts[6])
                bbox = np.array([left, top, left + width, top + height], dtype=float)
                if self.emit_detections:
                    frame_map[frame_id].append(DetectionResult(bbox=bbox, label=-1, confidence=score))
                else:
                    frame_map[frame_id].append(
                        TrackingResult(
                            uid=track_id,
                            label=-1,
                            bbox=bbox,
                            confidence=score,
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

