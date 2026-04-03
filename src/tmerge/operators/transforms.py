"""Simple built-in packet transformations."""

from __future__ import annotations

from dataclasses import dataclass

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.models.types import FramePacket, TrackingResult


@dataclass(slots=True)
class DetectionsToTracks(Operator):
    threshold: float = 0.5

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "DetectionsToTracks":
        return cls(threshold=float(config.get("threshold", 0.5)))

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        packet.tracks = [
            TrackingResult(
                uid=-1,
                label=detection.label,
                bbox=detection.bbox,
                confidence=detection.confidence,
                payload=detection,
            )
            for detection in packet.detections
            if detection.confidence >= self.threshold
        ]
        return packet

