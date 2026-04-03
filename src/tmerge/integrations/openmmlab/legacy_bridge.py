"""Adapters that let the new runtime host selected legacy operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.models.types import FramePacket


class _LegacyContextProxy:
    def __init__(self, backing: dict[str, Any]) -> None:
        self._backing = backing

    def get(self, key: str) -> Any:
        return self._backing[key]

    def put(self, key: str, value: Any) -> None:
        self._backing[key] = value

    def has(self, key: str) -> bool:
        return key in self._backing


class _NoopCollector:
    def emit(self, tables: dict[str, Any] | None = None) -> None:
        return None


@dataclass(slots=True)
class LegacyOperatorAdapter(Operator):
    operator_factory: Any
    operator_kwargs: dict[str, Any] = field(default_factory=dict)

    def prepare(self, context: RuntimeContext) -> None:
        self._legacy_operator = self.operator_factory(**self.operator_kwargs)
        self._legacy_context = _LegacyContextProxy(context.state)
        self._legacy_operator.setup(context=self._legacy_context, collector=_NoopCollector())
        self._legacy_operator.prepare()

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        from videosys.ingestion import fields as legacy_fields

        tables = {
            legacy_fields.DATA_FRAME_ID: packet.frame_id,
            legacy_fields.DATA_FRAME: packet.frame,
            legacy_fields.DATA_FRAME_META: packet.frame_meta,
            legacy_fields.DATA_OBJECT_DETECTION: packet.detections,
            legacy_fields.DATA_OBJECT_TRACK: packet.tracks,
            legacy_fields.DATA_OBJECT_TRACK_GT: packet.ground_truth_tracks,
            legacy_fields.DATA_TRACK_FEAT: packet.track_features,
        }
        tables.update(packet.extras)
        self._legacy_operator.process(tables)
        packet.detections = list(tables.get(legacy_fields.DATA_OBJECT_DETECTION, packet.detections))
        packet.tracks = list(tables.get(legacy_fields.DATA_OBJECT_TRACK, packet.tracks))
        packet.ground_truth_tracks = list(
            tables.get(legacy_fields.DATA_OBJECT_TRACK_GT, packet.ground_truth_tracks)
        )
        packet.track_features = list(tables.get(legacy_fields.DATA_TRACK_FEAT, packet.track_features))
        packet.extras = {
            key: value
            for key, value in tables.items()
            if key
            not in {
                legacy_fields.DATA_FRAME_ID,
                legacy_fields.DATA_FRAME,
                legacy_fields.DATA_FRAME_META,
                legacy_fields.DATA_OBJECT_DETECTION,
                legacy_fields.DATA_OBJECT_TRACK,
                legacy_fields.DATA_OBJECT_TRACK_GT,
                legacy_fields.DATA_TRACK_FEAT,
            }
        }
        return packet

    def cleanup(self, context: RuntimeContext) -> None:
        self._legacy_operator.cleanup()
