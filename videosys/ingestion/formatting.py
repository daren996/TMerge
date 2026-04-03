"""Formatting helpers for legacy `videosys` ingestion pipelines."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.ingestion.data import ObjectDetectionResult, ObjectTrackingResult

Tables = MutableMapping[str, Any]


class DetToTrackResult(Operator):
    """Convert detections above a confidence threshold into anonymous tracks."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def process(self, tables: Tables) -> None:
        detections: list[ObjectDetectionResult] = tables[fields.DATA_OBJECT_DETECTION]
        tables[fields.DATA_OBJECT_TRACK] = [
            ObjectTrackingResult(-1, det.label, det.bbox, det.confidence, det)
            for det in detections
            if det.confidence >= self.threshold
        ]
        self.collector.emit(tables)
