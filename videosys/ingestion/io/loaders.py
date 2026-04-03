"""Legacy MOT-format loaders used by older `videosys` pipelines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.ingestion.data import ObjectDetectionResult, ObjectTrackingResult

Tables = MutableMapping[str, Any]


def _parse_mot_line(line: str) -> tuple[int, int, np.ndarray, float]:
    parts = line.split(",")
    frame_id = int(parts[0])
    object_id = int(parts[1])
    left, top, width, height = map(float, parts[2:6])
    confidence = float(parts[6])
    bbox = np.array([left, top, left + width, top + height], dtype=float)
    return frame_id, object_id, bbox, confidence


class MOTDetLoader(Operator):
    """Load `det/det.txt` records from a MOT sequence folder."""

    def __init__(self, folder_path: str | Path) -> None:
        super().__init__()
        self.folder_path = Path(folder_path)
        self.frame_det_dict: defaultdict[int, list[ObjectDetectionResult]] = defaultdict(list)

    def prepare(self) -> None:
        det_file = self.folder_path / "det" / "det.txt"
        with det_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                frame_id, _, bbox, confidence = _parse_mot_line(line)
                self.frame_det_dict[frame_id].append(
                    ObjectDetectionResult(
                        bbox=bbox,
                        label=-1,
                        confidence=confidence,
                    )
                )

    def process(self, tables: Tables) -> None:
        frame_id = tables[fields.DATA_FRAME_ID]
        tables[fields.DATA_OBJECT_DETECTION] = self.frame_det_dict[frame_id]
        self.collector.emit(tables)


class MOTFormatLoader(Operator):
    """Load MOT-format tracking records into a configurable output field."""

    def __init__(self, file_path: str | Path, emit_key: str) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.emit_key = emit_key
        self.frame_det_dict: defaultdict[int, list[ObjectTrackingResult]] = defaultdict(list)

    def prepare(self) -> None:
        with self.file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                frame_id, object_id, bbox, confidence = _parse_mot_line(line)
                self.frame_det_dict[frame_id].append(
                    ObjectTrackingResult(
                        uid=object_id,
                        label=-1,
                        bbox=bbox,
                        confidence=confidence,
                        payload=ObjectDetectionResult(bbox=bbox, label=-1, confidence=1.0),
                    )
                )

    def process(self, tables: Tables) -> None:
        frame_id = tables[fields.DATA_FRAME_ID]
        tables[self.emit_key] = self.frame_det_dict[frame_id]
        self.collector.emit(tables)


class MOTResultLoader(MOTFormatLoader):
    """Load tracker outputs into `DATA_OBJECT_TRACK`."""

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path=file_path, emit_key=fields.DATA_OBJECT_TRACK)


class MOTGTLoader(MOTFormatLoader):
    """Load ground-truth tracks into `DATA_OBJECT_TRACK_GT`."""

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path=file_path, emit_key=fields.DATA_OBJECT_TRACK_GT)
