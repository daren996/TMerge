"""Utilities for reading and writing MOT-format tracking files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmerge.models.types import TrackingResult


@dataclass(slots=True)
class MotRow:
    frame_id: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float
    class_id: int = -1
    visibility: float = -1.0
    world_x: float = -1.0

    @property
    def bbox(self) -> np.ndarray:
        return np.array(
            [self.left, self.top, self.left + self.width, self.top + self.height],
            dtype=float,
        )


def load_mot_tracks(
    path: str | Path,
    *,
    min_confidence: float | None = None,
) -> dict[int, list[TrackingResult]]:
    tracks_by_frame: defaultdict[int, list[TrackingResult]] = defaultdict(list)
    for row in load_mot_rows(path, min_confidence=min_confidence):
        tracks_by_frame[row.frame_id].append(
            TrackingResult(
                uid=row.track_id,
                label=row.class_id,
                bbox=row.bbox,
                confidence=row.confidence,
            )
        )
    return dict(tracks_by_frame)


def load_mot_rows(path: str | Path, *, min_confidence: float | None = None) -> list[MotRow]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {file_path}")

    rows: list[MotRow] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise ValueError(f"Invalid MOT line in {file_path}: {raw_line.rstrip()}")
            confidence = float(parts[6])
            if min_confidence is not None and confidence < min_confidence:
                continue
            rows.append(
                MotRow(
                    frame_id=int(parts[0]),
                    track_id=int(parts[1]),
                    left=float(parts[2]),
                    top=float(parts[3]),
                    width=float(parts[4]),
                    height=float(parts[5]),
                    confidence=confidence,
                    class_id=int(parts[7]) if len(parts) > 7 and parts[7] not in {"", "-1"} else -1,
                    visibility=float(parts[8]) if len(parts) > 8 and parts[8] not in {"", "-1"} else -1.0,
                    world_x=float(parts[9]) if len(parts) > 9 and parts[9] not in {"", "-1"} else -1.0,
                )
            )
    return rows


def write_mot_rows(path: str | Path, rows: list[MotRow]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row.frame_id},{row.track_id},{row.left},{row.top},{row.width},{row.height},"
                f"{row.confidence},{row.class_id},{row.visibility},{row.world_x}\n"
            )
