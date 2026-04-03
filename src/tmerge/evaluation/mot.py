"""MOT evaluation helpers backed by motmetrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tmerge.io.mot import load_mot_tracks


def evaluate_mot_results(
    ground_truth_path: str | Path,
    result_path: str | Path,
    *,
    distance: str = "iou",
    distance_threshold: float = 0.5,
    gt_min_confidence: float = 1.0,
) -> str:
    mm = _require_motmetrics()
    accumulator = mm.mot.MOTAccumulator()
    distance_fn = mm.distances.iou_matrix if distance == "iou" else mm.distances.norm2squared_matrix

    gt_tracks = load_mot_tracks(ground_truth_path, min_confidence=gt_min_confidence)
    predicted_tracks = load_mot_tracks(result_path)
    frame_ids = sorted(set(gt_tracks) | set(predicted_tracks))

    for frame_id in frame_ids:
        gt_frame = gt_tracks.get(frame_id, [])
        pred_frame = predicted_tracks.get(frame_id, [])

        oids = np.array([track.uid for track in gt_frame])
        hids = np.array([track.uid for track in pred_frame])
        oboxes = np.array([track.bbox for track in gt_frame], dtype=float)
        hboxes = np.array([track.bbox for track in pred_frame], dtype=float)

        if len(oboxes) > 0:
            oboxes[:, 2] = oboxes[:, 2] - oboxes[:, 0]
            oboxes[:, 3] = oboxes[:, 3] - oboxes[:, 1]
        if len(hboxes) > 0:
            hboxes[:, 2] = hboxes[:, 2] - hboxes[:, 0]
            hboxes[:, 3] = hboxes[:, 3] - hboxes[:, 1]

        dists = np.empty((0, 0))
        if len(oids) > 0 and len(hids) > 0:
            dists = distance_fn(oboxes, hboxes, distance_threshold)
        accumulator.update(oids, hids, dists, frame_id)

    metrics_handler = mm.metrics.create()
    summary = metrics_handler.compute(
        accumulator,
        metrics=mm.metrics.motchallenge_metrics,
        name="acc",
    )
    return mm.io.render_summary(
        summary,
        formatters=metrics_handler.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )


def _require_motmetrics() -> Any:
    try:
        import motmetrics
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MOT evaluation requires optional dependency 'motmetrics'. "
            'Install with: python -m pip install motmetrics'
        ) from exc
    return motmetrics

