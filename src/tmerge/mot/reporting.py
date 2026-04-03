"""Batch evaluation helpers for multiple MOT result files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def evaluate_mot_batch(
    ground_truth_path: str | Path,
    methods: list[tuple[str, str | Path]],
    *,
    distance: str = "iou",
    distance_threshold: float = 0.5,
) -> pd.DataFrame:
    mm = _require_motmetrics()
    gt = mm.io.loadtxt(str(Path(ground_truth_path).expanduser()), fmt=mm.io.Format.MOT16, min_confidence=1)
    loaded_methods = [
        (name, mm.io.loadtxt(str(Path(path).expanduser()), fmt=mm.io.Format.MOT16))
        for name, path in methods
    ]
    accs = []
    names = []
    for name, method_data in loaded_methods:
        accs.append(mm.utils.compare_to_groundtruth(gt, method_data, distance, distth=distance_threshold))
        names.append(name)
    handler = mm.metrics.create()
    summary = handler.compute_many(
        accs,
        names=names,
        metrics=list(mm.metrics.motchallenge_metrics),
        generate_overall=False,
    )
    return summary.reset_index().rename(columns={"index": "method"})


def save_mot_batch_report(
    ground_truth_path: str | Path,
    methods: list[tuple[str, str | Path]],
    *,
    output_csv: str | Path | None = None,
    distance: str = "iou",
    distance_threshold: float = 0.5,
) -> pd.DataFrame:
    report = evaluate_mot_batch(
        ground_truth_path=ground_truth_path,
        methods=methods,
        distance=distance,
        distance_threshold=distance_threshold,
    )
    if output_csv is not None:
        output_path = Path(output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report


def _require_motmetrics() -> Any:
    try:
        import motmetrics
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Batch MOT reporting requires optional dependency 'motmetrics'. "
            'Install with: python -m pip install motmetrics'
        ) from exc
    return motmetrics

