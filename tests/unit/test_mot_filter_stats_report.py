from __future__ import annotations

from pathlib import Path

import pandas as pd

from tmerge.mot.filtering import filter_mot_results
from tmerge.mot.reporting import save_mot_batch_report
from tmerge.mot.statistics import mot_track_statistics, save_mot_track_statistics


def test_filter_mot_results_enforces_min_frames(tmp_path: Path) -> None:
    result_path = tmp_path / "result.txt"
    result_path.write_text(
        "1,1,0,0,5,5,0.9,-1,-1,-1\n"
        "2,1,0,0,5,5,0.9,-1,-1,-1\n"
        "1,2,0,0,5,5,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "filtered.txt"

    summary = filter_mot_results(result_path, output_path, min_frames=2)

    assert summary["rows_after"] == 2.0
    text = output_path.read_text(encoding="utf-8")
    assert ",2," not in text


def test_mot_track_statistics_computes_summary(tmp_path: Path) -> None:
    result_path = tmp_path / "result.txt"
    result_path.write_text(
        "1,1,0,0,5,5,0.9,-1,-1,-1\n"
        "2,1,0,0,5,5,0.9,-1,-1,-1\n"
        "1,2,0,0,5,5,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )

    summary = mot_track_statistics(result_path)

    assert summary["tracks"] == 2
    assert summary["max_duration"] == 2


def test_save_mot_batch_report_writes_csv(monkeypatch, tmp_path: Path) -> None:
    fake_report = pd.DataFrame({"method": ["sort"], "mota": [0.5]})
    monkeypatch.setattr(
        "tmerge.mot.reporting.evaluate_mot_batch",
        lambda *args, **kwargs: fake_report,
    )
    output_csv = tmp_path / "report.csv"

    report = save_mot_batch_report(
        ground_truth_path="gt.txt",
        methods=[("sort", "sort.txt")],
        output_csv=output_csv,
    )

    assert report.equals(fake_report)
    assert output_csv.exists()
