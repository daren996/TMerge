"""Statistics and reporting helpers for MOT result sets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from tmerge.io.mot import load_mot_rows


def mot_track_statistics(result_path: str | Path) -> dict[str, Any]:
    rows = load_mot_rows(result_path)
    durations = Counter(row.track_id for row in rows)
    if not durations:
        return {
            "rows": 0,
            "tracks": 0,
            "mean_duration": 0.0,
            "median_duration": 0.0,
            "max_duration": 0,
            "min_duration": 0,
        }
    values = list(durations.values())
    series = pd.Series(values, dtype="float64")
    return {
        "rows": len(rows),
        "tracks": len(durations),
        "mean_duration": round(float(series.mean()), 4),
        "median_duration": round(float(series.median()), 4),
        "max_duration": int(series.max()),
        "min_duration": int(series.min()),
    }


def save_mot_track_statistics(
    result_path: str | Path,
    *,
    output_csv: str | Path | None = None,
    output_plot: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    rows = load_mot_rows(result_path)
    durations = Counter(row.track_id for row in rows)
    data = pd.DataFrame(
        {
            "track_id": list(durations.keys()),
            "duration": list(durations.values()),
        }
    )
    if output_csv is not None:
        output_csv_path = Path(output_csv).expanduser()
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.sort_values("duration", ascending=False).to_csv(output_csv_path, index=False)
    if output_plot is not None:
        _save_duration_plot(data, output_plot, title=label or Path(result_path).stem)
    return mot_track_statistics(result_path)


def _save_duration_plot(data: pd.DataFrame, output_plot: str | Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plot export requires matplotlib. Install with: python -m pip install matplotlib"
        ) from exc
    output_plot_path = Path(output_plot).expanduser()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not data.empty:
        ax.hist(data["duration"], bins=max(10, min(40, len(data))), color="#3b82f6", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Track duration")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=150)
    plt.close(fig)

