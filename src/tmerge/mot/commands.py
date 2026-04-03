"""Top-level command handlers for MOT workflows."""

from __future__ import annotations

from pathlib import Path

from tmerge.evaluation.mot import evaluate_mot_results
from tmerge.mot.filtering import filter_mot_results
from tmerge.mot.reporting import save_mot_batch_report
from tmerge.mot.statistics import save_mot_track_statistics
from tmerge.visualization.mot import export_mot_video, visualize_mot_results


def run_evaluate(args) -> int:
    summary = evaluate_mot_results(
        ground_truth_path=args.ground_truth,
        result_path=args.result,
        distance=args.distance,
        distance_threshold=args.distance_threshold,
        gt_min_confidence=args.gt_min_confidence,
    )
    print(summary)
    return 0


def run_visualize(args) -> int:
    visualize_mot_results(
        data_path=args.data,
        result_path=args.result,
        start_frame=args.start_frame,
        thickness=args.thickness,
        font_scale=args.font_scale,
        wait_ms=args.wait_ms,
        window_name=args.window_name,
    )
    return 0


def run_export_video(args) -> int:
    frames_written = export_mot_video(
        data_path=args.data,
        result_path=args.result,
        output_path=args.output,
        fps=args.fps,
        thickness=args.thickness,
        font_scale=args.font_scale,
    )
    print(f"Wrote {frames_written} frames to {Path(args.output).expanduser()}")
    return 0


def run_filter(args) -> int:
    summary = filter_mot_results(
        result_path=args.result,
        output_path=args.output,
        image_dir=args.image_dir,
        min_frames=args.min_frames,
        min_width=args.min_width,
        min_height=args.min_height,
        drop_border_boxes=args.drop_border_boxes,
    )
    print(summary)
    return 0


def run_statistics(args) -> int:
    summary = save_mot_track_statistics(
        result_path=args.result,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        label=args.label,
    )
    print(summary)
    return 0


def run_batch_report(args) -> int:
    methods = []
    for item in args.method:
        if "=" not in item:
            raise ValueError(f"--method entries must use name=path format: {item}")
        name, path = item.split("=", 1)
        methods.append((name, path))
    report = save_mot_batch_report(
        ground_truth_path=args.ground_truth,
        methods=methods,
        output_csv=args.output_csv,
        distance=args.distance,
        distance_threshold=args.distance_threshold,
    )
    print(report.to_string(index=False))
    return 0
