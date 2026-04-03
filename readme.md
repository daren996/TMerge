# TMerge

TMerge is a research engineering toolkit for video query processing, track
merging experiments, and reproducible evaluation workflows.

The repository includes two layers:

- a packaged runtime under `src/tmerge` for CLI-driven pipelines and training
- legacy research code under `videosys`, `e2e`, and `scripts` for reference and
  dataset-specific experiments

Naming note:

- `src/` is the source layout root
- `tmerge` is the modern package name
- `videosys` is a legacy namespace retained for migration

## Overview

TMerge is organized around YAML configuration files and a single CLI entrypoint.
The main workflows currently covered by the package are:

- running pipeline jobs
- exporting tracks and track features
- launching ReID training workflows

Optional integrations such as OpenMMLab and TorchReID are installed only when
needed.

## Installation

Install the core package:

```bash
python -m pip install -e .
```

Install development tools:

```bash
python -m pip install -e ".[dev]"
```

Install optional research integrations:

```bash
python -m pip install -e ".[dev,openmmlab,reid]"
```

The project requires Python 3.10 or newer.

## Quick Start

Run the example pipeline:

```bash
export TMERGE_INPUT_DIR=/path/to/images
export TMERGE_OUTPUT_DIR=/path/to/output
tmerge run pipeline examples/pipeline.image-copy.yaml
```

Example configs are available in `examples/`:

- `examples/pipeline.image-copy.yaml`
- `examples/export.track-features.yaml`
- `examples/export.track-features-openmmlab.yaml`
- `examples/train.reid.yaml`

## How To Use

### Run a local smoke test

This verifies the new packaged runtime without any heavy external dependency:

```bash
export TMERGE_INPUT_DIR=/path/to/images
export TMERGE_OUTPUT_DIR=/tmp/tmerge-output
tmerge run pipeline examples/pipeline.image-copy.yaml
```

### Run the migrated MOT pipelines

The most common MOT workflows now live under `configs/mot/`.

Set the shared runtime variables:

```bash
export TMERGE_DEVICE=cuda:0
export TMERGE_MOT_INPUT=/path/to/MOT17/train/MOT17-11-DPM/img1
export TMERGE_MMDET_CONFIG=/path/to/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
export TMERGE_MMDET_CHECKPOINT=/path/to/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
export TMERGE_OUTPUT=/tmp/MOT17-11-DPM-faster_rcnn-sort.txt
```

Run SORT:

```bash
tmerge run pipeline configs/mot/mmdet_sort.yaml
```

Run DeepSORT:

```bash
tmerge run pipeline configs/mot/mmdet_deepsort.yaml
```

Run Tracktor:

```bash
tmerge run pipeline configs/mot/mmdet_tracktor.yaml
```

### Export track features

```bash
export TMERGE_INPUT_DIR=/path/to/images
export TMERGE_MOT_RESULT=/path/to/results.txt
export TMERGE_REID_MODEL=/path/to/model.pth
export TMERGE_OUTPUT_PKL=/tmp/track-features.pkl
tmerge export features examples/export.track-features.yaml
```

If you want the older OpenMMLab-style ReID feature extraction path:

```bash
export TMERGE_DEVICE=cuda:0
tmerge export features examples/export.track-features-openmmlab.yaml
```

### Generate a plain video from images

```bash
tmerge export video \
  --data /path/to/image-folder \
  --output /tmp/output.mp4 \
  --fps 30
```

### Train ReID

```bash
tmerge train reid examples/train.reid.yaml
```

Canonical training configs now live under `configs/training/reid/`.
The older `config/train/reid/` directory should be treated as legacy migration material.

### Evaluate, visualize, and export MOT results

Evaluate a result file against MOT ground truth:

```bash
tmerge mot evaluate \
  --ground-truth /path/to/MOT17/train/MOT17-11-DPM/gt/gt.txt \
  --result /tmp/MOT17-11-DPM-faster_rcnn-sort.txt
```

Visualize tracking results on images or video:

```bash
tmerge mot visualize \
  --data /path/to/MOT17/train/MOT17-11-DPM/img1 \
  --result /tmp/MOT17-11-DPM-faster_rcnn-sort.txt \
  --start-frame 1 \
  --wait-ms 100
```

Export a rendered MOT video:

```bash
tmerge mot export-video \
  --data /path/to/MOT17/train/MOT17-11-DPM/img1 \
  --result /tmp/MOT17-11-DPM-faster_rcnn-sort.txt \
  --output /tmp/MOT17-11-DPM-faster_rcnn-sort.mp4 \
  --fps 30
```

Filter MOT results before downstream ReID or analysis:

```bash
tmerge mot filter \
  --result /tmp/MOT17-11-DPM-faster_rcnn-sort.txt \
  --output /tmp/MOT17-11-DPM-faster_rcnn-sort.filtered.txt \
  --image-dir /path/to/MOT17/train/MOT17-11-DPM/img1 \
  --min-frames 100 \
  --drop-border-boxes
```

Compute track duration statistics:

```bash
tmerge mot statistics \
  --result /tmp/MOT17-11-DPM-faster_rcnn-sort.txt \
  --output-csv /tmp/MOT17-11-DPM-stats.csv \
  --output-plot /tmp/MOT17-11-DPM-stats.png
```

Generate a batch evaluation report:

```bash
tmerge mot batch-report \
  --ground-truth /path/to/MOT17/train/MOT17-11-DPM/gt/gt.txt \
  --method sort=/tmp/sort.txt \
  --method deepsort=/tmp/deepsort.txt \
  --output-csv /tmp/MOT17-11-DPM-report.csv
```

### Override config values from the CLI

```bash
tmerge run pipeline configs/mot/mmdet_sort.yaml \
  --set pipeline.name=mot-debug \
  --set pipeline.operators.2.tracker.match_iou_thr=0.4 \
  --set pipeline.operators.3.path=/tmp/debug-output.txt
```

## CLI

The package exposes the `tmerge` command:

```bash
tmerge run pipeline <config.yaml>
tmerge mot evaluate --ground-truth <gt.txt> --result <result.txt>
tmerge mot visualize --data <img-dir-or-video> --result <result.txt>
tmerge mot export-video --data <img-dir-or-video> --result <result.txt> --output <out.mp4>
tmerge mot filter --result <result.txt> --output <filtered.txt> [--image-dir <img-dir>]
tmerge mot statistics --result <result.txt> [--output-csv <stats.csv>] [--output-plot <stats.png>]
tmerge mot batch-report --ground-truth <gt.txt> --method <name=path> [--method <name=path> ...]
tmerge export tracks <config.yaml>
tmerge export features <config.yaml>
tmerge export video --data <img-dir> --output <out.mp4> [--fps 30]
tmerge train reid <config.yaml>
```

Configuration values can be overridden from the command line:

```bash
tmerge run pipeline config.yaml --set pipeline.name=debug --set pipeline.operators.1.report_interval=10
```

## Configuration

Pipeline configs are YAML documents with an ordered operator list. A minimal
pipeline looks like this:

```yaml
version: 1
pipeline:
  name: copy-images
  operators:
    - type: source.image_folder
      path: /data/images
    - type: reporter.progress
      report_interval: 50
    - type: sink.image_folder
      path: /tmp/output
```

Common operator groups include:

- source operators for images and videos
- reporting operators for runtime progress
- IO helpers for MOT result loading
- sink operators for frames, videos, results, and features
- integration adapters for external research tooling

Configs are validated before execution. The validator catches issues such as:

- missing `pipeline` or `training` sections
- empty `pipeline.operators`
- first operator not being a `source.*`
- unknown operator types
- missing required fields like `path`, `config_file`, or `checkpoint_file`

When validation fails, the CLI reports field-level errors and exits before the
runtime starts.

## Repository Layout

- `src/tmerge/`: packaged runtime, CLI, config loader, operators, and training
- `configs/`: maintained YAML workflows, including migrated MOT pipelines
- `configs/training/`: maintained training configs for the modern CLI
- `examples/`: sample YAML configs
- `tests/`: unit and integration tests for the packaged workflow
- `docs/appendix/`: dependency and migration notes
- `legacy/`: index and policy notes for historical assets
- `docs/*.md`: dataset and script cheat sheets
- `videosys/`, `e2e/`, `scripts/`: legacy research and experiment code

The important convention is:

- build new things in `src/tmerge/`
- migrate stable workflows into `configs/`
- treat `videosys/`, `e2e/`, and `scripts/` as legacy unless explicitly being ported

## Development

Run the test suite:

```bash
pytest
```

Run the main checks:

```bash
ruff check .
mypy src
```

## Documentation

- `docs/appendix/dependencies.md`: dependency layout and install options
- `docs/appendix/migration.md`: notes on the package-first workflow
- `docs/appendix/project-layout.md`: directory responsibilities and naming guidance
- `docs/appendix/legacy-e2e-tools.md`: which old e2e tools are now superseded
- `docs/mot_cheatsheet.md`: MOT-related commands
- `docs/kitti_cheatsheet.md`: KITTI-related commands
- `docs/pathtrack_cheatsheet.md`: PathTrack-related commands
- `docs/e2e_cheatsheet.md`: end-to-end workflow notes
- `docs/script_cheatsheet.md`: script usage reference
- `docs/trouble_shooting.md`: troubleshooting notes

## MOT Migration Map

The following legacy experiment configs now have YAML replacements:

- `e2e/configs/tracking/mmt_sort_private.py` -> `configs/mot/mmdet_sort.yaml`
- `e2e/configs/tracking/mmt_deepsort_private.py` -> `configs/mot/mmdet_deepsort.yaml`
- `e2e/configs/tracking/mmt_tracktor_private.py` -> `configs/mot/mmdet_tracktor.yaml`
- `config/train/reid/*.yaml` -> `configs/training/reid/*.yaml`

The following legacy tools now have direct CLI replacements:

- `e2e/configs/tools/gen_mot_result_video.py` -> `tmerge mot export-video`
- `e2e/configs/tools/visualize_mot_result.py` -> `tmerge mot visualize`
- `e2e/configs/tools/gen_track_features.py` -> `tmerge export features examples/export.track-features-openmmlab.yaml`
- `e2e/configs/tools/gen_track_features_torchreid.py` -> `tmerge export features examples/export.track-features.yaml`
- `e2e/configs/tools/gen_video_from_images.py` -> `tmerge export video`
- `e2e/simple/mot_eval.py` -> `tmerge mot evaluate`
- `scripts/mot/reid/filter_track_results.py` -> `tmerge mot filter`
- `scripts/mot/cal_statistics.py` -> `tmerge mot statistics`
- `scripts/mot/eval_mot_methods.py` -> `tmerge mot batch-report`

## Citation

If you use this repository in academic work, please cite:

```bibtex
@inproceedings{chao2023track,
  author = {Daren Chao and Yueting Chen and Nick Koudas and Xiaohui Yu},
  title = {Track Merging for Effective Video Query Processing},
  booktitle = {2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages = {164--176},
  year = {2023},
  doi = {10.1109/ICDE55515.2023.00020}
}
```
