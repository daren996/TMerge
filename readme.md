# TMerge

TMerge is a research engineering toolkit for video query processing, track
merging experiments, and reproducible evaluation workflows.

The repository includes two layers:

- a packaged runtime under `src/tmerge` for CLI-driven pipelines and training
- legacy research code under `videosys`, `e2e`, and `scripts` for reference and
  dataset-specific experiments

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

### Train ReID

```bash
tmerge train reid examples/train.reid.yaml
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
tmerge export tracks <config.yaml>
tmerge export features <config.yaml>
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
- `examples/`: sample YAML configs
- `tests/`: unit and integration tests for the packaged workflow
- `docs/appendix/`: dependency and migration notes
- `docs/*.md`: dataset and script cheat sheets
- `videosys/`, `e2e/`, `scripts/`: legacy research and experiment code

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
