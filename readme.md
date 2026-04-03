# TMerge

This repository contains the code for the ICDE 2023 paper
`Track Merging for Effective Video Query Processing`.

Paper authors:

- Daren Chao
- Yueting Chen
- Nick Koudas
- Xiaohui Yu

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

## Setup

This repository builds on the `mmdetection` project for object detection:
<https://github.com/open-mmlab/mmdetection>

Expected parent folder structure:

```text
- parent_folder:
    - TMerge (this project)
    - mmdetection (clone from the mmdetection project)
    - mmtracking (clone from the mmtracking project)
    - CenterNet (clone from https://github.com/yestinchen/CenterNet)
    - CenterTrack (clone from https://github.com/yestinchen/CenterTrack)
    - UMA-MOT (clone from https://github.com/yestinchen/UMA-MOT)
    - storage (folder to store datasets, results, etc.)
        - dataset (dataset path; we recommend using soft links to the actual datasets)
            - MOT17 (MOT17 challenge dataset)
        - results (folder to store results)
```

For models and checkpoints from `mmdetection`, please download them to the
corresponding folders manually before use.

## Dependencies

Please use the following versions:

- `mmcv`: `>=1.3.8, <1.4`
- `torch`: `1.9`
- `mmdetection`: `tags/2.15.1`
- `mmtracking`: forked version, branch `v0.6.0-mod`
