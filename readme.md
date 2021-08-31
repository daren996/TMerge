
# Setup

This repository uses the mmdetection project for object detection.
(https://github.com/open-mmlab/mmdetection)

The parent folder structure:

```
- parent_folder:
    - VideoSystem (this project)
    - mmdetection (clone the repository from mmdetection project).
    - mmtracking (clone from the repository from mmtracking project).
    - CenterNet (clone from `https://github.com/yestinchen/CenterNet`)
    - CenterTrack (clone from `https://github.com/yestinchen/CenterTrack`)
    - UMA-MOT (clone from `https://github.com/yestinchen/UMA-MOT`)
    - storage (folder to store all datasets, results, etc.)
        - dataset (dataset path, we recommand using soft links point to the actual datasets)
            - MOT17 (MOT17 challenge dataset)
        - results (folder to put results)
```

For models and checkpoints from `mmdetection` project, please download to the corresponding folders mannually before use.


**Requiremments for dependencies**
Please use the following versions:

mmcv: >=1.3.8, <1.4
torch: 1.9
mmdetection: tags/2.15.1
mmtracking: the forked version, branch: v0.6.0-mod