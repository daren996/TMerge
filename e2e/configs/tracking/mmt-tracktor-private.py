
from videosys.ingestion.tracking.mmtracking import MMTrackingDeepSORT
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer

args = dict(
    video='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    # pylint: disable=line-too-long
    checkpoint = '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

operators = [
    VideoSource(args['video']),
    MMLibCompatable(),
    MMMultiScaleFlipAug([
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='VideoCollect', keys=['img'])
    ], img_scale=(1088, 1088), flip=False),
    MMLibMoveData(),
    MMDetDetectorWithFeatures(config_file = args['config'], checkpoint_file=args['checkpoint']),
    MMTrackingDeepSORT(
        pretrains=dict(
            # pylint: disable=line-too-long
            reid='https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        ),
        motion=dict(type='KalmanFilter', center_only=False),
        reid=dict(
            type='BaseReID',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3, ),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
            head=dict(
                type='LinearReIDHead',
                num_fcs=1,
                in_channels=2048,
                fc_channels=1024,
                out_channels=128,
                norm_cfg=dict(type='BN1d'),
                act_cfg=dict(type='ReLU'))),
        tracker=dict(
            type='SortTracker',
            obj_score_thr=0.5,
            reid=dict(
                num_samples=10,
                img_scale=(256, 128),
                img_norm_cfg=None,
                match_score_thr=2.0),
            match_iou_thr=0.5,
            momentums=None,
            num_tentatives=2,
            num_frames_retain=100)
    ),
    ObjectTrackingVisualizer(display=True)
]
