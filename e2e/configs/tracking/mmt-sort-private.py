from videosys.ingestion.tracking.mmtracking import MMTrackingSORT
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorPipeline
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer

default_args = dict(
    video='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    config='../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
    # pylint: disable=line-too-long
    checkpoint='../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def operators(args):
    return [
        VideoSource(args.video),
        MMLibCompatable(),
        MMDetDetectorPipeline(args.config, args.checkpoint),
        MMMultiScaleFlipAug([
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ], img_scale=(1088, 1088), flip=False),
        MMLibMoveData(),
        MMTrackingSORT(
            motion=dict(type='KalmanFilter', center_only=False),
            tracker=dict(type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None)
        ),
        ObjectTrackingVisualizer(display=True)
    ]
