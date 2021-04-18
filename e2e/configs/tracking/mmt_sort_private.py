from e2e.configs.utils import create_source_for_path
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.io.tracking import MOTResultSink
from videosys.ingestion.tracking.mmtracking import MMTrackingSORT
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer

default_args = dict(
    # path='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    # pylint: disable=line-too-long
    checkpoint = '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    display=False,
    # output='../storage/results/mot16/MOT16-03-faster_rcnn-sort.txt',
    output='../storage/results/mot17/MOT17-11-DPM-faster_rcnn-sort.txt',
    no_report_save = False
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def operators(args):
    ops = [
        create_source_for_path(args.path),
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
        MMDetDetectorWithFeatures(config_file = args.config, checkpoint_file=args.checkpoint),
        MMTrackingSORT(
            motion=dict(type='KalmanFilter', center_only=False),
            tracker=dict(type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None)
        ),
    ]
    if not args.output == '':
        ops.append(MOTResultSink(args.output))
    if args.display:
        ops.append(ObjectTrackingVisualizer(display=True))
    ops.append(ProgressReporter(save_file = args.output+'.report' 
            if not args.no_report_save else None))
    return ops
