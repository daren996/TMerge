from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug

args = dict(
    file='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    # pylint: disable=line-too-long
    config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    # pylint: disable=line-too-long
    checkpoint_file = '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

operators = [
    VideoSource(args['file']),
    MMLibCompatable(),
    MMMultiScaleFlipAug([
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ], img_scale=(1088, 1088), flip=False),
    MMLibMoveData(),
    MMDetDetectorWithFeatures(config_file=args['config_file'], 
        checkpoint_file=args['checkpoint_file']),
    ObjectDetectionVisualizer(threshold=0.3)
]
