from videosys.ingestion.io.tracking import MOTResultSink
from videosys.ingestion.formatting import DetToTrackResult
from e2e.configs.utils import create_source_for_path
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer

description = "use mmdet object detection pipeline to detect objects"

default_args = dict(
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    # pylint: disable=line-too-long
    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    # pylint: disable=line-too-long
    checkpoint = '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    output='../storage/results/mot17/MOT17-11-DPM/faster_rcnn-det.txt',
    threshold='0.5',
    classes='',
    no_report_save = False,
    display= False,
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def operators(args):
    detect_classes = None if args.classes == '' else args.classes.split(',')
    ops =  [
        create_source_for_path(args.path),
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
        MMDetDetectorWithFeatures(config_file = args.config, 
            checkpoint_file=args.checkpoint, 
            detect_classes=detect_classes),
    ]
    if not args.output == '':
        ops.append(DetToTrackResult(float(args.threshold)))
        ops.append(MOTResultSink(args.output))
    if args.display:
        ops.append(ObjectDetectionVisualizer(display=True, threshold=float(args.threshold)))
    ops.append(ProgressReporter(save_file = args.output+'.report' 
            if not args.no_report_save else None))
    return ops
