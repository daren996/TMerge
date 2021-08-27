from e2e.configs.utils import create_source_for_path
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.io.tracking import MOTResultSink
from videosys.ingestion.tracking.mmtracking import MMTrackingTracktor
from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer

default_args = dict(
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    # pylint: disable=line-too-long
    checkpoint = '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    display=False,
    output='../storage/results/mot17/MOT17-11-DPM-faster_rcnn-tracktor.txt',
    no_report_save = False,
    obj_score_threshold='0.5',
    match_iou_threshold='0.5',
    num_samples='10',
    match_score_threshold='2.0',
    save_type=False,
    detect_classes='',
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
        MMDetDetectorWithFeatures(config_file = args.config, checkpoint_file=args.checkpoint, 
            detect_classes=None if len(args.detect_classes) == 0 else \
                args.detect_classes.replace('_',' ').split(',')),
        MMTrackingTracktor(
            pretrains=dict(
                # pylint: disable=line-too-long
                reid='https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
            ),
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
            motion=dict(
                type='CameraMotionCompensation',
                warp_mode='cv2.MOTION_EUCLIDEAN',
                num_iters=100,
                stop_eps=0.00001),
            tracker=dict(
                type='TracktorTracker',
                obj_score_thr=float(args.obj_score_threshold),
                regression=dict(
                    obj_score_thr=0.5,
                    nms=dict(type='nms', iou_threshold=0.6),
                    match_iou_thr=0.3),
                reid=dict(
                    num_samples=int(args.num_samples),
                    img_scale=(256, 128),
                    img_norm_cfg=None,
                    match_score_thr=float(args.match_score_threshold),
                    match_iou_thr=float(args.match_iou_threshold)),
                momentums=None,
                num_frames_retain=10)
        )
    ]
    if not args.output == '':
        ops.append(MOTResultSink(args.output, add_type_column=args.save_type))
    if args.display:
        ops.append(ObjectTrackingVisualizer(display=True))
    ops.append(ProgressReporter(save_file = args.output+'.report' 
            if not args.no_report_save else None))
    return ops
