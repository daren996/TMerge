"""Legacy tool config.

Preferred modern path:
    tmerge export features examples/export.track-features-openmmlab.yaml
"""

from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
from videosys.ingestion.io.loaders import MOTResultLoader
from videosys.ingestion.reid.io import TrackFeatureSink
from videosys.ingestion.reid.mmlib import MMLibExtractFeature
from videosys.ingestion.observer.reporter import ProgressReporter
from e2e.configs.utils import create_source_for_path

default_args = dict(
    data_path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    result_path = '../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person.txt',
    feature_save_path = '../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person-feat.pkl'
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def operators(args):
    ops = [
        create_source_for_path(args.data_path),
        MOTResultLoader(args.result_path),
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
        MMLibExtractFeature(
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
                    num_classes=378,
                    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                    loss_pairwise=dict(
                        type='TripletLoss', margin=0.3, loss_weight=1.0),
                    norm_cfg=dict(type='BN1d'),
                    act_cfg=dict(type='ReLU')),
                img_scale=(256, 128), 
                rescale=True),
        TrackFeatureSink(args.feature_save_path),
        ProgressReporter()
    ]
    return ops
