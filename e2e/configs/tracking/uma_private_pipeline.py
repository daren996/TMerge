from videosys.ingestion.tracking.uma_mot import UMAMOT
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
        MMDetDetectorPipeline(args.config, args.checkpoint, 
            detect_classes=['person']),
        UMAMOT(
            siamese_checkpoint='../UMA-MOT/models/npair0.1-id0.1-se_block2'
        ),
        ObjectTrackingVisualizer(display=True)
    ]
