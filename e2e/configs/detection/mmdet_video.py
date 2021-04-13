from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.objectdetection.mmdet import MMDetObjectDetector
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer

args = dict(
    file='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    # pylint: disable=line-too-long
    config_file = '../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
    # pylint: disable=line-too-long
    checkpoint_file = '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
)

operators = [
    VideoSource(args['file']),
    MMDetObjectDetector(args['config_file'], args['checkpoint_file']),
    ObjectDetectionVisualizer(threshold=0.3)
]