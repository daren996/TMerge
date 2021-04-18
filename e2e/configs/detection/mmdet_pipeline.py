from e2e.configs.utils import create_source_for_path
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.objectdetection.mmdet import MMDetDetectorPipeline
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer

description = "use mmdet object detection pipeline to detect objects"

default_args = dict(
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    # pylint: disable=line-too-long
    config_file = '../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
    # pylint: disable=line-too-long
    checkpoint_file = '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth',
    threshold='0.5',
)

def operators(args):
    ops =  [
        create_source_for_path(args.path),
        MMDetDetectorPipeline(args.config_file, args.checkpoint_file),
    ]
    if args.display:
        ops.append(ObjectDetectionVisualizer(display=True, threshold=float(args.threshold)))
    ops.append(ProgressReporter(save_file = args.output+'.report' 
            if not args.no_report_save else None))
    return ops
