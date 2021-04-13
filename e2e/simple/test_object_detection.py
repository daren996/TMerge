from videosys.ingestion.tracking.mmtracking import MMTrackingSORT
from videosys.ingestion.tracking.centertrack import CenterTrackTracking
from videosys.ingestion.objectdetection.centernet import CenterNetObjectDetector
from videosys.ingestion.tracking.sort_tracker import SORTOnlineTracker
from videosys.ingestion.tracking.deep_sort_tracker import DeepSORTOnlineTracker
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer
from videosys.ingestion.objectdetection.mmdet import MMDetObjectDetector
from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io.base import ImageSink, VideoSink, VideoSource
from videosys.ingestion.tracking.viou_tracker import VIOUOnlineTracker

from videosys.ingestion import fields

def detect_mmdet(video_path, config_file, checkpoint_file):
    print('detect video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(MMDetObjectDetector({
        'config_file': config_file,
        'checkpoint_file': checkpoint_file,
    }))
    builder.add_operator(ObjectDetectionVisualizer({
        'threshold': 0.3
    }))
    builder.build().start()

def detect_centernet(video_path):
    print('detect_video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(CenterNetObjectDetector())
    builder.add_operator(ObjectDetectionVisualizer({
        'threshold': 0.3
    }))
    builder.build().start()

def detect_mmdet_track(video_path, config_file, checkpoint_file):
    print('detect video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(MMDetObjectDetector({
        'config_file': config_file,
        'checkpoint_file': checkpoint_file,
        'classes': ['person']
    }))
    # builder.add_operator(IOUOnlineTracker())
    # builder.add_operator(VIOUOnlineTracker({
    #     'tracker': 'MIL'
    # }))
    # builder.add_operator(SORTOnlineTracker())
    # builder.add_operator(DeepSORTOnlineTracker())
    builder.add_operator(MMTrackingSORT())
    builder.add_operator(ObjectTrackingVisualizer({
        'display': True
    }))
    # builder.add_operator(ImageSink({
    #     'path': './output/test1',
    #     'image_key': fields.DATA_FRAME_TRACK
    # }))
    # builder.add_operator(VideoSink({
    #     'path': './output/test/',
    #     'name': 'out',
    #     'image_key': fields.DATA_FRAME_TRACK
    # }))
    builder.build().start()


def detect_centertrack(video_path):
    print('detect video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(CenterTrackTracking({
        # 'config_file': config_file,
        # 'checkpoint_file': checkpoint_file,
        # 'classes': ['person']
    }))
    # builder.add_operator(IOUOnlineTracker())
    # builder.add_operator(VIOUOnlineTracker({
    #     'tracker': 'MIL'
    # }))
    # builder.add_operator(SORTOnlineTracker())
    # builder.add_operator(DeepSORTOnlineTracker())
    builder.add_operator(ObjectTrackingVisualizer({
        'display': True
    }))
    # builder.add_operator(ImageSink({
    #     'path': './output/test1',
    #     'image_key': fields.DATA_FRAME_TRACK
    # }))
    # builder.add_operator(VideoSink({
    #     'path': './output/test/',
    #     'name': 'out',
    #     'image_key': fields.DATA_FRAME_TRACK
    # }))
    builder.build().start()

if __name__ == '__main__':
    # detect_mmdet(
    #     '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    #     '../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
    #     # pylint: disable=line-too-long
    #     '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    # )

    detect_mmdet_track(
        '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
        '../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
        # pylint: disable=line-too-long
        '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    )

    # detect_centernet(
    #     '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4'
    # )
    
    # detect_centertrack(
    #     '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4'
    # )
