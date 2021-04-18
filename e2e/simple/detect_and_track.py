from videosys.ingestion.objectdetection.mmdet \
    import MMDetDetectorPipeline
from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.tracking.iou_tracker import IOUOnlineTracker
from videosys.ingestion.io.tracking import TrackResultFolderSink

def detect_and_track(video_path, config_file, checkpoint_file, result_folder):
    print('detect video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource(video_path))
    builder.add_operator(MMDetDetectorPipeline(config_file,checkpoint_file))
    builder.add_operator(IOUOnlineTracker(ttl=-1, iou_threshold= 0.5, min_conf= 0.4))
    builder.add_operator(TrackResultFolderSink(result_folder))
    builder.build().start()

if __name__ == '__main__':
    detect_and_track(
        '../storage/dataset/videos/MOT16-03.mp4',
        '../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
        # pylint: disable=line-too-long
        '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth',
        '../storage/result/MOT16-03/'
    )
