from videosys.ingestion.tracking.mmtracking import MMTrackingMOT
from videosys.ingestion.metrics.mot_metrics import MOTMetricsReporter
from videosys.ingestion.io.loaders import MOTDetLoader, MOTGTLoader
from videosys.ingestion.io.base import ImageSource, VideoSource
from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.visualize.detection import ObjectDetectionVisualizer
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer
from videosys.ingestion.tracking.iou_tracker import IOUBatchTracker, IOUOnlineTracker
from videosys.ingestion.tracking.viou_tracker import VIOUOnlineTracker
from videosys.ingestion.tracking.sort_tracker import SORTOnlineTracker
from videosys.ingestion.tracking.deep_sort_tracker import DeepSORTOnlineTracker

def run_mot(source_path, mot_det):
    print('detect from:', source_path)
    builder = SimplePipelineBuilder()
    # builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(ImageSource({'folder': source_path}))
    builder.add_operator(MOTDetLoader({
        'folder_path': mot_det
    }))
    builder.add_operator(MMTrackingMOT({
        'config_file': 
            '../mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py',
        'checkpoint_file': ''
    }))
    # builder.add_operator(VIOUOnlineTracker())
    # builder.add_operator(IOUOnlineTracker())
    # builder.add_operator(IOUBatchTracker())
    # builder.add_operator(SORTOnlineTracker())
    # builder.add_operator(DeepSORTOnlineTracker())
    builder.add_operator(MOTGTLoader({
        'folder_path': mot_det
    }))
    # builder.add_operator(ObjectDetectionVisualizer({
    #     'threshold': 0
    # }))
    # builder.add_operator(ObjectTrackingVisualizer({
    #     'visualize_track': False,
    #     'visualize_track_gt': True
    # }))
    builder.add_operator(MOTMetricsReporter())
    builder.build().start()

if __name__ == '__main__':
    dataset = '/media/ytchen/hdd/dataset/MOT17/train/MOT17-11-DPM'
    # dataset = '/media/ytchen/hdd/dataset/MOT17/train/MOT17-02-DPM'
    run_mot('{}/img1'.format(dataset), dataset)
