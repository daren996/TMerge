from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.tracking.mmtracking import MMTrackingMOT

def mmtracking(video_path, config_file, checkpoint_file):
    print('tracking video', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource({'file': video_path}))
    builder.add_operator(MMTrackingMOT({
        'config_file': config_file,
        'checkpoint_file': checkpoint_file
    }))
    builder.build().start()

if __name__ == '__main__':
    mmtracking(
        '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
        '../mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py',
        # pylint: disable=line-too-long
        '../mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    )
