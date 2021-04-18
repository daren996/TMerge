from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.tracking.mmtracking import MMTrackingMOT

def mmtracking(video_path, config_file):
    print('tracking video', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource(video_path))
    builder.add_operator(MMTrackingMOT(config_file))
    builder.build().start()

if __name__ == '__main__':
    mmtracking(
        '../storage/dataset/videos/MOT16-03.mp4',
        '../mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py',
    )
