from videosys.ingestion.visualize.image import ImageVisualizer
from videosys.ingestion.io.base import VideoSource

args = dict(
    video='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    auto_play=True
)

operators = [
    VideoSource(args['video']),
    ImageVisualizer(args['auto_play'])
]
