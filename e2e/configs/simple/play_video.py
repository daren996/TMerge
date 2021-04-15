from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.visualize.image import ImageVisualizer
from videosys.ingestion.io.base import VideoSource

description = "play video"

default_args = dict(
    video=('video path', '/media/ytchen/hdd/dataset/videos/MOT16-03.mp4'),
    no_auto_play=('whether to play the video manully', False)
)

def operators(args):
    return [
        VideoSource(args.video),
        ImageVisualizer(not args.no_auto_play),
        ProgressReporter()
    ]
