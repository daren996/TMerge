from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.visualize.image import ImageVisualizer
from videosys.ingestion.io.base import ImageSource

# args: {key=(des, default_value)}

description = "play images from folder"

default_args = dict(
    folder='../storage/dataset/MOT17/train/MOT17-02-DPM/img1',
    no_auto_play=False
)

def operators(args):
    return [
        ImageSource(args.folder),
        ImageVisualizer(not args.no_auto_play),
        ProgressReporter()
    ]
