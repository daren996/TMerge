from videosys.ingestion.debugger.operators import SkipToFrame
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.io.loaders import MOTResultLoader
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer
from e2e.configs.utils import create_source_for_path

default_args = dict(
    data_path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    result_path = None,
    thickness = '3',
    font_scale='1',
    start_frame='1',
    image_prefix='',
    wait_ms='500'
)

def operators(args):
    ops = [
        create_source_for_path(args.data_path, args.image_prefix),
        MOTResultLoader(args.result_path),
        SkipToFrame(int(args.start_frame)),
        ObjectTrackingVisualizer(display=True, thickness=int(args.thickness), 
            font_scale = float(args.font_scale), wait_ms=int(args.wait_ms)),
        ProgressReporter()
    ]
    return ops
