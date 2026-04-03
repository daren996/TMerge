"""Legacy tool config.

Superseded by:
    tmerge mot export-video --data ... --result ... --output ...
"""

from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.io.loaders import MOTResultLoader
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer
from videosys.ingestion.io.base import VideoSink
from videosys.ingestion import fields
from e2e.configs.utils import create_source_for_path

default_args = dict(
    data_path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    result_path = None,
    output_folder = None,
    output_name = None,
    fps = '',
    thickness = '3',
    font_scale='1',
)

def operators(args):
    ops = [
        create_source_for_path(args.data_path),
        MOTResultLoader(args.result_path),
        ObjectTrackingVisualizer(display=False, thickness=int(args.thickness), 
            font_scale = float(args.font_scale)),
        VideoSink(args.output_folder, args.output_name, fields.DATA_FRAME_TRACK, 
            None if args.fps == '' else int(args.fps)),
        ProgressReporter()
    ]
    return ops
