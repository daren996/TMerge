from videosys.ingestion.visualize.sot import SOTVisualizer
from videosys.ingestion.io.base import VideoSource
from videosys.ingestion.tracking.sot.base import SelectROI
from videosys.ingestion.tracking.sot.mmtracking import MMTrackSOTPipeline

default_args = dict(
    video='../storage/dataset/videos/MOT16-03.mp4',
    config_file='../mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py',
    checkpoint_file='../storage/models/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth',
)

def operators(args):
    return [
        VideoSource(args.video),
        SelectROI(),
        MMTrackSOTPipeline(args.config_file, args.checkpoint_file),
        SOTVisualizer()
    ]
