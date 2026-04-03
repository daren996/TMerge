# pylint: disable-all
"""Legacy tool config.

Preferred modern path:
    tmerge export features examples/export.track-features.yaml
"""

from videosys.ingestion.reid.io import TrackFeatureSink
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.reid.bgs import BackgroundSubtraction
from videosys.ingestion.reid.torchreid import TorchReIdFeatureExtracor
from videosys.ingestion.io.loaders import MOTResultLoader
from e2e.configs.utils import create_source_for_path


default_args = dict(
    data_path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    result_path = '../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person.txt',
    feature_save_path = '../storage/results/mot17/MOT17-11-DPM/feats-raw/faster_rcnn-deepsort-person-feat.pkl',
    model_name = 'osnet_x1_0',
    model_path = ''
)

def operators(args):
    ops = [
        create_source_for_path(args.data_path),
        MOTResultLoader(args.result_path),
        # BackgroundSubtraction(),  # BGS
        TorchReIdFeatureExtracor(args.model_name, args.model_path),
        TrackFeatureSink(args.feature_save_path),
        ProgressReporter()
    ]
    return ops
