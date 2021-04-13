from videosys.ingestion.visualize.image import ImageVisualizer
from videosys.ingestion.io.base import ImageSource

args = dict(
    folder='/media/ytchen/hdd/dataset/2DMOT2015/test/ADL-Rundle-1/img1',
    auto_play=True
)

operators = [
    ImageSource(args['folder']),
    ImageVisualizer(args['auto_play'])
]
