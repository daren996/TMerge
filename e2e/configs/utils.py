import os
from videosys.ingestion.io.base import ConcatMultiImageSource, ImageSource, VideoSource

def create_source_for_path(path):
    if ':' in path:
        # multi-images
        return ConcatMultiImageSource(path.split(':'))
    return ImageSource(path) if os.path.isdir(path) else VideoSource(path)
