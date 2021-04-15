import os
from videosys.ingestion.io.base import ImageSource, VideoSource

def create_source_for_path(path):
    return ImageSource(path) if os.path.isdir(path) else VideoSource(path)
