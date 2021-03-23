'''
data classes
'''
from dataclasses import dataclass

@dataclass
class VideoMeta:
    fps: int
    width: int
    height: int
    frame_count: int

@dataclass
class ImageFolderMeta:
    total: int
