'''
data classes
'''
from dataclasses import dataclass, field

# ============= meta classes
@dataclass
class VideoMeta:
    fps: int
    width: int
    height: int
    frame_count: int

@dataclass
class ImageFolderMeta:
    total: int


# ============= data classes


# --------- for object detection
@dataclass
class ObjectDetectionResult:
    # [x1, y1, x2,y2]
    bbox: list
    # type id
    label: int
    confidence: float
    # empty.
    accuracy: float = 0

class ObjectDetectionResultWithFeature(ObjectDetectionResult):
    feature: list

# ----------- for object tracking
@dataclass
class Tracklet:
    uid: int
    start_frame: int
    bboxes: list = field(default_factory=list)
    max_score: float = 0
    missing: int = 0
    count: int = 0
    payload: object = None

    def __getattr__(self, attr):
        if attr == 'bbox':
            return None if len(self.bboxes) == 0 else self.bboxes[-1]
        raise AttributeError()
