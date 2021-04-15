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
    path: str

@dataclass
class ImageFolderMeta:
    total: int
    path: str


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
class ObjectTrackingResult:
    uid: int
    label: int
    bbox: list
    confidence: float
    payload: object = None # other data.
    

@dataclass
class SOTResult:
    bbox: list
    score: float
    