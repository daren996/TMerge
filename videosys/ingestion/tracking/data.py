from dataclasses import dataclass, field

from videosys.ingestion.data import ObjectTrackingResult

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


def tracklet_to_result(tracklet, updated_bbox=None):
    bbox = tracklet.bbox if updated_bbox is None else updated_bbox
    if tracklet.payload is not None:
        return ObjectTrackingResult(tracklet.uid, tracklet.payload.label, 
            bbox, tracklet.payload.confidence, tracklet.payload)
    else:
        return ObjectTrackingResult(tracklet.uid, -1, 
            bbox, -1)
