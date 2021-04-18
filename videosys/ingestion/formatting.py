from videosys.ingestion.base import Operator
from videosys.ingestion.data import ObjectTrackingResult
from videosys.ingestion import fields

class DetToTrackResult(Operator):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def process(self, tables):
        dets = tables[fields.DATA_OBJECT_DETECTION]
        tracks = []
        for det in dets:
            if det.confidence >= self.threshold:
                tracks.append(ObjectTrackingResult(-1, 
                    det.label, det.bbox, det.confidence, det))
        tables[fields.DATA_OBJECT_TRACK] = tracks
        self.collector.emit(tables)
