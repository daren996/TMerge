from centertrack.detector import Detector
from centertrack.opts import opts

from videosys.ingestion.base import Operator
from videosys.ingestion.data import ObjectDetectionResult, ObjectTrackingResult
from videosys.ingestion import fields

class CenterTrackTracking(Operator):

    def prepare(self):
        self.tracker = Detector(opts().init(
            args = [
                'tracking',
                '--load_model',
                'models/centertrack/coco_tracking.pth',
            ]
        ))
        self.context.put(fields.META_OBJECT_DETECTION_CLASSES,
            self.tracker.debugger.names)
    
    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        det_result = self.tracker.run(frame)
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        bbox_result = det_result['results']

        track_result = []
        detection_result = []
        for r in bbox_result:
            detection_result.append(ObjectDetectionResult(r['bbox'], 
                r['class'], r['score']))
            track_result.append(ObjectTrackingResult(r['tracking_id'], 
                r['class'], r['bbox'], r['score'], detection_result[-1]))
        
        tables[fields.DATA_OBJECT_DETECTION] = detection_result
        tables[fields.DATA_OBJECT_TRACK] = track_result
        self.collector.emit(tables)
        