import logging
from centertrack.detector import Detector
from centertrack.opts import opts

from videosys.ingestion.base import Operator
from videosys.ingestion.data import ObjectDetectionResult, ObjectTrackingResult
from videosys.ingestion import fields

class CenterTrackTracking(Operator):
    def __init__(self, model_path, detect_classes= None, num_class=None):
        super().__init__()
        self.model_path = model_path
        self.detect_classes = detect_classes
        self.num_class = num_class

    def prepare(self):
        extra_params = []
        if self.num_class is not None:
            extra_params += ['--num_class', str(self.num_class)]
        self.tracker = Detector(opts().init(
            args = [
                'tracking',
                '--load_model',
                self.model_path,
                *extra_params
            ]
        ))
        classes = self.tracker.debugger.names
        self.context.put(fields.META_OBJECT_DETECTION_CLASSES,
            classes)

        if self.detect_classes is not None:
            ignored_classes = [c for c in self.detect_classes if c not in classes]
            if len(ignored_classes) > 0:
                logging.warning("this model cannot detect class: %s", ignored_classes)

            self.detect_classes = [classes.index(c) for c in 
                set(self.detect_classes) - set(ignored_classes)]
    
    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        det_result = self.tracker.run(frame)
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        
        bbox_result = det_result['results']

        track_result = []
        detection_result = []
        for r in bbox_result:
            clazz = r['class'] -1
            if self.detect_classes is None or clazz in self.detect_classes:
                detection_result.append(ObjectDetectionResult(r['bbox'], 
                    clazz, r['score']))
                track_result.append(ObjectTrackingResult(r['tracking_id'], 
                    clazz, r['bbox'], r['score'], detection_result[-1]))
        
        tables[fields.DATA_OBJECT_DETECTION] = detection_result
        tables[fields.DATA_OBJECT_TRACK] = track_result
        self.collector.emit(tables)
        