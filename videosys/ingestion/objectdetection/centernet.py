from centernet.detectors.ctdet import CtdetDetector
from centernet.utils.debugger import coco_class_name
from centernet.opts import opts

from videosys.ingestion.data import ObjectDetectionResult
from videosys.ingestion.base import Operator
from videosys.ingestion import fields


class CenterNetObjectDetector(Operator):

    def prepare(self):
        self.detector = CtdetDetector(opts().init(
            args = ['ctdet',
             '--load_model',
             'models/centernet/ctdet_coco_dla_2x.pth']
        ))
        # coco class.
        self.context.put(fields.META_OBJECT_DETECTION_CLASSES, coco_class_name)

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        det_result = self.detector.run(frame)

        # return dict (key:label 1~n) -> (x,5)
        bbox_result = det_result['results']
        
        obj_result = []
        for label, bboxes in bbox_result.items():
            for bbox in bboxes:
                obj_result.append(ObjectDetectionResult(bbox[:-1], label-1, bbox[-1]))
        
        tables[fields.DATA_OBJECT_DETECTION] = obj_result
        self.collector.emit(tables)
