
import logging
import cv2

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.utils.visualize_utils import draw_bbox_and_labels


class ObjectDetectionVisualizer(Operator):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.window_name = 'detect_vis'
        self.__threshold = self._get_config('threshold', 0.3)

    def prepare(self):
        if self.context.has(fields.META_OBJECT_DETECTION_CLASSES):
            self.__class_names = self.context.get(fields.META_OBJECT_DETECTION_CLASSES)
        else:
            self.__class_names = None
            logging.warning("warning: NO CLASSES given, will use label values")

    def __extract_id(self, detect_result):
        return detect_result.label

    def __extract_bbox(self, detect_result):
        return detect_result.bbox

    def __generate_label(self, detect_result):
        if self.__class_names is None:
            return '({}):{:0.2f}'.format(detect_result.label, detect_result.confidence)
        return '{}:{:0.2f}'.format(self.__class_names[detect_result.label], 
            detect_result.confidence)

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        result = tables[fields.DATA_OBJECT_DETECTION]
        result = [r for r in result if r.confidence >= self.__threshold]
        image = draw_bbox_and_labels(frame, result, self.__extract_bbox, 
            self.__generate_label, id_func=self.__extract_id)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1000)
