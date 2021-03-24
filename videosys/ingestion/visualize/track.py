import cv2

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.utils.visualize_utils import draw_bbox_and_labels

class ObjectTrackingVisualizer(Operator):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.window_name = 'track_vis'

    def prepare(self):
        self.__class_names = self.context.get(fields.META_OBJECT_DETECTION_CLASSES)
    
    def __extract_bbox(self, tracklet):
        return tracklet.payload.bbox

    def __generate_label(self, tracklet):
        return '{}/{}:{:0.2f}'.format(tracklet.uid,
            self.__class_names[tracklet.payload.label], 
            tracklet.payload.confidence)

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        result = tables[fields.DATA_OBJECT_TRACK]
        # visualize.
        image = draw_bbox_and_labels(frame, result, self.__extract_bbox, 
            self.__generate_label)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(100)
