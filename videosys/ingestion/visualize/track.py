import logging
import cv2

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.utils.visualize_utils import draw_bbox_and_labels

class ObjectTrackingVisualizer(Operator):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.window_name = 'track_vis'
        self.__visualize_track = self._get_config('visualize_track', True)
        self.__visualize_track_gt = self._get_config('visualize_track_gt', False)

    def prepare(self):
        if self.context.has(fields.META_OBJECT_DETECTION_CLASSES):
            self.__class_names = self.context.get(fields.META_OBJECT_DETECTION_CLASSES)
        else:
            self.__class_names = None
            logging.warning("warning: NO CLASSES given, will use label values")

    def __extract_bbox(self, tracklet):
        return tracklet.bbox

    def __extract_id(self, tracklet):
        if tracklet.label >=0:
            return tracklet.label
        return tracklet.uid 

    def __generate_label(self, tracklet, is_gt = False):
        if is_gt:
            return '{}'.format(tracklet.uid)
        if self.__class_names is None:
            return '{}/({})\n{:0.2f}'.format(tracklet.uid, 
                tracklet.payload.label, tracklet.payload.confidence)
        return '{}/{}\n{:0.2f}'.format(tracklet.uid,
            self.__class_names[tracklet.payload.label], 
            tracklet.payload.confidence)
    

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        if self.__visualize_track:
            result = tables[fields.DATA_OBJECT_TRACK]
            # visualize.
            image = draw_bbox_and_labels(frame, result, self.__extract_bbox, 
                self.__generate_label, id_func=self.__extract_id)
        if self.__visualize_track_gt:
            result = tables[fields.DATA_OBJECT_TRACK_GT]
            image = draw_bbox_and_labels(frame, result, self.__extract_bbox, 
                lambda x: self.__generate_label(x, True), id_func=self.__extract_id)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(100)
