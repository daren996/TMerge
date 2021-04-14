import cv2
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class SelectROI(Operator):

    def __init__(self, window_name="select_roi"):
        super().__init__()
        self.window_name = window_name
    
    def prepare(self):
        self.init_bbox = None

    def process(self, tables):
        if self.init_bbox is None:
            frame = tables[fields.DATA_FRAME]
            init_bbox = list(cv2.selectROI(self.window_name, frame, False, False))
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]
            self.init_bbox = init_bbox
            cv2.destroyWindow(self.window_name)
        tables[fields.SELECT_ROI] = self.init_bbox
        self.collector.emit(tables)
        