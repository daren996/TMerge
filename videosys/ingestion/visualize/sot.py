import cv2

from videosys.ingestion import fields
from videosys.ingestion.base import Operator
from videosys.utils.visualize_utils import draw_bbox_and_labels

class SOTVisualizer(Operator):

    def __init__(self, window_name='sot_vis'):
        super().__init__()
        self.window_name = window_name

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        result = tables[fields.DATA_SOT_RESULT]
        image = draw_bbox_and_labels(frame, [result], lambda x: x.bbox, 
            lambda x: '{:.2f}'.format(x.score) if x.score is not None else 'None', 
            id_func=lambda _:1)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(100)
