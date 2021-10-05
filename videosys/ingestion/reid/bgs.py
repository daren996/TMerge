"""
Background Subtraction Operator.
"""

import cv2

from videosys.ingestion.base import Operator
from videosys.ingestion import fields


class BackgroundSubtraction(Operator):
    def __init__(self, algo='MOG2'):
        super().__init__()
        self.algo = algo
    
    def prepare(self):
        if self.algo == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            self.backSub = cv2.createBackgroundSubtractorKNN()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    def process(self, tables):
        image = tables[fields.DATA_FRAME]
        fgmask = self.backSub.apply(image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)  # denoising
        frame_bgs = cv2.bitwise_and(image, image, mask=fgmask)
        tables[fields.DATA_FRAME] = frame_bgs  # substitude original image
        tables[fields.DATA_FRAME_BGS] = frame_bgs
        self.collector.emit(tables)
