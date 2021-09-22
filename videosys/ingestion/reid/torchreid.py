import os
from torchreid.utils import FeatureExtractor
import cv2
import torch
from PIL import Image

from videosys.ingestion.base import Operator
from videosys.ingestion.data import TrackFeature
from videosys.ingestion import fields

class TorchReIdFeatureExtracor(Operator):
    def __init__(self, model_name, model_path, device='cuda'):
        super(TorchReIdFeatureExtracor, self).__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
    
    def prepare(self):
        self.extractor = FeatureExtractor(model_name = self.model_name, 
                                          model_path = self.model_path, device = self.device)
    
    def process(self, tables):
        image = tables[fields.DATA_FRAME]
        tracks = tables[fields.DATA_OBJECT_TRACK]

        bboxes = []
        ids = []
        for track in tracks:
            bboxes.append(track.bbox)
            ids.append(track.uid)

        if len(bboxes) == 0:
            tables[fields.DATA_TRACK_FEAT] = []
            self.collector.emit(tables)
            return
            
        bboxes = torch.tensor(bboxes)

        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=len(image[0]))
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=len(image))

        croped_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 == x1:
                x2 = x1 + 1
            if y2 == y1:
                y2 = y1 + 1
            croped_image = image[y1: y2, x1:x2]
            # covert BGR to RGB
            croped_image = cv2.cvtColor(croped_image, cv2.COLOR_BGR2RGB)
            croped_images.append(croped_image)
        # crop images.
        features = self.extractor(croped_images)

        result = []
        for t, feat in zip(tracks, features):
            result.append(TrackFeature(t.uid, t.bbox, feat.cpu()))
        tables[fields.DATA_TRACK_FEAT] = result
        self.collector.emit(tables)
        