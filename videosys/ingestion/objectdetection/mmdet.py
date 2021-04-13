import logging

import numpy as np
from mmdet.apis import init_detector, inference_detector

from videosys.ingestion import fields

from videosys.ingestion.data import ObjectDetectionResult
from ..base import Operator

class MMDetObjectDetector(Operator):
    '''
    use mmdet lib to perform object detection.
    supported models can be found at
        https://mmdetection.readthedocs.io/en/latest/modelzoo_statistics.html
    '''
    def __init__(self, config_file, checkpoint_file, device='cuda:0', 
            detect_classes=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.detect_classes = detect_classes
        super().__init__()

    def prepare(self):
        config_file = self.config_file
        checkpoint_file = self.checkpoint_file
        device = self.device
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.context.put(fields.SHARED_MMDET_MODEL, self.model)
        self.context.put(fields.META_OBJECT_DETECTION_CLASSES, self.model.CLASSES)

        if self.detect_classes is not None:
            ignored_classes = [c for c in self.detect_classes if c not in self.model.CLASSES]
            if len(ignored_classes) > 0:
                logging.warning("this model cannot detect class: %s", ignored_classes)

            self.detect_classes = [self.model.CLASSES.index(c) for c in 
                set(self.detect_classes) - set(ignored_classes)]
    
    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        result = inference_detector(self.model, frame)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        # result dimension: label -> (x, 5)
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        assert len(bboxes[0]) == 5, 'expecting a confidence value from the NN'
        labels = np.concatenate(labels)
        # print('bbox:',bboxes[0], labels[0])
        obj_result = []
        for bbox, label in zip(bboxes, labels):
            if self.detect_classes is None or label in self.detect_classes:
                obj_result.append(ObjectDetectionResult(bbox[:-1], label, bbox[-1]))
        tables[fields.DATA_OBJECT_DETECTION] = obj_result
        self.collector.emit(tables)
