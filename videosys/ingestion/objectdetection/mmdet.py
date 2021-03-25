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
    
    def prepare(self):
        config_file = self._get_config('config_file')
        checkpoint_file = self._get_config('checkpoint_file')
        device = self._get_config('device', 'cuda:0')
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.context.put(fields.SHARED_MMDET_MODEL, self.model)
        self.context.put(fields.META_OBJECT_DETECTION_CLASSES, self.model.CLASSES)
    
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
            obj_result.append(ObjectDetectionResult(bbox[:-1], label, bbox[-1]))
        tables[fields.DATA_OBJECT_DETECTION] = obj_result
        self.collector.emit(tables)
