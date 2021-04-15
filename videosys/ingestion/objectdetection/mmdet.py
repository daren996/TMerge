import logging
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.core import bbox2result
from mmdet.models import detectors

from videosys.ingestion import fields

from videosys.ingestion.data import ObjectDetectionResult
from ..base import Operator

class MMDetDetectorPipeline(Operator):
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

        tables[fields.DATA_OBJECT_DETECTION] = self._collect_bbox_result(bbox_result)

        self.collector.emit(tables)
    
    def _collect_bbox_result(self, bbox_result):
        # result dimension: label -> (x, 5)
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        if len(bboxes) == 0:
            return []
        assert len(bboxes[0]) == 5, 'expecting a confidence value from the NN'
        labels = np.concatenate(labels)
        # print('bbox:',bboxes[0], labels[0])
        obj_result = []
        for bbox, label in zip(bboxes, labels):
            if self.detect_classes is None or label in self.detect_classes:
                obj_result.append(ObjectDetectionResult(bbox[:-1], label, bbox[-1]))
        return obj_result

class MMDetDetectorWithFeatures(MMDetDetectorPipeline):

    def __init__(self, rescale=True, **kwargs):
        self.rescale=rescale
        super().__init__(**kwargs)

    def process(self, tables):
        data = tables[fields.COMPAT_MMLIB]
        img = data['img'][0]
        img_metas = data['img_metas'][0]

        with torch.no_grad():
            x = self.model.extract_feat(img)
            if hasattr(self.model, 'roi_head'):
                # TODO: check whether this is the case
                # if public_bboxes is not None:
                #     public_bboxes = [_[0] for _ in public_bboxes]
                #     proposals = public_bboxes
                # else:
                #     proposals = self.detector.rpn_head.simple_test_rpn(
                #         x, img_metas)
                proposals = self.model.rpn_head.simple_test_rpn(x, img_metas)
                det_bboxes, det_labels = self.model.roi_head.simple_test_bboxes(
                    x,
                    img_metas,
                    proposals,
                    self.model.roi_head.test_cfg,
                    rescale=self.rescale)
                # TODO: support batch inference
                det_bboxes = det_bboxes[0]
                det_labels = det_labels[0]
                num_classes = self.model.roi_head.bbox_head.num_classes
            elif hasattr(self.model, 'bbox_head'):
                outs = self.model.bbox_head(x)
                result_list = self.model.bbox_head.get_bboxes(
                    *outs, img_metas=img_metas, rescale=self.rescale)
                # TODO: support batch inference
                det_bboxes = result_list[0][0]
                det_labels = result_list[0][1]
                num_classes = self.model.bbox_head.num_classes
            else:
                raise TypeError('model must has roi_head or bbox_head.')
            # print('det_bboxes', det_bboxes.shape)
            # emit x & det values.
            bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
            tables[fields.DATA_OBJECT_DETECTION] = self._collect_bbox_result(bbox_result)
            tables[fields.DATA_FEATURES] = x
            self.collector.emit(tables)
