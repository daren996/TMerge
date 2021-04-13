from mmtrack.apis import inference_mot, init_model
from mmtrack.core import restore_result, track2result
from mmtrack.models.mot import DeepSORT
from mmtrack.models.mot.trackers.sort_tracker import SortTracker
from mmtrack.models.motion.kalman_filter import KalmanFilter

import torch

from videosys.ingestion.data import ObjectTrackingResult
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class MMTrackingMOT(Operator):

    def prepare(self):
        config_file = self._get_config('config_file')
        # checkpoint_file = self._get_config('checkpoint_file')
        checkpoint_file = None
        device = self._get_config('device', 'cuda:0')
        self.model = init_model(config_file, 
            checkpoint_file, device)

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        fid = tables[fields.DATA_FRAME_ID]
        result = inference_mot(self.model, frame, fid)
        result = result['track_results']
        bboxes, labels, ids = restore_result(result, return_ids=True)
        result = []
        for bbox, label, uid in zip(bboxes, labels, ids):
            result.append(ObjectTrackingResult(uid, label, bbox[:4], bbox[4]))
            # print(result[-1])
        tables[fields.DATA_OBJECT_TRACK] = result
        self.collector.emit(tables)

class MMTrackingSORT(Operator):
    """
    use the SORT algorithm implemented in mm-tracking lib.
    """
    
    def prepare(self):
        self.model = DeepSORT(
            motion=dict(type='KalmanFilter', center_only=False),
            tracker=dict(type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None)
        )
        self.tracker = self.model.tracker

    def process(self, tables):
        num_classes = len(self.context.get(fields.META_OBJECT_DETECTION_CLASSES))
        frame = tables[fields.DATA_FRAME]
        fid = tables[fields.DATA_FRAME_ID]
        detections = tables[fields.DATA_OBJECT_DETECTION]
        img_metas = dict()

        bboxes_tensor = torch.tensor([[*d.bbox, d.confidence] for d in detections])
        labels_tensor = torch.tensor([d.label for d in detections])
        
        # convert tensor results to np.array
        bboxes, labels, ids = self.tracker.track(frame, img_metas, 
            self.model, bboxes_tensor, labels_tensor, fid)
        tracking_result = track2result(bboxes, labels, ids, num_classes)
        bboxes, labels, ids = restore_result(tracking_result, return_ids=True)
        
        result = []
        for bbox, label, uid in zip(bboxes, labels, ids):
            result.append(ObjectTrackingResult(uid, label, bbox[:4], bbox[4]))
            # print(result[-1])
        tables[fields.DATA_OBJECT_TRACK] = result
        self.collector.emit(tables)
        
        

