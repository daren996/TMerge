from uma_mot.tracker.mot_tracker import MOT_Tracker

from videosys.ingestion.data import ObjectTrackingResult
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class UMAMOT(Operator):

    def __init__(self, siamese_checkpoint, max_age=10, context_amount=0.3, 
        occlusion_threshold=0.8, association_threshold=0.7, iou_threshold=0.25):
        super().__init__()
        self.siamese_checkpoint = siamese_checkpoint
        self.max_age = max_age
        self.context_amount = context_amount
        self.occlusion_threshold = occlusion_threshold
        self.association_threshold = association_threshold
        self.iou_threshold = iou_threshold

    def prepare(self):
        frame_rate = 30
        if self.context.has(fields.META_VIDEO):
            video_meta = self.context.get(fields.META_VIDEO)
            frame_rate = video_meta.fps
        self.tracker = MOT_Tracker(self.max_age, self.occlusion_threshold, 
            self.association_threshold, self.iou_threshold, self.context_amount,
            self.siamese_checkpoint, frame_rate)
    
    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        frame = tables[fields.DATA_FRAME]
        detections = tables[fields.DATA_OBJECT_DETECTION]

        trackers = self.tracker.update(frame, fid, detections, 
            bbox_func=lambda x : x.bbox, bbox_type='tlbr', return_raw=True)
        results = []
        for t in trackers:
            # xywh to tlbr
            bbox = t.track_bbox
            bbox[2:] += bbox[:2]
            results.append(ObjectTrackingResult(t.track_id, t.payload.label, 
                bbox, t.payload.confidence, t.payload))

        tables[fields.DATA_OBJECT_TRACK] = results
        self.collector.emit(tables)
