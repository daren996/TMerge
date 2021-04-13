'''
Implementation based on 
    https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py
and 
    https://github.com/adipandas/multi-object-tracker/blob/master/motrackers/iou_tracker.py
'''

from collections import OrderedDict

from videosys.ingestion.base import Operator
from videosys.ingestion import fields

from .utils import iou
from .data import Tracklet, tracklet_to_result

class IOUTracker:
    '''
    Track objects based on IOU only.
    '''
    def __init__(self, ttl=-1, iou_threshold=0.5, \
        min_detection_confidence=0.4):
        '''
        Parameters
        -----
        ttl: time to live for each id. After ttl frames, the object will be considered as removed.
        '''
        self.ttl=ttl
        self.iou_threshold=iou_threshold
        self.min_detection_confidence=min_detection_confidence
        self.trackers_active=OrderedDict()
        self.frame_num =0
        self.next_id = 1
    
    def __new_tracker(self, det):
        tracker = Tracklet(self.next_id, self.frame_num, [det.bbox], det.confidence, 0, 1, det)
        self.next_id += 1
        return tracker

    def update(self, detections):
        '''
        update the tracker with results in one frame.
        '''
        dets = [i for i in detections if i.confidence >= self.min_detection_confidence]
        updated_tracks = []

        for tracklet in self.trackers_active.values():
            found = False
            if len(dets) > 0:
                idx, best_match = max(enumerate(dets), \
                    key=lambda x: iou(tracklet.bboxes[-1], x[1].bbox))
                if iou(tracklet.bboxes[-1], best_match.bbox) >= self.iou_threshold:
                    # update.
                    tracklet.bboxes.append(best_match.bbox)
                    tracklet.max_score = max(tracklet.max_score, best_match.confidence)
                    # update payload.
                    tracklet.payload = best_match
                    tracklet.count += 1

                    updated_tracks.append(tracklet)
                    # remove from best matching detection from detections
                    del dets[idx]
                    found = True
                    # add to updated
                    updated_tracks.append(tracklet)

            if not found:
                tracklet.missing += 1
                # remove from active dict.
                if self.ttl >=0 and tracklet.missing > self.ttl:
                    del self.trackers_active[tracklet.uid]
            
        # create new tracks
        new_tracks = [self.__new_tracker(det) for det in dets]
        tracklet_in_frame = updated_tracks[:]
        for tracklet in new_tracks:
            self.trackers_active[tracklet.uid] = tracklet
            tracklet_in_frame.append(tracklet)

        self.frame_num += 1
        return tracklet_in_frame
        

class IOUOnlineTracker(Operator):

    def __init__(self, ttl=-1, iou_threshold=0.5, min_conf=0.4):
        self.tracker = IOUTracker(
            ttl = ttl,
            iou_threshold= iou_threshold,
            min_detection_confidence= min_conf
        )
        super().__init__()

    def process(self, tables):
        detections = tables[fields.DATA_OBJECT_DETECTION]
        tracklets = self.tracker.update(detections)
        tables[fields.DATA_OBJECT_TRACK] = [tracklet_to_result(t) for t in tracklets]
        self.collector.emit(tables)


class IOUBatchTracker(Operator):

    def __init__(self, ttl, result_only=False, min_conf=0.4, 
            max_conf = 0.5, t_min=5, iou_threshold=0.5):
        self.__ttl = ttl
        self.__result_only = result_only
        self.__max_detection_confidence = max_conf
        # at least how many frames an object should appear
        self.__t_min = t_min
        self.tracker = IOUTracker(
            ttl = self.__ttl,
            iou_threshold= iou_threshold,
            min_detection_confidence= min_conf
        )
        self.max_tracklets = OrderedDict()
        self.track_results = []
        self.buffered_tables = []
        super().__init__()
    
    def process(self, tables):
        detections = tables[fields.DATA_OBJECT_DETECTION]
        tracklets = self.tracker.update(detections)
        self.track_results.append(tracklets)
        for track in tracklets:
            self.max_tracklets[track.uid] = track
        if not self.__result_only:
            self.buffered_tables.append(tables)
        
    def cleanup(self):
        # filter based on other two params.
        keep_set = set()
        for uid, tracklet in self.max_tracklets.items():
            if tracklet.max_score >= self.__max_detection_confidence and \
                len(tracklet.bboxes) >= self.__t_min:
                keep_set.add(uid)
        if self.__result_only:
            for frame_result in self.track_results:
                self.collector.emit({fields.DATA_OBJECT_TRACK: 
                [tracklet_to_result(i) for i in frame_result if i.uid in keep_set]})
        else:
            for tables, frame_result in zip(self.buffered_tables, self.track_results):
                tables[fields.DATA_OBJECT_TRACK] = [i for i in frame_result if i.uid in keep_set]
                self.collector.emit(tables)
        