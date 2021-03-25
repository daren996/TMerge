'''
Implementation based on 
    https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py
and 
    https://github.com/adipandas/multi-object-tracker/blob/master/motrackers/iou_tracker.py
'''

from collections import OrderedDict

from videosys.ingestion.base import Operator
from videosys.ingestion import fields
from videosys.ingestion.data import Tracklet

from .utils import iou

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

    def prepare(self):
        self.tracker = IOUTracker(
            ttl = self._get_config('ttl', -1),
            iou_threshold= self._get_config('iou_threshold', 0.5),
            min_detection_confidence= self._get_config('min_conf', 0.4)
        )

    def process(self, tables):
        detections = tables[fields.DATA_OBJECT_DETECTION]
        tables[fields.DATA_OBJECT_TRACK] = self.tracker.update(detections)
        self.collector.emit(tables)


class IOUBatchTracker(Operator):

    def prepare(self):
        self.__ttl = self._get_config('ttl', -1)
        self.__result_only = self._get_config('result_only', True)
        self.__max_detection_confidence = self._get_config('max_conf', 0.5)
        # at least how many frames an object should appear
        self.__t_min = self._get_config('t_min', 5)
        self.tracker = IOUTracker(
            ttl = self.__ttl,
            iou_threshold= self._get_config('iou_threshold', 0.5),
            min_detection_confidence= self._get_config('min_conf', 0.4)
        )
        self.max_tracklets = OrderedDict()
        self.track_results = []
        self.buffered_tables = []
    
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
        for idx, frame_result in enumerate(self.track_results):
            tables = self.buffered_tables[idx] if not self.__result_only else {}
            tables[fields.DATA_OBJECT_TRACK] = [i for i in frame_result if i.uid in keep_set]
            self.collector.emit(tables)
        