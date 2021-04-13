import numpy as np 

from videosys.ingestion.data import ObjectTrackingResult
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

from .lib.deep_sort import preprocessing
from .lib.deep_sort.tracker import Tracker
from .lib.deep_sort.detection import Detection
from .lib.deep_sort.generate_detections import create_box_encoder
from .lib.deep_sort import nn_matching

def convert_boxes(boxes):
    """
    convert to xywh
    """
    returned_boxes = []
    for box in boxes:
        box_xywh = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]
        if box_xywh != [0,0,0,0]:
            returned_boxes.append(box_xywh)
    return returned_boxes

class DeepSORTOnlineTracker(Operator):

    def __init__(self, min_threshold=0, model='models/deep_sort/mars-small128.pb'):
        self.__min_threshold = min_threshold
        self.model_filename = model
        super().__init__()

    def prepare(self):
        self.encoder = create_box_encoder(self.model_filename, batch_size=1)

        max_cosine_distance = 0.2
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.nms_max_overlap = 1
        self.tracker = Tracker(metric)
    
    def process(self, tables):
        detections = tables[fields.DATA_OBJECT_DETECTION]
        detections = [det for det in detections if det.confidence >= self.__min_threshold]

        frame = tables[fields.DATA_FRAME]
        converted_boxes = convert_boxes([det.bbox for det in detections])

        features = self.encoder(frame, converted_boxes)

        indices = preprocessing.non_max_suppression(
            np.array(converted_boxes), self.nms_max_overlap, 
            np.array([d.confidence for d in detections]))

        dets_with_features = [Detection(converted_boxes[i], detections[i].confidence, 
            features[i], detections[i]) for i in indices]
        
        # process.
        self.tracker.predict()
        self.tracker.update(dets_with_features)

        confirmed = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            confirmed.append(ObjectTrackingResult(track.track_id, track.payload.label, 
                track.to_tlbr(), track.payload.confidence, track.payload))
        tables[fields.DATA_OBJECT_TRACK] = confirmed
        self.collector.emit(tables)
