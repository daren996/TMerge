from videosys.ingestion.base import Operator
from videosys.ingestion import fields

from .lib.deep_sort.tracker import Tracker
from .lib.deep_sort.detection import Detection
from .lib.deep_sort.generate_detections import create_box_encoder
from .lib.deep_sort import nn_matching
from .data import Tracklet

def convert_boxes(boxes):
    returned_boxes = []
    for box in boxes:
        box_xywh = [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]
        if box_xywh != [0,0,0,0]:
            returned_boxes.append(box_xywh)
    return returned_boxes

class DeepSORTOnlineTracker(Operator):

    def __init__(self, config=None):
        super().__init__(config=config)
        self.__min_threshold = self._get_config('min_threshold', 0.3)

    def prepare(self):
        model_filename = self._get_config('model','models/deep_sort/mars-small128.pb')
        self.encoder = create_box_encoder(model_filename, batch_size=1)

        max_cosine_distance = 0.5
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        self.tracker = Tracker(metric)
    
    def process(self, tables):
        detections = tables[fields.DATA_OBJECT_DETECTION]
        detections = [det for det in detections if det.confidence >= self.__min_threshold]

        frame = tables[fields.DATA_FRAME]
        converted_boxes = convert_boxes([det.bbox for det in detections])

        features = self.encoder(frame, converted_boxes)
        dets_with_features = [Detection(bbox, det.confidence, feature, det) 
            for det, feature, bbox in zip(detections, features, converted_boxes)]
        # process.
        self.tracker.predict()
        self.tracker.update(dets_with_features)
        tables[fields.DATA_OBJECT_TRACK] = [Tracklet(i.track_id, -1, 
            [i.payload.bbox], -1, -1, -1, i.payload) for i in self.tracker.tracks]
        self.collector.emit(tables)
