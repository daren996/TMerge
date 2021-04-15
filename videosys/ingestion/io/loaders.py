# load other files given frame id.
from collections import defaultdict
import numpy as np

from videosys.ingestion.base import Operator
from videosys.ingestion import fields
from videosys.ingestion.data import ObjectDetectionResult, ObjectTrackingResult

class MOTDetLoader(Operator):
    """
    load det.txt provided by MOT datasets
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        super().__init__()

    def prepare(self):
        self.frame_det_dict = defaultdict(list)
        with open('{}/det/det.txt'.format(self.folder_path)) as f:
            for line in f.readlines():
                arr = line.split(',')
                frame_id = int(arr[0])
                l, t, w, h = float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])
                self.frame_det_dict[frame_id].append(ObjectDetectionResult(
                    np.array([l, t, l+w, t +h]),
                    -1,
                    float(arr[6])
                ))

        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        tables[fields.DATA_OBJECT_DETECTION] = self.frame_det_dict[fid]
        self.collector.emit(tables)

class MOTFormatLoader(Operator):
    """
    load ground truth gt.txt provided by MOT datasets.
    
    data will be put into fields.DATA_OBJECT_TRACK_GT
    """
    def __init__(self, file_path, emit_key):
        self.file_path = file_path
        self.emit_key = emit_key
        super().__init__()

    def prepare(self):
        self.frame_det_dict = defaultdict(list)
        with open(self.file_path) as f:
            for line in f.readlines():
                arr = line.split(',')
                frame_id = int(arr[0])
                l, t, w, h = float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])
                bbox = np.array([l, t, l+w, t +h])
                self.frame_det_dict[frame_id].append(ObjectTrackingResult(
                    int(arr[1]),
                    -1,
                    bbox,
                    float(arr[6]),
                    ObjectDetectionResult(bbox, -1, 1)
                ))

        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        tables[self.emit_key] = self.frame_det_dict[fid]
        self.collector.emit(tables)

class MOTResultLoader(MOTFormatLoader):
    def __init__(self, file_path):
        super().__init__(file_path, emit_key=fields.DATA_OBJECT_TRACK)

class MOTGTLoader(MOTFormatLoader):
    def __init__(self, file_path):
        super().__init__(file_path, emit_key=fields.DATA_OBJECT_TRACK_GT)
