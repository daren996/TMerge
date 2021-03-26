# load other files given frame id.
from collections import defaultdict
import numpy as np

from videosys.ingestion.base import Operator
from videosys.ingestion import fields
from videosys.ingestion.data import ObjectDetectionResult, Tracklet

class MOTDetLoader(Operator):
    """
    load det.txt provided by MOT datasets
    """

    def prepare(self):
        self.folder_path = self._get_config('folder_path')
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

class MOTGTLoader(Operator):
    """
    load ground truth gt.txt provided by MOT datasets.
    
    data will be put into fields.DATA_OBJECT_TRACK_GT
    """

    def prepare(self):
        self.folder_path = self._get_config('folder_path')
        self.frame_det_dict = defaultdict(list)
        with open('{}/gt/gt.txt'.format(self.folder_path)) as f:
            for line in f.readlines():
                arr = line.split(',')
                frame_id = int(arr[0])
                l, t, w, h = float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])
                bbox = np.array([l, t, l+w, t +h])
                self.frame_det_dict[frame_id].append(Tracklet(
                    int(arr[1]),
                    -1,
                    [bbox],
                    -1, -1, -1,
                    ObjectDetectionResult(bbox, -1, 1)
                ))

        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        tables[fields.DATA_OBJECT_TRACK_GT] = self.frame_det_dict[fid]
        self.collector.emit(tables)
