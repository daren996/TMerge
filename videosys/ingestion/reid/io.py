
import os
import numpy as np
import pandas as pd
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class TrackFeatureSink(Operator):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        parent_folder = os.path.dirname(file_path)
        if not os.path.isdir(parent_folder):
            os.makedirs(parent_folder)
        self.data = []
        self.head = ['fid', 'left', 'top', 'right', 'bottom', 'id', 'label', 'score', 'feature']

    def process(self, tables):
        frame_id = tables[fields.DATA_FRAME_ID]
        tracks = tables[fields.DATA_OBJECT_TRACK]
        track_feats = tables[fields.DATA_TRACK_FEAT]

        # save as numpy array: [fid, bboxes, id, label, score]
        track_dict = dict()
        for track in tracks:
            track_dict[track.uid] = track
        
        for track_f in track_feats:
            track = track_dict[track_f.uid]
            # print(track_f.feature.get_device())
            self.data.append([frame_id, *track_f.bbox, track_f.uid, \
                track.label, track.confidence, [track_f.feature]])

        self.collector.emit(tables)
        
    def cleanup(self):
        df = pd.DataFrame(data = np.array(self.data), columns = self.head)
        df.to_pickle(self.file_path)
