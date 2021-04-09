
import os

from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class AbstractTrackResultSink(Operator):

    def __init__(self, config):
        super().__init__(config=config)
        self.__save_at_end = self._get_config('save_at_end', False)
        self.__buffers = []
    
    def store(self, fid, track_results):
        pass

    def process(self, tables):
        # tables -> results in one frame 
        if not self.__save_at_end:
            track_results = tables[fields.DATA_OBJECT_TRACK]
            fid = tables[fields.DATA_FRAME_ID]
            self.store(fid, track_results)
        else:
            self.__buffers.append(tables)

    def cleanup(self):
        if self.__save_at_end:
            for tables in self.__buffers:
                track_results = tables[fields.DATA_OBJECT_TRACK]
                fid = tables[fields.DATA_FRAME_ID]
                self.store(fid, track_results)

class TrackResultFolderSink(AbstractTrackResultSink):
    def prepare(self):
        file_path = self._get_config('output_folder')
        # create if not exist
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
        self.file_path = file_path

    def store(self, fid, track_results):
        with open('{}/{}.txt'.format(self.file_path, fid), 'w') as f:
            for tracklet in track_results:
                # FIXME: we assume the payload is of type ObjectDetectionResult
                assert tracklet.payload is not None
                f.write('{};{};{};{}'.format(tracklet.uid, tracklet.bbox, \
                    tracklet.payload.label, tracklet.payload.confidence))
                f.write('\n')


class MOTResultSink(AbstractTrackResultSink):
    def prepare(self):
        file_path = self._get_config('path')
        parent = os.path.dirname(file_path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        # empty file.
        if os.path.exists(file_path):
            os.remove(file_path)
        self.file_path = file_path
    
    def store(self, fid, track_results):
        with open(self.file_path, 'a+') as f:
            for t in track_results:
                f.write(','.join([str(i) for i  in [fid, t.uid, t.bbox[0], t.bbox[1], 
                    t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1], 
                    1, -1, -1, -1]]))
                f.write('\n')
