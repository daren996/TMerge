# other common operators
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class SkipToFrame(Operator):
    def __init__(self, start_fid):
        super(SkipToFrame, self).__init__()
        self.start_fid = start_fid

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        if fid >= self.start_fid:
            self.collector.emit(tables)
