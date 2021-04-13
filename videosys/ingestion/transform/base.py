from videosys.ingestion import fields
from videosys.ingestion.base import Operator

class BaseResizeTransformer(Operator):
    def __init__(self, height, width, method='bilinear'):
        self.method = method
        self.new_height = height
        self.new_width = width
        super().__init__()
    
    def resize(self, frame):
        return frame

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        tables[fields.DATA_FRAME] = self.resize(frame)
