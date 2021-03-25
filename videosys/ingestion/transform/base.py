from videosys.ingestion import fields
from videosys.ingestion.base import Operator

class BaseResizeTransformer(Operator):
    def prepare(self):
        self.method = self._get_config('method', 'bilinear')
        self.new_height = self._get_config('height')
        self.new_width = self._get_config('width')
    
    def resize(self, frame):
        return frame

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        tables[fields.DATA_FRAME] = self.resize(frame)
