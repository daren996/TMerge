from mmtrack.apis import inference_sot, init_model

from videosys.ingestion.data import SOTResult
from videosys.ingestion import fields
from videosys.ingestion.base import Operator

class MMTrackSOTPipeline(Operator):
    
    def __init__(self, config, checkpoint, device='cuda:0'):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        super().__init__()
    
    def prepare(self):
        self.model = init_model(self.config, self.checkpoint, self.device)

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        frame = tables[fields.DATA_FRAME]
        init_bbox = tables[fields.SELECT_ROI]
        # fid starts with 0.
        result = inference_sot(self.model, frame, init_bbox, fid-1)

        score, track_bbox = result['score'], result['bbox']
        tables[fields.DATA_SOT_RESULT] = SOTResult(track_bbox, score)
        self.collector.emit(tables)
