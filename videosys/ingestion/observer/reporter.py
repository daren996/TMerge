import time
import os
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class ProgressReporter(Operator):

    def __init__(self, report_interval=100, save_file=None):
        super().__init__()
        self.report_interval = report_interval
        self.save_file = save_file
    
    def prepare(self):
        self.type = None
        self.total_frames = None
        if self.context.has(fields.META_VIDEO):
            self.type = 'video'
            meta = self.context.get(fields.META_VIDEO)
            self.total_frames = meta.frame_count
            self.path  = meta.path
        elif self.context.has(fields.META_IMAGE):
            self.type = 'image folder'
            meta = self.context.get(fields.META_IMAGE)
            self.total_frames = meta.total
            self.path = meta.path
        if self.save_file is not None:
            # remove file first.
            if os.path.exists(self.save_file):
                os.remove(self.save_file)
            parent_folder = os.path.dirname(self.save_file)
            if not os.path.isdir(parent_folder):
                os.makedirs(parent_folder)
        self.start = time.time()
        self.report('begin to process: [{}], path: [{}]'.format(self.type, self.path))
    
    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        if fid % self.report_interval == 0:
            time_used = time.time() - self.start
            self.report('processing: {:0}/{:0}, {:.0%}, time elapsed: {:.2f}s, fps: {:.2f}'.format(
                fid, self.total_frames, fid/self.total_frames, time_used, fid/time_used))
        self.collector.emit(tables)

    def report(self, msg):
        if self.save_file is not None:
            with open(self.save_file, 'a+') as f:
                f.write(msg+'\n')
        print(msg)

    def cleanup(self):
        time_used = time.time() - self.start
        self.report('done. time elapsed: {:.2f}s. fps: {:.2f}'.format(time_used, self.total_frames/time_used))
