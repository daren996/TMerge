'''
Source & Sink operators
'''
import os
from functools import cmp_to_key
import cv2

from videosys.ingestion import fields, data
from videosys.ingestion.base import Source, Operator

# ============ source operators
class VideoSource(Source):

    def prepare(self):
        video = cv2.VideoCapture(self.config['file'])
        # parse metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.context.put(fields.META_VIDEO, data.VideoMeta(fps, width, height, frame_count))
        self.__video = video
        self.fid = 0

    def process(self, tables=None):
        if self.__video.isOpened():
            ret, frame = self.__video.read()
            if ret:
                self.fid += 1
                self.collector.emit({fields.DATA_FRAME:frame, \
                    fields.DATA_FRAME_ID: self.fid})

    def has_next(self):
        return self.__video.isOpened()

    def cleanup(self):
        self.__video.release()

class ImageSource(Source):
    def prepare(self):
        # read images.
        images = []
        for root, _, files in os.walk(self._get_config('folder')):
            for file in files:
                images.append((os.path.join(root, file), int(file[:file.index('.')])))
        self.images = images
        # sort images.
        self.images.sort(key=cmp_to_key(lambda x1, x2: x1[1] - x2[1]))
        self.context.put(fields.META_IMAGE, data.ImageFolderMeta(len(self.images)))
        self.current = 0

    def process(self, tables=None):
        self.current += 1
        frame = cv2.imread(self.images[self.current][0])
        self.collector.emit({fields.DATA_FRAME: frame, fields.DATA_FRAME_ID: self.current})
    
    def has_next(self):
        return self.current +1 < len(self.images)
    
# =============== sink operators
    
class VideoSink(Operator):
    pass

class ImageSink(Operator):
    pass
