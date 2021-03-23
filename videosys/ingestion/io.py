'''
Source operators
'''
import cv2

from .base import Operator, Source

class VideoSource(Source):

    def prepare(self):
        video = cv2.VideoCapture(self.config['file'])
        # parse metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.context.put('video_meta', {
            'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count
        })
        self.__video = video

    def process(self):
        if self.__video.isOpened():
            ret, frame = self.__video.read()
            if ret:
                self.collector.emit({'frame':frame})

    def has_next(self):
        return self.__video.isOpened()

    def cleanup(self):
        self.__video.release()

class ImageSource(Operator):
    pass

class VideoSink(Operator):
    pass

class ImageSink(Operator):
    pass