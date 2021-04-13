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

    def __init__(self, file):
        self.file = file
        super(VideoSource, self).__init__()

    def prepare(self):
        video = cv2.VideoCapture(self.file)
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
                # add image metas
                img_meta = dict()
                img_meta['img_shape'] = frame.shape
                img_meta['ori_shape'] = frame.shape

                self.collector.emit({fields.DATA_FRAME:frame, \
                    fields.DATA_FRAME_ID: self.fid, 
                    fields.DATA_FRAME_META: img_meta})

    def has_next(self):
        return self.__video.isOpened()

    def cleanup(self):
        self.__video.release()

class ImageSource(Source):
    def __init__(self, folder):
        self.folder = folder
        super(ImageSource, self).__init__()

    def prepare(self):
        # read images.
        images = []
        for root, _, files in os.walk(self.folder):
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
        # add image metas
        img_meta = dict()
        img_meta['img_shape'] = frame.shape
        img_meta['ori_shape'] = frame.shape
        self.collector.emit({fields.DATA_FRAME: frame,
            fields.DATA_FRAME_ID: self.current, 
            fields.DATA_FRAME_META: img_meta}
        )
    
    def has_next(self):
        return self.current +1 < len(self.images)
    
# =============== sink operators
    
class VideoSink(Operator):
    def __init__(self, path, name, image_key = fields.DATA_FRAME):
        self.path = path
        self.name = name
        self.image_key = image_key
        super().__init__()

    def prepare(self):
        self.file_name = os.path.join(self.path, '{}.mp4'.format(self.name))
        self.out = None
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if self.context.has(fields.META_VIDEO):
            video_meta = self.context.get(fields.META_VIDEO)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(self.file_name, fourcc, video_meta.fps, 
                (int(video_meta.width), int(video_meta.height)))
            

    def process(self, tables):
        # store frame.
        image = tables[self.image_key]
        if self.out is None:
            # init
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(self.file_name, fourcc, 25.0,
                 (int(image.shape[0]),int(image.shape[1])))
        self.out.write(image)
    
    def cleanup(self):
        self.out.release()

class ImageSink(Operator):
    def __init__(self, path, image_key = fields.DATA_FRAME):
        self.path = path
        self.image_key = fields.DATA_FRAME
        super().__init__()

    def prepare(self):
        # create folder.
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def process(self, tables):
        # store frame.
        fid = tables[fields.DATA_FRAME_ID]
        image = tables[self.image_key]
        cv2.imwrite(os.path.join(self.path, '{}.png'.format(fid)), image)
