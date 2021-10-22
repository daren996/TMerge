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
        self.context.put(fields.META_VIDEO, 
            data.VideoMeta(fps, width, height, frame_count, self.file))
        self.__video = video
        self.fid = 0
        self.__end = False

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
            else:
                self.__end = True

    def has_next(self):
        return not self.__end

    def cleanup(self):
        self.__video.release()
    
class ImageSource(Source):
    def __init__(self, folder, image_prefix=''):
        self.folder = folder
        self.image_prefix = image_prefix
        super(ImageSource, self).__init__()

    def read_image_folder(self, folder):
        images = []
        for root, _, files in os.walk(folder):
            for file in files:
                image_seq_str = file[len(self.image_prefix): file.index('.')]
                images.append((os.path.join(root, file), int(image_seq_str)))
        # sort images.
        images.sort(key=cmp_to_key(lambda x1, x2: x1[1] - x2[1]))
        return images

    def prepare(self):
        # read images.
        self.images = self.read_image_folder(self.folder)
        self.context.put(fields.META_IMAGE, data.ImageFolderMeta(len(self.images), self.folder))
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

class ConcatMultiImageSource(ImageSource):
    def __init__(self, folder_list):
        self.folder_list = folder_list
        super().__init__(None)
    
    def prepare(self):
        # read images
        images_list = []
        for folder in self.folder_list:
            images = self.read_image_folder(folder)
            images_list +=images
        self.images = images_list
        self.context.put(fields.META_IMAGE, 
            data.ImageFolderMeta(len(self.images), self.folder_list))
        self.current = 0
    
# =============== sink operators
    
class VideoSink(Operator):
    def __init__(self, path, name, image_key = fields.DATA_FRAME, fps=None):
        self.path = path
        self.name = name
        self.image_key = image_key
        self.fps = fps
        super().__init__()

    def prepare(self):
        self.file_name = os.path.join(self.path, '{}.mp4'.format(self.name))
        self.out = None
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if self.context.has(fields.META_VIDEO):
            video_meta = self.context.get(fields.META_VIDEO)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.file_name, fourcc, 
                self.fps if self.fps is not None else video_meta.fps, 
                (int(video_meta.width), int(video_meta.height)))
            

    def process(self, tables):
        # store frame.
        image = tables[self.image_key]
        if self.out is None:
            # init
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.file_name, fourcc, 
                self.fps if self.fps is not None else 30.,
                 (int(image.shape[1]),int(image.shape[0])))
        self.out.write(image)
        self.collector.emit(tables)
    
    def cleanup(self):
        self.out.release()

class ImageSink(Operator):
    def __init__(self, path, image_key = fields.DATA_FRAME):
        self.path = path
        self.image_key = image_key
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
        self.collector.emit(tables)
