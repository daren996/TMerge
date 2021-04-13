import cv2

from videosys.utils.visualize_utils import listen_key_events, add_text_to_image
from videosys.ingestion import fields
from videosys.ingestion.base import Operator

class ImageVisualizer(Operator):
    def __init__(self, auto_play=True, window_name='img_vis'):
        super().__init__()
        self.__auto_play = auto_play
        self.window_name=window_name

    def prepare(self):
        self.__fps = self.context.get(fields.META_VIDEO).fps \
            if self.context.has(fields.META_VIDEO) else 1

    def generate_text(self, frame_id):
        txt = []
        if self.context.has(fields.META_VIDEO):
            video_meta = self.context.get(fields.META_VIDEO)
            txt.append('fps: {}'.format(video_meta.fps))
            txt.append('frameCount: {}'.format(video_meta.frame_count))
        elif self.context.has(fields.META_IMAGE):
            image_meta = self.context.get(fields.META_IMAGE)
            txt.append('total:{}'.format(image_meta.total))
        txt.append('current: {}'.format(frame_id))
        return txt

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        frame_id = tables[fields.DATA_FRAME_ID]
        frame = add_text_to_image(frame, self.generate_text(frame_id))
        cv2.imshow(self.window_name, frame)
        if self.__auto_play:
            cv2.waitKey(round(1000/self.__fps))
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.controller.stop()
            self.collector.emit()
        else:
            listen_key_events(self.window_name,
                { 'n': lambda: self.collector.emit },
                on_close=self.controller.stop)

    def cleanup(self):
        cv2.destroyWindow(self.window_name)
