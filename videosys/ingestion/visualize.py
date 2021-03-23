import cv2

from videosys.utils.visualize_utils import listen_key_events
from .base import Operator

class ImageVisualizer(Operator):
    def __init__(self, config):
        super().__init__(config=config)
        self.__auto_play = self._get_config('auto_play', True)
        self.window_name='img_vis'

    def process(self, tables):
        frame = tables['frame']
        cv2.imshow(self.window_name, frame)
        if self.__auto_play:
            cv2.waitKey(1)
            self.collector.emit()
        else:
            listen_key_events(self.window_name,
                { 'n': lambda: self.collector.emit },
                on_close=self.controller.stop)

    def cleanup(self):
        cv2.destroyWindow(self.window_name)
