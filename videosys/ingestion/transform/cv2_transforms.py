import cv2
from .base import BaseResizeTransformer

cv2_method_mapping = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

class CV2ResizeTransformer(BaseResizeTransformer):
    def resize(self, frame):
        return cv2.resize(frame, (self.new_height, self.new_width), 
            interpolation=cv2_method_mapping[self.method])
            