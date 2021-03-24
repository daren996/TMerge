from videosys.utils.color import color_val
import cv2
import numpy as np

def listen_key_events(window, funcs, wait_timeout=1000, on_close=None):
    '''
    window: window name
    funcs: dict, key -> handle func
    '''
    key_code = -1
    while key_code < 0:
        key_code = cv2.waitKey(wait_timeout)
        if on_close is not None and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            on_close()
            return
    for (key, handler) in funcs.items():
        if key_code & 0xFF == ord(key):
            handler()

def add_text_to_image(image, txt, line_height=40, copy_image=True, font_color = (255,255,255), 
    font = cv2.FONT_HERSHEY_SIMPLEX, left_offset = 20, font_scale = 1, line_type = 2):
    if copy_image:
        image = image.copy()
    # height, width, _ = image.shape
    for (i, line) in enumerate(txt):
        cv2.putText(image, line, (left_offset, line_height * (i+1)), font, \
            font_scale, font_color, line_type)
    return image

def draw_bbox_and_labels(image, objs, bbox_func, label_func, \
    copy_image = True, bbox_color='green', label_color='green', thickness=1, \
        font = cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5):
    bbox_color = color_val(bbox_color)
    label_color = color_val(label_color)
    if copy_image:
        image = image.copy()
    for obj in objs:
        bbox = bbox_func(obj).astype(np.int32)
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=thickness)
        label_txt = label_func(obj)
        cv2.putText(image, label_txt, (bbox[0], bbox[1]-2), font, font_scale, label_color)
    return image
