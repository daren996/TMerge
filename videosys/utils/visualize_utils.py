import cv2
import numpy as np

from videosys.utils.color import color_val, rand_color

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
    copy_image = True, id_func=None, bbox_color=None, label_color=None, thickness=1, \
        font = cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
        label_reverse_bg=False):
    bbox_color = color_val(bbox_color) if bbox_color else None
    label_color = color_val(label_color) if label_color else None
    if copy_image:
        image = image.copy()
    text_width, text_height = 10, 15
    for obj in objs:
        bbox = bbox_func(obj).astype(np.int32)
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        if id_func is not None:
            oid = id_func(obj)
            bbox_color = rand_color(oid)
            label_color = rand_color(oid)
        cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=thickness)
        label_txt = label_func(obj)
        x1, y1 =bbox[0], bbox[1]
        for label in label_txt.split('\n'):
            width = len(label) * text_width
            if label_reverse_bg:
                image[y1:y1 + text_height, x1:x1 + width, :] = label_color
            cv2.putText(image, label, (x1, y1 + text_height -2), font, font_scale, 
                (0,0,0) if label_reverse_bg else label_color, thickness=font_thickness)
            y1 += text_height
    return image
