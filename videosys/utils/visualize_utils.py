import cv2

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
