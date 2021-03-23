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
