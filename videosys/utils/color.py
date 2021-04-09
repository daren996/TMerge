from enum import Enum
import random

import seaborn as sns


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

def color_val(color):
    if isinstance(color, str):
        return Color[color].value
    return None

def rand_color(seed, cv2=True):
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    if cv2: 
        return [int(255 * _c) for _c in color][::-1]
    return color
