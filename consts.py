from enum import Enum

folder_path = "images"

# image shape
img_height = 720
img_width = 1280

# opacity level
opacity = 0.2


class Modes(Enum):
    DRAW = "draw"
    SELECT = "select"


class Shapes(Enum):
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"


class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (200, 50, 50)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)


class BrushSize(Enum):
    THIN = 5
    REGULAR = 25
    THICK = 60
