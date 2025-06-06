import os

# Fixed dimensions for resized face images used in recognition
FIXED_FACE_WIDTH = 300
FIXED_FACE_HEIGHT = 300

# Default scale and vertical shift used during face cropping
DEFAULT_SCALE = 1.5
DEFAULT_VERTICAL_SHIFT = 0.0

BASE_DIR = os.path.dirname(__file__)
FACES_DIR = os.path.join(BASE_DIR, "Faces")

REFERENCE_IMAGES = {
    "Sakaryaspor": os.path.join(FACES_DIR, "face1.jpg"),
    "Inter": os.path.join(FACES_DIR, "face2.jpg"),
    }