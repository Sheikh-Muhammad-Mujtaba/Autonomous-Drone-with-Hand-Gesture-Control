"""Central configuration for the Drone Reporter project.

Every tunable and artifact path lives here so all modules resolve
project files relative to this file (not the current working
directory).  Import as `import config` or `from config import ...`.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Video / display
# ---------------------------------------------------------------------------
DISPLAY_W = 360          # Drone View / pose input width
DISPLAY_H = 240          # Drone View / pose input height
TARGET_FPS = 30          # loop rate; also the Tello RC send budget
GESTURE_CROP_SIZE = 320  # hand crop fed to the gesture classifier

# ---------------------------------------------------------------------------
# Timing diagnostics (printed every TIMING_EVERY loops when enabled)
# ---------------------------------------------------------------------------
DEBUG_TIMING = False
TIMING_EVERY = 30

# ---------------------------------------------------------------------------
# Model artifacts (stored under model/)
# ---------------------------------------------------------------------------
GESTURE_MODEL_PATH = str(
    PROJECT_ROOT / "model" / "hand_gesture_model_improve.h5"
)
LABEL_ENCODER_PATH = str(
    PROJECT_ROOT / "model" / "label_encoder_improve.pkl"
)
FACE_CASCADE_PATH = str(
    PROJECT_ROOT / "model" / "haarcascade_frontalface_default.xml"
)
YOLO_MODEL_PATH = str(
    PROJECT_ROOT / "model" / "yolov8s.pt"
)

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
CALIBRATION_PHOTOS_DIR = str(PROJECT_ROOT / "calibration_photos")
