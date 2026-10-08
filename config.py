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
# Gesture recognition
# ---------------------------------------------------------------------------
# Minimum softmax probability for a classifier result to count at all.
GESTURE_CONFIDENCE_THRESHOLD = 0.80
# MediaPipe Hands detection confidence. The crop is already centred on a
# raised wrist, so a lower bar than the old 0.7 finds hands reliably.
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
# Consecutive identical results needed before a gesture is acted on.
GESTURE_CONFIRM_MOVE = 3     # continuous moves (up/down/left/turn/...)
GESTURE_CONFIRM_STOP = 4     # "stop" = land; fast, but not single-frame
GESTURE_CONFIRM_TOGGLE = 6   # "flip" toggles face mode; most deliberate
GESTURE_ONE_SHOT_COOLDOWN_S = 2.0  # min gap between two one-shot gestures
GESTURE_STREAK_TIMEOUT_S = 0.5     # a streak with no new result this long resets

# ---------------------------------------------------------------------------
# Reporter tracker (YOLO person tracking)
# ---------------------------------------------------------------------------
# Running YOLO flat out on every core doubled pose+hand latency
# (33 ms alone -> 71 ms with YOLO, 8-core CPU). Two torch threads bring it
# back to ~41 ms; a 416 input makes YOLO ~2.3x faster (~11 fps tracking).
REPORTER_TORCH_THREADS = 2
REPORTER_IMAGE_SIZE = 416   # 640 = original; raise if far-away people are missed
REPORTER_MAX_FPS = 8.0      # upper bound on YOLO+ByteTrack updates

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
