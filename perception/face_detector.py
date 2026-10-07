# # --- face_tracking_module.py ---
# import os
# import cv2
# import numpy as np

# fbRange = [4000, 6000]  # Default values, can be changed in main script

# # FIX: build the cascade path relative to this script's location
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CASCADE_PATH = os.path.join(BASE_DIR, "model", "haarcascade_frontalface_default.xml")


# def findFace(img):
#     faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
#     myFaceListC, myFaceListArea = [], []

#     for (x, y, w, h) in faces:
#         cx, cy = x + w // 2, y + h // 2
#         area = w * h
#         myFaceListC.append([cx, cy])
#         myFaceListArea.append(area)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

#     if myFaceListArea:
#         i = myFaceListArea.index(max(myFaceListArea))
#         return img, [myFaceListC[i], myFaceListArea[i]]
#     else:
#         return img, [[0, 0], 0]


# def trackFace(info, w, pid, pError):
#     area = info[1]
#     x, y = info[0]
#     fb = 0

#     error = x - w // 2

#     # Dead zone
#     if abs(error) < 20:
#         error = 0

#     # PID control
#     speed = pid[0] * error + pid[1] * (error - pError)

#     # Slow down near center
#     if abs(error) < 60:
#         speed *= 0.5

#     speed = int(np.clip(speed, -30, 30))  # limit yaw speed

#     # Forward/Backward logic
#     if area > 6000:
#         fb = -20
#     elif area < 4000 and area != 0:
#         fb = 20
#     else:
#         fb = 0

#     if x == 0:
#         error = 0
#         speed = 0
#         fb = 0

#     print(f"Forward/Backward: {fb}, Yaw: {speed}, pError: {error}")
#     return fb, speed, error
# --- face_tracking_module.py ---
import os
import cv2
import numpy as np
from collections import deque

from config import FACE_CASCADE_PATH

fbRange = [4000, 6000]  # kept only for backward reference, no longer used directly

# FIX: build the cascade path relative to this script's location
CASCADE_PATH = FACE_CASCADE_PATH

# NEW: real-world distance calibration (from calibrate_distance.py).
# You measured w=144px at exactly 100cm (1 meter) -> focal length below.
KNOWN_FACE_WIDTH_CM = 14
FOCAL_LENGTH = (144 * 100) / KNOWN_FACE_WIDTH_CM   # = 1028.57

# NEW: target distance in cm the drone should try to hold from the face,
# and how much slack (in cm) around that target counts as "close enough".
TARGET_DISTANCE_CM = 100     # keep the drone ~1 meter away
DISTANCE_DEADZONE_CM = 15    # +/- 15cm around target is treated as "good"

# NEW: rolling buffer of recent face-width readings, used to smooth out
# frame-to-frame jitter from the Haar cascade before we act on it.
WIDTH_SMOOTHING_WINDOW = 5
_width_history = deque(maxlen=WIDTH_SMOOTHING_WINDOW)

# NEW: proportional forward/backward control tuning.
FB_MAX_SPEED = 30          # cap on forward/backward speed
FB_GAIN = 0.6              # how strongly distance-error (cm) converts to speed
HARD_STOP_DISTANCE_CM = 50 # if drone gets closer than this, force max backward regardless of gain


def _smoothedWidth(raw_width):
    """Returns a moving-average of the last few face-width readings.
    A single Haar cascade detection can jitter in size even when the
    person hasn't moved, which would otherwise make fb twitch."""
    if raw_width > 0:
        _width_history.append(raw_width)
    if len(_width_history) == 0:
        return 0
    return sum(_width_history) / len(_width_history)


def estimateDistanceCM(face_width_px):
    """Converts a detected face pixel-width into an estimated real-world
    distance in centimeters, using the focal length calibrated at 1m."""
    if face_width_px <= 0:
        return None
    return (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / face_width_px


def findFace(img):
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC, myFaceListArea, myFaceListWidth = [], [], []

    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        area = w * h
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
        myFaceListWidth.append(w)  # NEW: track pixel width for distance estimation
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    if myFaceListArea:
        i = myFaceListArea.index(max(myFaceListArea))
        # NEW: info now includes face width as a 3rd element
        return img, [myFaceListC[i], myFaceListArea[i], myFaceListWidth[i]]
    else:
        return img, [[0, 0], 0, 0]


def trackFace(info, w, pid, pError):
    x, y = info[0]
    raw_width = info[2] if len(info) > 2 else 0
    fb = 0

    # NEW: smooth the width reading, then convert it into a real distance in cm
    smoothed_width = _smoothedWidth(raw_width)
    distance_cm = estimateDistanceCM(smoothed_width) if smoothed_width > 0 else None

    error = x - w // 2

    # Dead zone
    if abs(error) < 20:
        error = 0

    # PID control
    speed = pid[0] * error + pid[1] * (error - pError)

    # Slow down near center
    if abs(error) < 60:
        speed *= 0.5

    speed = int(np.clip(speed, -30, 30))  # limit yaw speed

    # NEW: forward/backward logic based on real distance in cm instead of
    # pixel area. Positive fb = move forward (too far), negative = move
    # backward (too close). A hard-stop override forces max backward
    # speed if the drone gets dangerously close, regardless of gain.
    if raw_width == 0 or distance_cm is None:
        # no face currently detected - don't move on stale data
        fb = 0
    elif distance_cm < HARD_STOP_DISTANCE_CM:
        fb = -FB_MAX_SPEED  # too close - back away immediately, override normal gain
    elif abs(distance_cm - TARGET_DISTANCE_CM) < DISTANCE_DEADZONE_CM:
        fb = 0  # close enough to target distance - hold position
    else:
        distance_error = TARGET_DISTANCE_CM - distance_cm  # positive = too far, negative = too close
        fb = int(np.clip(distance_error * FB_GAIN, -FB_MAX_SPEED, FB_MAX_SPEED))

    if x == 0:
        error = 0
        speed = 0
        fb = 0

    dist_display = f"{distance_cm:.1f}cm" if distance_cm else "N/A"
    print(f"Forward/Backward: {fb}, Yaw: {speed}, pError: {error}, distance: {dist_display}")
    return fb, speed, error
