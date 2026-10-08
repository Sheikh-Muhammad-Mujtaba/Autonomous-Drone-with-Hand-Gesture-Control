"""Face detection + PID tracking for Face Tracking Mode.

These are the ACTIVE versions from the flight loop: trackFace() sends
its own RC via the drone handle injected by main.py, and returns the
tuple (lr, fb, ud, yv, error) so the caller can feed the movement into
PositionTracker.update().
"""

import cv2
import numpy as np

from config import DEBUG_TIMING, FACE_CASCADE_PATH

fbRange = [4000, 6000]

faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# The connected Tello instance — injected by main.py after connecting,
# because trackFace() sends its own RC commands.
_drone = None


def set_drone(drone):
    """Store the connected Tello instance (called once by main.py)."""
    global _drone
    _drone = drone


def findFace(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    if len(faces) == 0:
        return img, [[0, 0], 0]

    biggest = None
    maxArea = 0
    for (x, y, fw, fh) in faces:
        area = fw * fh
        if area > maxArea:
            biggest = (x, y, fw, fh)
            maxArea = area

    if biggest is not None:
        x, y, fw, fh = biggest
        cx = x + fw // 2
        cy = y + fh // 2
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return img, [[cx, cy], maxArea]

    return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    """
    Face-tracking PID that sends its own RC directly.

    Returns a tuple of (lr, fb, ud, yv, error) — the RC values that were
    actually sent plus the tracking error for the next PID iteration.
    The caller should feed the RC values into PositionTracker.update()
    so dead reckoning accounts for face-tracking movement too.
    """
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - (pError if pError is not None else 0))
    speed = int(np.clip(speed * 1.5, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    sent_lr, sent_fb, sent_ud, sent_yv = 0, fb, 0, speed
    if DEBUG_TIMING:  # 30 prints/s stalls the Windows console + OpenCV
        print("Fb", fb, "ud", 0, "yaw", speed)
    _drone.send_rc_control(sent_lr, sent_fb, sent_ud, sent_yv)
    return (sent_lr, sent_fb, sent_ud, sent_yv, error)
