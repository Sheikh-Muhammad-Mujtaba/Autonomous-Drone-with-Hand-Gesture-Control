"""Gesture classifier worker — MediaPipe/Keras off the display+RC thread."""

import threading

import cv2
import numpy as np

from config import GESTURE_CROP_SIZE
from perception.gesture_detector import PredictClass


def square_box_in_frame(cx, cy, side, frame_w, frame_h):
    """Square (x1, y1, x2, y2) of `side` px centred on (cx, cy).

    The box is shifted (not shrunk) to stay inside the frame, so the crop
    stays square and resizing it never distorts the hand. `side` is capped
    at the frame's shorter edge.
    """
    side = int(max(1, min(side, frame_w, frame_h)))
    x1 = int(min(max(cx - side // 2, 0), frame_w - side))
    y1 = int(min(max(cy - side // 2, 0), frame_h - side))
    return x1, y1, x1 + side, y1 + side


def hand_crop_from_raw(raw, display_bbox, display_size):
    """Cut the gesture crop from the full-resolution frame.

    The pose gate works on the small display image (e.g. 360x240). Cropping
    there left the hand at ~60-90 px, upscaled to 320 px — too blurry for
    MediaPipe Hands. Scale the bbox up to the raw frame instead (the display
    image is a non-uniform resize, so the box is re-squared in raw pixels).

    Returns:
        (crop resized to GESTURE_CROP_SIZE, raw crop box, (raw_w, raw_h)).
    """
    raw_h, raw_w = raw.shape[:2]
    disp_w, disp_h = display_size
    sx, sy = raw_w / float(disp_w), raw_h / float(disp_h)
    x1, y1, x2, y2 = display_bbox
    cx = int((x1 + x2) / 2.0 * sx)
    cy = int((y1 + y2) / 2.0 * sy)
    side = max((x2 - x1) * sx, (y2 - y1) * sy)
    box = square_box_in_frame(cx, cy, side, raw_w, raw_h)
    bx1, by1, bx2, by2 = box
    crop = cv2.resize(raw[by1:by2, bx1:bx2], (GESTURE_CROP_SIZE, GESTURE_CROP_SIZE))
    return crop, box, (raw_w, raw_h)


class GestureWorker:
    """MediaPipe/Keras off the display+RC thread; only runs when the
    pose gate reports a raised hand (submit_crop is only called then)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._crop = None
        self._crop_geometry = (None, None)
        self._gesture = None
        self._vis = np.zeros((GESTURE_CROP_SIZE, GESTURE_CROP_SIZE, 3), dtype=np.uint8)
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gesture-worker")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def submit_crop(self, crop, crop_box=None, frame_size=None):
        """Queue the newest hand crop (latest wins).

        crop_box / frame_size locate the crop in the full frame so the
        classifier sees landmarks in its training coordinate space.
        """
        with self._lock:
            self._crop = crop
            self._crop_geometry = (crop_box, frame_size)

    def latest(self):
        with self._lock:
            return self._seq, self._gesture, self._vis

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                crop = self._crop
                crop_box, frame_size = self._crop_geometry
                self._crop = None
            if crop is None:
                self._stop.wait(0.001)
                continue
            gesture, vis = PredictClass(crop, crop_box, frame_size)
            with self._lock:
                self._gesture = gesture
                self._vis = vis
                self._seq += 1
