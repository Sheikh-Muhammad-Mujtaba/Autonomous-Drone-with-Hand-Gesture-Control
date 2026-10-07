"""Gesture classifier worker — MediaPipe/Keras off the display+RC thread."""

import threading

import numpy as np

from config import GESTURE_CROP_SIZE
from perception.gesture_detector import PredictClass


class GestureWorker:
    """MediaPipe/Keras off the display+RC thread; only runs when the
    pose gate reports a raised hand (submit_crop is only called then)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._crop = None
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

    def submit_crop(self, crop):
        with self._lock:
            self._crop = crop

    def latest(self):
        with self._lock:
            return self._seq, self._gesture, self._vis

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                crop = self._crop
                self._crop = None
            if crop is None:
                self._stop.wait(0.001)
                continue
            gesture, vis = PredictClass(crop)
            with self._lock:
                self._gesture = gesture
                self._vis = vis
                self._seq += 1
