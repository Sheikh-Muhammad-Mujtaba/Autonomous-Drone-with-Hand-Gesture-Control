"""Low-latency drone video reader (single latest BGR frame)."""

import threading
import time

import cv2


class LowLatencyFrameRead:
    """
    Seconds of delay = queued UDP/H.264 frames. This reader:
    - uses a tiny UDP fifo so old packets are dropped (overrun_nonfatal)
    - OpenCV buffer size 1
    - background thread overwrites one BGR frame; display never walks a queue
    Do NOT also call me.get_frame_read().
    """

    def __init__(self, address: str):
        if "?" not in address:
            # Bigger fifo so P-frames still have their preceding PPS/keyframe
            # context — fifo_size=500 was dropping frames that decoded fine,
            # causing "non-existing PPS" errors and visible stutter.
            address = address + "?overrun_nonfatal=1&fifo_size=3000"
        self.cap = cv2.VideoCapture(address, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print("VIDEO OPEN FAILED:", address)
        else:
            print("Video: low-latency OpenCV (drop old frames)")
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tello-video")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if not self.cap.isOpened():
                time.sleep(0.02)
                continue
            try:
                # grab() is cheap; retrieve only the last grabbed frame this tick.
                if not self.cap.grab():
                    time.sleep(0.001)
                    continue
                # Drain fewer extra grabs — 6 was too aggressive and discarded
                # frames whose PPS/keyframe reference we still needed.
                for _ in range(2):
                    if not self.cap.grab():
                        break
                if self._stop.is_set():
                    break
                ok, frame = self.cap.retrieve()
            except cv2.error:
                # Decoder hiccups (PPS loss, buffer races with stop()) throw
                # opaque C++ exceptions out of grab()/retrieve(). A dead video
                # thread means no feed at all — swallow and retry instead.
                time.sleep(0.02)
                continue
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
        try:
            self.cap.release()
        except Exception:
            pass
        self._thread.join(timeout=1.0)
