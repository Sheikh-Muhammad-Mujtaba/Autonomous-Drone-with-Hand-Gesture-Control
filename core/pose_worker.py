"""Pose gate worker — MediaPipe Pose off the display+RC thread."""

import threading

from perception.body_pose import PoseHandGate


class PoseWorker:
    """MediaPipe Pose gate off the display+RC thread. Always runs on the
    full frame (cheap); its result decides whether the gesture model
    runs at all. The raised-hand bbox becomes the gesture crop — no
    fixed box."""

    def __init__(self):
        self._gate = PoseHandGate()
        self._lock = threading.Lock()
        self._frame = None
        self._hand_info = None
        self._vis = None
        self._person_present = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pose-worker")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._gate.close()

    def submit_frame(self, frame):
        with self._lock:
            self._frame = frame

    def latest(self):
        with self._lock:
            return self._hand_info, self._vis

    def is_person_present(self) -> bool:
        """True if the last pose frame found a person (any pose, not just hand-up)."""
        with self._lock:
            return self._person_present

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                self._stop.wait(0.001)
                continue
            hand_info, vis = self._gate.process(frame)
            with self._lock:
                self._hand_info = hand_info
                self._vis = vis
                self._person_present = self._gate.is_person_present()
