"""
MediaPipe Pose "hand up" gate.

The hand-gesture classifier is expensive to run, so it should only be
invoked when someone is actually signalling. This module answers one
cheap question on the full frame: "is a hand raised above the shoulder?"

The main flight loop uses the returned wrist bbox to crop the region
around the raised hand and feed only that crop to the gesture model —
no more fixed 130x130 box. When no hand is up, the gesture model is
skipped entirely (GestureDetector loads it lazily on first use).

Converted from a standalone webcam script into a module so the drone
loop can reuse one Pose instance (creation is expensive).
"""

import cv2
import mediapipe as mp

# Hand-up detection constants (tuned on the original webcam script)
SHOULDER_HEIGHT_RATIO = 0.85  # Wrist must be above 85% of shoulder height
MIN_CONFIDENCE = 0.5

# Crop geometry: square around the raised wrist, sized relative to the
# person's shoulder width so it adapts to camera distance.
HAND_CROP_SHOULDER_SCALE = 2.2  # crop side = 2.2x shoulder width
HAND_CROP_UP_SHIFT = 0.35       # center the crop above the wrist


class PoseHandGate:
    """Wraps a single MediaPipe Pose instance (reuse — creation is slow)."""

    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw = mp.solutions.drawing_utils
        self._last_had_person = False

    def process(self, frame):
        """Analyze a BGR frame.

        Returns (info, vis):
          info — dict with "which" ("left"/"right") and "bbox" (x1, y1,
                 x2, y2) of the raised hand, or None when no hand is
                 raised (also None when no person is detected)
          vis  — copy of the frame with pose landmarks drawn (preview)
        """
        h, w = frame.shape[:2]
        vis = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        self._last_had_person = results.pose_landmarks is not None

        if not results.pose_landmarks:
            return None, vis

        lm = results.pose_landmarks.landmark
        pose_lm = mp.solutions.pose.PoseLandmark
        self._draw.draw_landmarks(
            vis, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        left_shoulder = lm[pose_lm.LEFT_SHOULDER]
        right_shoulder = lm[pose_lm.RIGHT_SHOULDER]
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        # Shoulder width in pixels — sizes the hand crop.
        shoulder_dx = abs(right_shoulder.x - left_shoulder.x) * w
        if shoulder_dx < 1:
            shoulder_dx = w / 3  # degenerate pose; conservative default

        left_wrist = lm[pose_lm.LEFT_WRIST]
        right_wrist = lm[pose_lm.RIGHT_WRIST]

        # Hand is "up" when the wrist clears the shoulder line with enough
        # tracking confidence (lower y = higher on screen).
        left_up = (
            left_wrist.y < avg_shoulder_y * SHOULDER_HEIGHT_RATIO
            and left_wrist.visibility > MIN_CONFIDENCE
        )
        right_up = (
            right_wrist.y < avg_shoulder_y * SHOULDER_HEIGHT_RATIO
            and right_wrist.visibility > MIN_CONFIDENCE
        )

        if not left_up and not right_up:
            return None, vis

        # The gesture model expects a single hand — prefer the more
        # confidently tracked wrist when both are up.
        if left_up and not right_up:
            wrist, which = left_wrist, "left"
        elif right_up and not left_up:
            wrist, which = right_wrist, "right"
        elif right_wrist.visibility >= left_wrist.visibility:
            wrist, which = right_wrist, "right"
        else:
            wrist, which = left_wrist, "left"

        # Square crop around the wrist, biased upward so the hand (not the
        # forearm) dominates, clamped to frame bounds.
        side = int(shoulder_dx * HAND_CROP_SHOULDER_SCALE)
        cx = int(wrist.x * w)
        cy = int(wrist.y * h) - int(side * HAND_CROP_UP_SHIFT)
        x1 = max(0, min(w - 1, cx - side // 2))
        y1 = max(0, min(h - 1, cy - side // 2))
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        cv2.putText(vis, f"HAND UP ({which})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return {"which": which, "bbox": (x1, y1, x2, y2)}, vis

    def is_person_present(self) -> bool:
        """Return True if the last process() call detected a person.

        This is a cheap read — it does NOT re-run MediaPipe.  Callers
        (e.g. the voice-listening thread) can poll this to gate
        continuous VAD listening without adding frame-processing load.
        """
        return self._last_had_person

    def close(self):
        self._pose.close()
