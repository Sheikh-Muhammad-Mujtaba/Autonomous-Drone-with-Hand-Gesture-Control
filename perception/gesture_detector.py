# Silence MediaPipe / TF / protobuf before those libraries load.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import logging
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype",
    category=UserWarning,
)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

from config import (
    GESTURE_CONFIDENCE_THRESHOLD,
    GESTURE_MODEL_PATH,
    HAND_DETECTION_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
    LABEL_ENCODER_PATH,
)

MODEL_PATH = GESTURE_MODEL_PATH
ENCODER_PATH = LABEL_ENCODER_PATH
CONFIDENCE_THRESHOLD = GESTURE_CONFIDENCE_THRESHOLD

model = None  # lazy — loaded on first gesture via _ensure_gesture_engine
le = None  # lazy — loaded on first gesture via _ensure_gesture_engine

label_map = {
    'A': 'stop',
    'B': 'up',
    'C': 'down',
    'D': 'left',
    'E': 'right',
    'F': 'flip',
    'G': 'turn left',
    'H': 'turn right',
    'I': 'move backward',
    'J': 'move farward'
    # 'Q' (a real flip, only 96 training samples) is deliberately unmapped:
    # PredictClass reports it as None so it can never drive the drone.
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = None  # lazy — created on first gesture via _ensure_gesture_engine

# Filled each PredictClass call so the main loop can log stages.
last_timing_ms = {"mediapipe": 0.0, "inference": 0.0}


def _ensure_gesture_engine():
    """
    Load the Keras model + label encoder + MediaPipe Hands on first use.

    The main loop only calls PredictClass when the pose gate reports a
    raised hand, so sessions without any gesture never pay the model load
    cost (a few seconds + memory) at all.
    """
    global model, le, hands
    if model is None:
        print("[Gesture] Loading hand gesture model (first use)...")
        model = load_model(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
            model_complexity=0,  # lighter graph; complexity 1 was a large part of the ~28ms
        )
        # Warm up so the first live frame does not pay trace cost.
        _infer_tf(tf.zeros((1, 42), dtype=tf.float32))
    return model, le


@tf.function(reduce_retracing=True)
def _infer_tf(x):
    # First trace happens inside _ensure_gesture_engine after the model
    # is loaded, so `model` is never None here.
    return model(x, training=False)


def infer_landmarks(sample_landmarks):
    """Fast path: compiled graph. Input shape (1, 42)."""
    pred = _infer_tf(tf.convert_to_tensor(sample_landmarks, dtype=tf.float32))
    return np.asarray(pred)


def _landmark_features(hand_landmarks, crop_box, frame_size):
    """Flatten 21 landmarks into the (1, 42) x/y vector the model expects.

    The model was trained on landmarks from the WHOLE mirrored webcam frame
    (see custom_hand_gesture_model_training/collect data.py), where a hand
    spans ~12% of the width. MediaPipe here runs on a tight hand crop, where
    the hand fills most of the image — a completely different input range
    that drove confidence below the threshold. When the crop's position in
    the original frame is known, map every landmark back into mirrored
    full-frame coordinates so the model sees the distribution it learned.

    Args:
        hand_landmarks: MediaPipe landmarks, normalized to the flipped crop.
        crop_box: (x1, y1, x2, y2) of the crop in the unflipped full frame,
            in pixels, or None to use crop-relative coordinates.
        frame_size: (width, height) of the full frame, or None.
    """
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    if crop_box is not None and frame_size is not None:
        x1, y1, x2, y2 = crop_box
        frame_w, frame_h = frame_size
        # The crop was mirrored before MediaPipe; in the mirrored full frame
        # its left edge sits at frame_w - x2.
        pts[:, 0] = (frame_w - x2 + pts[:, 0] * (x2 - x1)) / frame_w
        pts[:, 1] = (y1 + pts[:, 1] * (y2 - y1)) / frame_h
    return pts.reshape(1, -1)


def PredictClass(frame, crop_box=None, frame_size=None):
    """Classify the hand sign in a BGR hand crop.

    Args:
        frame: BGR hand crop (any size).
        crop_box: optional (x1, y1, x2, y2) of the crop in the full frame.
        frame_size: optional (width, height) of the full frame. Pass both
            so landmarks match the training coordinate space.

    Returns:
        (gesture or None, crop with landmarks drawn).
    """
    _ensure_gesture_engine()  # lazy load; first call pays the load cost

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    result = hands.process(framergb)
    last_timing_ms["mediapipe"] = (time.perf_counter() - t0) * 1000.0
    last_timing_ms["inference"] = 0.0

    if not result.multi_hand_landmarks:
        return None, frame

    hand_landmarks = result.multi_hand_landmarks[0]  # max_num_hands=1
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    sample_landmarks = _landmark_features(hand_landmarks, crop_box, frame_size)

    t1 = time.perf_counter()
    prediction = infer_landmarks(sample_landmarks)
    last_timing_ms["inference"] = (time.perf_counter() - t1) * 1000.0

    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[0][predicted_class])
    if confidence < CONFIDENCE_THRESHOLD:
        return None, frame

    sign = le.inverse_transform([predicted_class])[0]
    gesture = label_map.get(sign)
    if gesture is None:
        return None, frame

    cv2.putText(frame, f"{gesture} {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return gesture, frame
