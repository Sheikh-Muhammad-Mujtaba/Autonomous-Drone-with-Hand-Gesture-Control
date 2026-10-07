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

from config import GESTURE_MODEL_PATH, LABEL_ENCODER_PATH

MODEL_PATH = GESTURE_MODEL_PATH
ENCODER_PATH = LABEL_ENCODER_PATH
CONFIDENCE_THRESHOLD = 0.85

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
            min_detection_confidence=0.7,
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


def PredictClass(frame):
    _ensure_gesture_engine()  # lazy load; first call pays the load cost

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    result = hands.process(framergb)
    last_timing_ms["mediapipe"] = (time.perf_counter() - t0) * 1000.0
    last_timing_ms["inference"] = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            sample_landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)

            t1 = time.perf_counter()
            prediction = infer_landmarks(sample_landmarks)
            last_timing_ms["inference"] = (time.perf_counter() - t1) * 1000.0

            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            if confidence < CONFIDENCE_THRESHOLD:
                return None, frame

            predicted_gesture_old = le.inverse_transform([predicted_class])[0]
            predicted_gesture_new = label_map.get(predicted_gesture_old, predicted_gesture_old)

            cv2.putText(frame, f"Gesture: {predicted_gesture_new}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return predicted_gesture_new, frame

    return None, frame
