"""No-drone model verification.

Loads every production model exactly the way the flight program loads
them — same modules, same paths, same parameters — and prints only the
results: load status, one sample inference per model, and timing.

Nothing here talks to a drone, opens the GUI, or touches the microphone.

Run from the project root (or anywhere — the path bootstrap handles it):
    py -3.10 test_script\\test_models.py
    py -3.10 test_script\\test_models.py --skip-voice
    py -3.10 test_script\\test_models.py --video test_video\\test1.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running directly from anywhere: put the project root on sys.path
# so `config` and the package imports resolve.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Silence TF / MediaPipe banner spam the same way main.py does.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import numpy as np

from config import (
    FACE_CASCADE_PATH,
    GESTURE_MODEL_PATH,
    LABEL_ENCODER_PATH,
    PROJECT_ROOT,
    YOLO_MODEL_PATH,
)

RESULTS = []  # (name, ok, detail)


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def check_model_files():
    """All model artifacts must exist under model/."""
    files = {
        "gesture model": GESTURE_MODEL_PATH,
        "label encoder": LABEL_ENCODER_PATH,
        "yolo weights": YOLO_MODEL_PATH,
        "face cascade": FACE_CASCADE_PATH,
    }
    missing = [n for n, p in files.items() if not os.path.isfile(p)]
    if missing:
        report("model files", False, "missing: " + ", ".join(missing))
        return
    report("model files", True, "all 4 artifacts present under model/")


def test_gesture_engine():
    """Same load path as PredictClass() on its first call: Keras model +
    label encoder + MediaPipe Hands, then a warm-up inference."""
    try:
        import perception.gesture_detector as gd

        t0 = time.perf_counter()
        gd._ensure_gesture_engine()
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = gd.infer_landmarks(np.zeros((1, 42), dtype=np.float32))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        idx = int(np.argmax(pred))
        sign = gd.le.inverse_transform([idx])[0]
        conf = float(pred[0][idx])
        report(
            "gesture engine",
            True,
            f"loaded in {load_s:.1f}s; warm-up inference {infer_ms:.1f}ms "
            f"-> sign '{sign}' ({conf:.3f}) on empty input",
        )
    except Exception as exc:
        report("gesture engine", False, f"{type(exc).__name__}: {exc}")


def test_pose_gate():
    """Same construction as PoseWorker: one MediaPipe Pose instance."""
    try:
        from perception.body_pose import PoseHandGate

        t0 = time.perf_counter()
        gate = PoseHandGate()
        load_s = time.perf_counter() - t0

        info, _vis = gate.process(np.zeros((240, 360, 3), dtype=np.uint8))
        report(
            "pose gate",
            info is None,
            f"loaded in {load_s:.1f}s; blank frame -> hand={info} "
            "(expected: None)",
        )
    except Exception as exc:
        report("pose gate", False, f"{type(exc).__name__}: {exc}")


def test_yolo(video_path=None):
    """Same YOLO + ByteTrack call as the reporter worker (person only)."""
    try:
        from ultralytics import YOLO

        from perception.reporter_tracker import (
            CONFIDENCE,
            DEVICE,
            IMAGE_SIZE,
            IOU,
            TRACKER_CFG,
            YOLO_MODEL,
        )

        t0 = time.perf_counter()
        model = YOLO(YOLO_MODEL)
        load_s = time.perf_counter() - t0

        if video_path:
            _run_yolo_on_video(
                model, video_path, load_s, TRACKER_CFG,
                CONFIDENCE, IOU, IMAGE_SIZE, DEVICE,
            )
            return

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model.track(
            frame, persist=True, tracker=TRACKER_CFG, classes=[0],
            conf=CONFIDENCE, iou=IOU, imgsz=IMAGE_SIZE, device=DEVICE,
            verbose=False,
        )
        boxes = results[0].boxes
        n = 0 if boxes is None or boxes.id is None else len(boxes.id)
        report(
            "yolo person tracking",
            n == 0,
            f"loaded in {load_s:.1f}s; blank frame -> {n} persons "
            "(expected: 0)",
        )
    except Exception as exc:
        report("yolo person tracking", False, f"{type(exc).__name__}: {exc}")


def _run_yolo_on_video(model, video_path, load_s, tracker, conf, iou,
                       imgsz, device):
    """Feed up to 60 video frames through the exact production .track()
    call (no VLM / API keys needed) and summarize person detections."""
    import cv2

    if not os.path.isfile(video_path):
        report("yolo person tracking", False, f"video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total = 0
    frames_with_people = 0
    seen_ids = set()
    t0 = time.perf_counter()

    while cap.isOpened() and total < 60:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        results = model.track(
            frame, persist=True, tracker=tracker, classes=[0],
            conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            frames_with_people += 1
            seen_ids.update(int(i) for i in boxes.id.cpu().numpy())
    cap.release()
    wall_s = time.perf_counter() - t0
    report(
        "yolo person tracking",
        total > 0,
        f"loaded in {load_s:.1f}s; video: {total} frames in {wall_s:.1f}s "
        f"({total / wall_s:.1f} fps), people in {frames_with_people} frames, "
        f"track IDs seen: {sorted(seen_ids) or 'none'}",
    )


def test_face_cascade():
    """Same module-level cascade load as the face-tracking PID."""
    try:
        import core.face_tracking as ft

        img = np.zeros((240, 360), dtype=np.uint8)
        faces = ft.faceCascade.detectMultiScale(img, 1.2, 8)
        report(
            "face cascade",
            len(faces) == 0,
            f"loaded; blank frame -> {len(faces)} faces (expected: 0)",
        )
    except Exception as exc:
        report("face cascade", False, f"{type(exc).__name__}: {exc}")


def test_voice_pipeline():
    """Production loads faster-whisper at import time — importing
    control.voice_model IS the load test. First run downloads the weights
    into model/faster-whisper-medium/ (one-time, needs internet)."""
    try:
        t0 = time.perf_counter()
        import control.voice_model as vm  # noqa: F401  (import = load)
        load_s = time.perf_counter() - t0
        report(
            "voice pipeline (whisper)",
            True,
            f"loaded in {load_s:.1f}s ({vm.WHISPER_MODEL_SIZE}, "
            f"{vm.WHISPER_DEVICE}); VAD + LLM + TTS modules OK",
        )
    except Exception as exc:
        report("voice pipeline (whisper)", False, f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="No-drone model verification")
    parser.add_argument(
        "--skip-voice", action="store_true",
        help="skip the heavy faster-whisper load",
    )
    parser.add_argument(
        "--video", metavar="PATH",
        help="also run the production YOLO track() call on the first "
             "60 frames of a video",
    )
    args = parser.parse_args()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.version.split()[0]}")
    print()

    check_model_files()
    test_gesture_engine()
    test_pose_gate()
    test_yolo(args.video)
    test_face_cascade()
    if not args.skip_voice:
        test_voice_pipeline()

    print()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 60)
    if failed:
        print(f"RESULT: {len(failed)} check(s) failed -> {', '.join(failed)}")
        sys.exit(1)
    print(f"RESULT: all {len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
