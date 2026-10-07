"""
calibrate_distance.py

Purpose:
    One-time calibration helper. Stand exactly 1 meter away from the
    Tello drone's camera, run this script, press SPACE to capture a
    photo, and it will:
      1. Detect your face with the Haar cascade
      2. Draw the bounding box on the photo
      3. Save the photo to disk so you can visually confirm the box
      4. Print the detected face pixel width
      5. Compute and print the camera's focal_length using the
         known 1 meter distance, so you can paste that value into
         FaceTracker.py for real-world distance estimation.

Usage:
    python scripts/caliberation.py

Controls:
    SPACE - capture the current frame and run detection
    q     - quit
"""

import sys
from pathlib import Path

# Allow running directly (python scripts/caliberation.py) — put the
# project root on sys.path so the config import resolves.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import time
import cv2
from djitellopy import tello

from config import CALIBRATION_PHOTOS_DIR, FACE_CASCADE_PATH

# ---- Calibration constants ----
KNOWN_DISTANCE_CM = 100        # you standing exactly 1 meter (100 cm) away
KNOWN_FACE_WIDTH_CM = 14       # average adult face width, adjust if you measure your own

CASCADE_PATH = FACE_CASCADE_PATH
OUTPUT_DIR = CALIBRATION_PHOTOS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

faceCascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_face_width(img):
    """Runs face detection, draws the bounding box, and returns
    (annotated_img, face_width_px) for the largest detected face.
    Returns face_width_px = 0 if no face is found."""
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    if len(faces) == 0:
        return img, 0

    # pick the largest face (closest / most prominent)
    biggest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = biggest

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"width: {w}px", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, w


def main():
    me = tello.Tello()
    me.connect()
    print("Battery:", me.get_battery())

    me.streamon()
    time.sleep(2)  # let the video stream start properly

    print("\n=== Distance Calibration ===")
    print(f"Stand exactly {KNOWN_DISTANCE_CM} cm from the drone's camera.")
    print("Press SPACE to capture a photo and measure your face width.")
    print("Press q to quit.\n")

    try:
        while True:
            frame_read = me.get_frame_read()
            img = frame_read.frame

            if img is None:
                continue

            preview = img.copy()
            cv2.putText(preview, "Press SPACE to capture, q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Calibration - Live Feed", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Capture and run detection on this frame
                annotated, face_width_px = detect_face_width(img.copy())

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join(OUTPUT_DIR, f"calib_{timestamp}.jpg")
                cv2.imwrite(photo_path, annotated)

                cv2.imshow("Captured Photo - Bounding Box", annotated)
                print(f"Saved photo: {photo_path}")

                if face_width_px == 0:
                    print("No face detected - try again, make sure you're centered and well lit.\n")
                else:
                    focal_length = (face_width_px * KNOWN_DISTANCE_CM) / KNOWN_FACE_WIDTH_CM
                    print(f"Detected face width: {face_width_px} px")
                    print(f"Calculated FOCAL_LENGTH = {focal_length:.2f}")
                    print("Copy this value into FaceTracker.py as FOCAL_LENGTH.\n")

            elif key == ord('q'):
                break

    finally:
        print("Shutting down...")
        try:
            me.streamoff()
        except Exception as e:
            print("Error stopping stream:", e)
        try:
            me.end()
        except Exception as e:
            print("Error ending connection:", e)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
