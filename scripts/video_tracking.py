import sys
from pathlib import Path

# Allow running directly (python scripts/video_tracking.py) — put the
# project root on sys.path so the config import resolves.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv
from PIL import Image

from config import YOLO_MODEL_PATH

import cv2
import os
import json
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "test_video/test2.mp4 "

OUTPUT_VIDEO = "output/tracked_video.mp4"

YOLO_MODEL = YOLO_MODEL_PATH

GEMINI_MODEL = "gemini-3.6-flash"

# ------------------------------------------------------------
# CPU
# ------------------------------------------------------------

DEVICE = "cpu"


# ============================================================
# YOLO SETTINGS
# ============================================================

# Lower confidence helps prevent losing people
# when detection confidence temporarily drops.
CONFIDENCE = 0.35

IOU = 0.50

# 640 is faster on CPU.
IMAGE_SIZE = 640


# ============================================================
# BYTE TRACK SETTINGS
# ============================================================

TRACKER = "custom_bytetrack.yaml"


# ============================================================
# GEMINI SETTINGS
# ============================================================

# Gemini checks the microphone every 2 seconds.
VLM_INTERVAL_SECONDS = 2.0

MIN_VLM_CONFIDENCE = 0.70

MAX_VLM_WIDTH = 1280


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    raise RuntimeError(
        "GEMINI_API_KEY is not configured in .env"
    )


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(YOLO_MODEL)

print("YOLO model loaded.")


# ============================================================
# LOAD GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=api_key
)

print("Gemini client initialized.")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_directory = os.path.dirname(
    OUTPUT_VIDEO
)

if output_directory:

    os.makedirs(
        output_directory,
        exist_ok=True
    )


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("========================================")
print("VIDEO INFORMATION")
print("========================================")

print(
    f"Resolution : "
    f"{frame_width} x {frame_height}"
)

print(
    f"FPS        : "
    f"{fps:.2f}"
)

print(
    f"Frames     : "
    f"{total_frames}"
)

print(
    f"Duration   : "
    f"{duration:.2f} seconds"
)

print("========================================")
print()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (
        frame_width,
        frame_height
    )
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# MICROPHONE STATE
# ============================================================

microphone_track_id = None

microphone_confidence = 0.0

last_vlm_time = -999999


# ============================================================
# TRACK HISTORY
# ============================================================

# Stores the last known position of every Track ID.

track_history = {}


# ============================================================
# GEMINI FUNCTION
# ============================================================

def find_microphone_holder(frame):

    """
    Send an annotated frame to Gemini.

    Gemini identifies which EXISTING Track ID
    is physically holding the microphone.
    """

    # --------------------------------------------------------
    # Resize for Gemini
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    if width > MAX_VLM_WIDTH:

        scale = (
            MAX_VLM_WIDTH / width
        )

        new_width = MAX_VLM_WIDTH

        new_height = int(
            height * scale
        )

        vlm_frame = cv2.resize(
            frame,
            (
                new_width,
                new_height
            )
        )

    else:

        vlm_frame = frame


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        vlm_frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = """
Analyze this video frame carefully.

People have already been detected and tracked
using YOLO and ByteTrack.

Each person has a label:

Track ID 1
Track ID 2
Track ID 3
Track ID 4
etc.

Your task is ONLY to identify which EXISTING
Track ID is physically holding the microphone.

IMPORTANT RULES:

1. Do NOT create a new ID.

2. Only return a Track ID that is visible
   in this image.

3. Carefully locate the microphone.

4. Determine which person's HAND is actually
   holding the microphone.

5. Do NOT select a person just because they
   are close to the microphone.

6. Follow the microphone handle to the person's
   hand.

7. If the microphone is being held toward another
   person's mouth, the person holding the microphone
   is the microphone holder.

8. The person speaking into the microphone is NOT
   necessarily the microphone holder.

9. If the microphone holder cannot be determined
   confidently, return null.

10. If there is no microphone, return null.

Return ONLY valid JSON.

Example:

{
    "microphone_holder_id": 7,
    "confidence": 0.94
}

If uncertain:

{
    "microphone_holder_id": null,
    "confidence": 0.0
}
"""


    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    try:

        response = gemini_client.models.generate_content(

            model=GEMINI_MODEL,

            contents=[
                pil_image,
                prompt
            ]
        )

    except Exception as e:

        print()
        print(
            f"Gemini error: {e}"
        )

        return None, 0.0


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    text = response.text.strip()

    print()
    print("Gemini response:")
    print(text)


    # --------------------------------------------------------
    # Remove markdown
    # --------------------------------------------------------

    text = re.sub(
        r"```json|```",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

        holder_id = data.get(
            "microphone_holder_id"
        )

        confidence = float(
            data.get(
                "confidence",
                0.0
            )
        )

        return (
            holder_id,
            confidence
        )

    except Exception as e:

        print(
            f"Could not parse Gemini JSON: {e}"
        )

        return (
            None,
            0.0
        )


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

start_time = time.time()


print()
print("========================================")
print("STARTING VIDEO PROCESSING")
print("========================================")
print()


while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()

    if not success:

        break


    frame_number += 1


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(

        frame,

        persist=True,

        tracker=TRACKER,

        classes=[0],

        conf=CONFIDENCE,

        iou=IOU,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # CURRENT TRACKS
    # ========================================================

    current_tracks = []


    if (
        result.boxes is not None
        and
        result.boxes.id is not None
    ):

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes
            .conf
            .cpu()
            .numpy()
        )


        for (
            box,
            track_id,
            confidence
        ) in zip(
            boxes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # Store track
            # ------------------------------------------------

            person = {

                "id": int(
                    track_id
                ),

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "confidence": float(
                    confidence
                )
            }


            current_tracks.append(
                person
            )


            # ------------------------------------------------
            # Update track history
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            track_history[
                int(track_id)
            ] = {

                "center": (
                    center_x,
                    center_y
                ),

                "last_frame": (
                    frame_number
                ),

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }


    # ========================================================
    # DRAW TRACKS
    # ========================================================

    for person in current_tracks:

        track_id = person["id"]

        x1, y1, x2, y2 = (
            person["box"]
        )


        # ----------------------------------------------------
        # Is microphone holder?
        # ----------------------------------------------------

        is_microphone_holder = (

            microphone_track_id
            is not None

            and

            track_id
            == microphone_track_id
        )


        # ----------------------------------------------------
        # Microphone holder
        # ----------------------------------------------------

        if is_microphone_holder:

            box_color = (
                0,
                0,
                255
            )

            label = (
                f"ID {track_id} - MIC"
            )

            thickness = 4


        # ----------------------------------------------------
        # Normal person
        # ----------------------------------------------------

        else:

            box_color = (
                0,
                255,
                0
            )

            label = (
                f"ID {track_id}"
            )

            thickness = 2


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (
                x1,
                y1
            ),

            (
                x2,
                y2
            ),

            box_color,

            thickness
        )


        # ----------------------------------------------------
        # Draw label
        # ----------------------------------------------------

        cv2.putText(

            frame,

            label,

            (
                x1,
                max(
                    y1 - 10,
                    25
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            box_color,

            2
        )


    # ========================================================
    # CURRENT VIDEO TIME
    # ========================================================

    current_video_time = (
        frame_number / fps
    )


    # ========================================================
    # GEMINI CHECK
    # ========================================================

    should_check_vlm = (

        current_video_time
        -
        last_vlm_time

        >=

        VLM_INTERVAL_SECONDS
    )


    if should_check_vlm:

        if len(current_tracks) > 0:

            print()
            print(
                "----------------------------------------"
            )

            print(
                f"VLM CHECK: "
                f"{current_video_time:.2f}s"
            )

            print(
                "Current Track IDs: "
                +
                str(
                    [
                        p["id"]
                        for p in current_tracks
                    ]
                )
            )


            # ------------------------------------------------
            # Copy frame
            # ------------------------------------------------

            vlm_frame = frame.copy()


            # ------------------------------------------------
            # Ask Gemini
            # ------------------------------------------------

            (
                holder_id,
                confidence
            ) = find_microphone_holder(
                vlm_frame
            )


            # ------------------------------------------------
            # Existing Track IDs
            # ------------------------------------------------

            valid_ids = {

                p["id"]

                for p in current_tracks
            }


            # ------------------------------------------------
            # Valid Gemini result
            # ------------------------------------------------

            if (

                holder_id is not None

                and

                holder_id in valid_ids

                and

                confidence
                >=
                MIN_VLM_CONFIDENCE
            ):

                microphone_track_id = int(
                    holder_id
                )

                microphone_confidence = (
                    confidence
                )


                print()
                print(
                    "========================================"
                )

                print(
                    f"MICROPHONE HOLDER: "
                    f"ID {microphone_track_id}"
                )

                print(
                    f"Confidence: "
                    f"{microphone_confidence:.2f}"
                )

                print(
                    "========================================"
                )


            # ------------------------------------------------
            # Gemini says no holder
            # ------------------------------------------------

            elif holder_id is None:

                print(
                    "Gemini could not identify "
                    "a microphone holder."
                )


            # ------------------------------------------------
            # Invalid result
            # ------------------------------------------------

            else:

                print(
                    "Gemini returned an invalid "
                    "or low-confidence ID."
                )


            # ------------------------------------------------
            # Update VLM timer
            # ------------------------------------------------

            last_vlm_time = (
                current_video_time
            )


    # ========================================================
    # STATUS
    # ========================================================

    status = (

        f"Frame "
        f"{frame_number}/"
        f"{total_frames}"
    )


    if microphone_track_id is not None:

        status += (

            f" | MIC ID: "
            f"{microphone_track_id}"
        )


    cv2.putText(

        frame,

        status,

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            255,
            255
        ),

        2
    )


    # ========================================================
    # WRITE OUTPUT FRAME
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:

        elapsed = (
            time.time()
            -
            start_time
        )


        if elapsed > 0:

            processing_fps = (
                frame_number
                /
                elapsed
            )

        else:

            processing_fps = 0


        if total_frames > 0:

            progress = (
                frame_number
                /
                total_frames
                *
                100
            )

        else:

            progress = 0


        print(

            f"\rProgress: "
            f"{progress:.1f}% | "

            f"Processing FPS: "
            f"{processing_fps:.2f}",

            end=""
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()


# ============================================================
# SUMMARY
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


print()
print()
print(
    "========================================"
)

print(
    "PROCESSING COMPLETED"
)

print(
    "========================================"
)

print(
    f"Frames processed : "
    f"{frame_number}"
)

print(
    f"Processing time  : "
    f"{elapsed:.2f} seconds"
)


if elapsed > 0:

    print(
        f"Average FPS     : "
        f"{frame_number / elapsed:.2f}"
    )


print(
    f"Output video     : "
    f"{OUTPUT_VIDEO}"
)


if microphone_track_id is not None:

    print(
        f"Last MIC Track ID: "
        f"{microphone_track_id}"
    )

else:

    print(
        "No microphone holder identified."
    )


print(
    "========================================"
)