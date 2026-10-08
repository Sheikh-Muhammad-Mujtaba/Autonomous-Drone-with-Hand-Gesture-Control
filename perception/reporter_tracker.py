"""
Real-time "reporter" (microphone-holder) identification for the flight loop.

Identification logic is the SAME as video_tracking.py — untouched:
  * YOLO person detection + ByteTrack persistent Track IDs (classes=[0])
    on the drone video feed
  * Every VLM_INTERVAL_SECONDS the current annotated frame is sent to a
    vision LLM, which maps the physical microphone holder onto an
    EXISTING Track ID
  * Result accepted only when the ID is currently visible AND confidence
    >= MIN_VLM_CONFIDENCE (0.70)

VLM provider chain (same idea as the voice pipeline):
  1. Gemini VLM first
  2. Groq VLM second — fast/free (qwen vision model, hard timeout)
  3. OpenRouter VLM third — google/gemma-4-31b-it:free
  4. If all three fail -> (None, 0.0) = "no holder identified this round"

The heavy YOLO + VLM work runs on a dedicated worker thread (same
latest-frame-dropping pattern as the pose/gesture workers) so the 30 Hz
RC + display loop never blocks on inference.

The flight loop then applies the "reporter guard":
  * gestures: a raised hand only counts when it geometrically belongs to
    the reporter's person box (the pose gate still runs as before)
  * voice:    non-critical commands only when the reporter is present
              (takeoff / land / hover / emergency stay ungated)
"""

import base64
import concurrent.futures
import json
import os
import re
import threading
import time

import cv2

from config import (
    REPORTER_IMAGE_SIZE,
    REPORTER_MAX_FPS,
    REPORTER_TORCH_THREADS,
    YOLO_MODEL_PATH,
)


# --------------------------------------------------------------------------
# YOLO / tracking config (same values as video_tracking.py)
# --------------------------------------------------------------------------
YOLO_MODEL = YOLO_MODEL_PATH
TRACKER_CFG = "custom_bytetrack.yaml"
DEVICE = "cpu"

CONFIDENCE = 0.35
IOU = 0.50
IMAGE_SIZE = REPORTER_IMAGE_SIZE  # video_tracking.py uses 640; see config.py

VLM_INTERVAL_SECONDS = 2.0
MIN_VLM_CONFIDENCE = 0.70
MAX_VLM_WIDTH = 1280

# --------------------------------------------------------------------------
# VLM provider chain: Gemini first, Groq second, OpenRouter (Gemma 4) last.
# Groq and OpenRouter are OpenAI-compatible and cheap/free.
# --------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_VLM_MODEL_NAME = "qwen/qwen3.8-27b"  # Groq's vision-capable model
GROQ_VLM_TIMEOUT_SECONDS = 4.0            # hard per-request budget

# Tried in order; on any error (e.g. 503 "high demand") the next model is
# used before falling back to Groq. Lite models are less congested.
GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)
# Hard HTTP budget for the Gemini fallback so a stuck request can never
# freeze the tracking thread (default in the SDK is 300 s!).
GEMINI_HTTP_TIMEOUT_S = 20.0

# OpenRouter fallback (OpenAI-compatible) — "Gemma 4" free model.
# Accepts the key under either of these .env names.
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT_SECONDS = 15.0   # free models can queue behind traffic
OPENROUTER_MAX_TOKENS = 256         # tiny JSON answer — stays under free caps


# Prompt shared by ALL providers (identical to video_tracking.py).
_HOLDER_PROMPT = """
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


class ReporterTracker:
    """Tracks people with YOLO/ByteTrack and lets a vision LLM identify
    which Track ID is physically holding the microphone (the "reporter")."""

    def __init__(self, vlm_interval_seconds=VLM_INTERVAL_SECONDS):
        # Lazy imports: heavy vision/LLM packages are only loaded when the
        # reporter guard is actually active.
        try:
            from dotenv import load_dotenv
            from google import genai
            from openai import OpenAI
            from PIL import Image
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Reporter tracking needs: ultralytics, openai, "
                "google-genai, python-dotenv, pillow"
            ) from exc

        load_dotenv()

        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        openrouter_key = (os.getenv("OPEN_ROUTER_API_GAMMA_MODEL")
                          or os.getenv("OPENROUTER_API_KEY"))
        if not groq_key and not gemini_key and not openrouter_key:
            raise RuntimeError(
                "Reporter tracking needs GROQ_API_KEY, GEMINI_API_KEY "
                "and/or OPEN_ROUTER_API_GAMMA_MODEL in .env"
            )

        print("[Reporter] Loading YOLO model...")
        # Leave CPU cores for MediaPipe pose/hands: torch defaults to every
        # core, which doubled gesture latency while YOLO ran.
        import torch
        torch.set_num_threads(REPORTER_TORCH_THREADS)
        self._yolo = YOLO(YOLO_MODEL)

        self._Image = Image
        self._groq = None
        self._gemini = None
        self._openrouter = None
        self._genai_types = genai.types
        if gemini_key:
            self._gemini = genai.Client(
                api_key=gemini_key,
                http_options=genai.types.HttpOptions(
                    # google-genai takes this timeout in MILLISECONDS.
                    timeout=int(GEMINI_HTTP_TIMEOUT_S * 1000),
                ),
            )
            print("[Reporter] Gemini VLM primary ON.")
        if groq_key:
            self._groq = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
            print("[Reporter] Groq VLM fallback ON.")
        if openrouter_key:
            self._openrouter = OpenAI(api_key=openrouter_key,
                                      base_url=OPENROUTER_BASE_URL)
            print("[Reporter] OpenRouter VLM fallback ON.")
        print("[Reporter] YOLO ready.")

        self._vlm_interval = float(vlm_interval_seconds)

        self._lock = threading.Lock()
        self._frame = None
        self._frame_w = 0
        self._frame_h = 0
        self._current_tracks = []   # [{"id", "box", "confidence"}]
        self._track_history = {}
        self._reporter_id = None
        self._reporter_confidence = 0.0
        self._reporter_box = None   # raw-frame box of the reporter track
        self._last_track_t = 0.0    # monotonic time of last YOLO result
        self._last_vlm_t = -1e9
        self._vlm_future = None
        # VLM rounds run here — NOT on the YOLO thread — so a slow API call
        # can never stall person tracking (which keeps the boxes moving).
        self._vlm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reporter-vlm"
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="reporter-worker"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API for the flight loop
    # ------------------------------------------------------------------

    def submit_frame(self, frame):
        """Give the worker the newest raw BGR frame (latest wins)."""
        with self._lock:
            self._frame = frame
            self._frame_w, self._frame_h = frame.shape[1], frame.shape[0]

    @property
    def reporter_track_id(self):
        with self._lock:
            return self._reporter_id

    @property
    def reporter_confidence(self):
        with self._lock:
            return self._reporter_confidence

    def is_reporter_present(self, max_age_s=2.0):
        """True when the last YOLO pass still sees the mic holder."""
        with self._lock:
            if self._reporter_id is None:
                return False
            if time.monotonic() - self._last_track_t > max_age_s:
                return False
            return any(
                p["id"] == self._reporter_id for p in self._current_tracks
            )

    def hand_belongs_to_reporter(self, hand_bbox, out_w, out_h,
                                 pad_x=0.35, pad_top=0.6, pad_bottom=0.1,
                                 max_age_s=2.0):
        """Geometric reporter check for the gesture gate.

        hand_bbox is in the pose/gesture image space (out_w x out_h, e.g.
        360x240).  The reporter's YOLO box lives in the raw camera space;
        it is scaled into the pose space (both are stretched from the same
        raw frame) and padded generously so a hand raised above the head
        still counts.  The centre of the raised-hand bbox must land inside.
        """
        with self._lock:
            if self._reporter_id is None or self._reporter_box is None:
                return False
            if time.monotonic() - self._last_track_t > max_age_s:
                return False
            rx1, ry1, rx2, ry2 = self._reporter_box
            raw_w = self._frame_w
            raw_h = self._frame_h
        if raw_w <= 0 or raw_h <= 0:
            return False

        sx = out_w / float(raw_w)
        sy = out_h / float(raw_h)
        box_w = rx2 - rx1
        box_h = ry2 - ry1

        rx1 = (rx1 - box_w * pad_x) * sx
        rx2 = (rx2 + box_w * pad_x) * sx
        ry1 = (ry1 - box_h * pad_top) * sy
        ry2 = (ry2 + box_h * pad_bottom) * sy

        hx = (hand_bbox[0] + hand_bbox[2]) / 2.0
        hy = (hand_bbox[1] + hand_bbox[3]) / 2.0
        return rx1 <= hx <= rx2 and ry1 <= hy <= ry2

    def current_tracks_for(self, out_w, out_h):
        """Return tracked persons with boxes scaled to (out_w, out_h) space.

        Each element: {"id": int, "box": (x1,y1,x2,y2), "is_mic": bool}.
        The box coordinates are in the output image space (e.g. 360x240 for
        the Drone View).  Use this to draw YOLO bounding boxes on any
        display window without dealing with raw-frame scaling manually.
        """
        with self._lock:
            raw_w = self._frame_w
            raw_h = self._frame_h
            if raw_w <= 0 or raw_h <= 0:
                return []
            sx = out_w / float(raw_w)
            sy = out_h / float(raw_h)
            mic_id = self._reporter_id
            result = []
            for p in self._current_tracks:
                x1, y1, x2, y2 = p["box"]
                result.append({
                    "id": p["id"],
                    "box": (int(x1 * sx), int(y1 * sy),
                            int(x2 * sx), int(y2 * sy)),
                    "is_mic": p["id"] == mic_id,
                })
            return result

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._vlm_executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                self._stop.wait(0.005)
                continue
            t0 = time.monotonic()
            try:
                self._process_frame(frame)
            except Exception as exc:
                # A YOLO/VLM hiccup must never kill the worker silently.
                print(f"[Reporter] worker error: {exc}")
                time.sleep(0.05)
            # Rate cap: back-to-back CPU YOLO starves the pose/hand models.
            spare = (1.0 / REPORTER_MAX_FPS) - (time.monotonic() - t0)
            if spare > 0:
                self._stop.wait(spare)

    def _process_frame(self, frame):
        # Same YOLO + ByteTrack call as video_tracking.py (person only).
        results = self._yolo.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            classes=[0],
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )
        result = results[0]

        now = time.monotonic()
        current_tracks = []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, confidence in zip(boxes, track_ids,
                                                 confidences):
                x1, y1, x2, y2 = map(int, box)
                current_tracks.append({
                    "id": int(track_id),
                    "box": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                })
                self._track_history[int(track_id)] = {
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "box": (x1, y1, x2, y2),
                }

        with self._lock:
            self._current_tracks = current_tracks
            self._last_track_t = now
            self._reporter_box = None
            if self._reporter_id is not None:
                for p in current_tracks:
                    if p["id"] == self._reporter_id:
                        self._reporter_box = p["box"]
                        break

        # VLM cadence — the round runs on a SEPARATE executor thread so a
        # slow Groq/Gemini call can NEVER block YOLO tracking.  Keeping the
        # tracker unblocked is what keeps the on-screen bounding boxes
        # following the person instead of freezing.
        vlm_busy = False
        vlm_due = False
        with self._lock:
            vlm_busy = self._vlm_future is not None
            vlm_due = (len(current_tracks) > 0
                       and now - self._last_vlm_t >= self._vlm_interval)
        if vlm_due and not vlm_busy:
            annotated = self._annotate(frame, current_tracks,
                                       self._reporter_id)
            snapshot = list(current_tracks)
            with self._lock:
                if self._vlm_future is None:  # re-check under the lock
                    self._vlm_future = self._vlm_executor.submit(
                        self._vlm_round, annotated, snapshot
                    )

    def _vlm_round(self, annotated, snapshot_tracks):
        """One identification round, off the YOLO thread.  Never raises;
        always resets the in-flight marker and the cadence clock."""
        try:
            holder_id, confidence = self._find_microphone_holder(annotated)
            self._apply_vlm_result(holder_id, confidence, snapshot_tracks)
        except Exception as exc:
            print(f"[Reporter] VLM round error: {exc}")
        finally:
            with self._lock:
                self._vlm_future = None
                self._last_vlm_t = time.monotonic()

    def _apply_vlm_result(self, holder_id, confidence, current_tracks):
        """Same acceptance rule as video_tracking.py: holder must be one of
        the CURRENT tracks and confidence >= MIN_VLM_CONFIDENCE."""
        valid_ids = {p["id"] for p in current_tracks}
        if (holder_id is not None and holder_id in valid_ids
                and confidence >= MIN_VLM_CONFIDENCE):
            with self._lock:
                self._reporter_id = int(holder_id)
                self._reporter_confidence = confidence
                self._reporter_box = None
                for p in current_tracks:
                    if p["id"] == self._reporter_id:
                        self._reporter_box = p["box"]
                        break
            print("========================================")
            print(f"MICROPHONE HOLDER (REPORTER): ID "
                  f"{self._reporter_id}")
            print(f"Confidence: {confidence:.2f}")
            print("========================================")
        elif holder_id is None:
            print("[Reporter] VLM could not identify a "
                  "microphone holder.")
        else:
            print("[Reporter] VLM returned an invalid or "
                  "low-confidence ID.")

    # ------------------------------------------------------------------
    # Drawing (same colors/labels as video_tracking.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate(frame, tracks, reporter_id):
        annotated = frame.copy()
        for person in tracks:
            track_id = person["id"]
            x1, y1, x2, y2 = person["box"]
            is_mic = reporter_id is not None and track_id == reporter_id
            if is_mic:
                box_color = (0, 0, 255)
                label = f"ID {track_id} - MIC"
                thickness = 4
            else:
                box_color = (0, 255, 0)
                label = f"ID {track_id}"
                thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color,
                          thickness)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        return annotated

    # ------------------------------------------------------------------
    # VLM mic-holder identification
    # ------------------------------------------------------------------

    def _find_microphone_holder(self, frame):
        """Identify the mic holder across the provider chain in order:
        Gemini first, then Groq, then OpenRouter (Gemma 4).

        A provider answer that parses is accepted as-is (even
        "null holder") — the next provider is only consulted when the
        current one actually FAILS (timeout/network/bad JSON).  Returns
        (holder_id, confidence).
        """
        height, width = frame.shape[:2]
        if width > MAX_VLM_WIDTH:
            scale = MAX_VLM_WIDTH / width
            new_height = int(height * scale)
            vlm_frame = cv2.resize(frame, (MAX_VLM_WIDTH, new_height))
        else:
            vlm_frame = frame

        if self._gemini is not None:
            try:
                holder_id, confidence = self._ask_gemini_holder(vlm_frame)
                print(f"[Reporter] Mic holder via Gemini VLM: "
                      f"ID {holder_id} (conf {confidence:.2f})")
                return holder_id, confidence
            except Exception as exc:
                print(f"[Reporter] Gemini VLM failed ({exc}) — "
                      f"falling back to Groq")
        else:
            print("[Reporter] No GEMINI_API_KEY — using Groq VLM only.")

        if self._groq is not None:
            try:
                holder_id, confidence = self._ask_groq_holder(vlm_frame)
                print(f"[Reporter] Mic holder via Groq VLM: "
                      f"ID {holder_id} (conf {confidence:.2f})")
                return holder_id, confidence
            except Exception as exc:
                print(f"[Reporter] Groq VLM timed out/failed ({exc}) — "
                      f"falling back to OpenRouter")
        else:
            print("[Reporter] No GROQ_API_KEY — using OpenRouter VLM only.")

        try:
            return self._ask_openrouter_holder(vlm_frame)
        except Exception as exc:
            print(f"[Reporter] OpenRouter VLM timed out/failed ({exc}) — "
                  f"all providers failed this round")
            return None, 0.0

    def _ask_groq_holder(self, frame):
        """One Groq vision call with a hard per-request timeout.

        Raises on any failure (timeout, network, bad JSON) so the caller
        can fall back to Gemini.
        """
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not JPEG-encode frame for Groq")
        data_url = ("data:image/jpeg;base64,"
                    + base64.b64encode(buf.tobytes()).decode("utf-8"))

        response = self._groq.chat.completions.create(
            model=GROQ_VLM_MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You identify the microphone holder in an "
                            "annotated frame. Return ONLY valid JSON."},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": _HOLDER_PROMPT},
                     {"type": "image_url",
                      "image_url": {"url": data_url}},
                 ]},
            ],
            temperature=0,
            # Groq free tier caps qwen3.8-27b at 1000 output tokens per
            # minute; the reply is a tiny JSON, so 256 is more than enough
            # and keeps every request safely under the limit (fixes 429s).
            max_tokens=256,
            timeout=GROQ_VLM_TIMEOUT_SECONDS,
            # qwen3.8 defaults to slow "thinking" mode on Groq — disabled
            # because this is a short structured JSON task.
            reasoning_effort="none",
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        data = json.loads(raw)  # raises -> caller falls back to Gemini
        holder_id = data.get("microphone_holder_id")
        confidence = float(data.get("confidence", 0.0))
        return holder_id, confidence

    def _gemini_config(self, model):
        """Request config with thinking turned down for a short JSON task.

        Gemini 3.x "thinks" by default, which pushes an image request past
        the HTTP budget; it takes thinking_level. Gemini 2.5 models use a
        token budget instead (0 = off). AFC is off because no tools are
        passed (also silences its warning).
        """
        types = self._genai_types
        if model.startswith("gemini-2.5"):
            thinking = types.ThinkingConfig(thinking_budget=0)
        else:
            thinking = types.ThinkingConfig(thinking_level="MINIMAL")
        return types.GenerateContentConfig(
            thinking_config=thinking,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

    def _ask_gemini_holder(self, frame):
        """Gemini vision call — the original video_tracking.py path.

        Tries each model in GEMINI_MODELS until one answers. Raises if all
        fail (network, timeout, overload, bad JSON) so the caller falls
        back to the next provider in the chain.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = self._Image.fromarray(rgb)

        errors = []
        response = None
        for model in GEMINI_MODELS:
            try:
                response = self._gemini.models.generate_content(
                    model=model,
                    contents=[pil_image, _HOLDER_PROMPT],
                    config=self._gemini_config(model),
                )
                break
            except Exception as exc:
                errors.append(f"{model}: {str(exc)[:80]}")
                print(f"[Reporter] Gemini {model} failed — trying next model")
        if response is None:
            raise RuntimeError("all Gemini models failed: " + " | ".join(errors))

        text = response.text.strip()
        print()
        print(f"[Reporter] Gemini response ({model}):")
        print(text)

        text = re.sub(r"```json|```", "", text,
                      flags=re.IGNORECASE).strip()

        try:
            data = json.loads(text)
            holder_id = data.get("microphone_holder_id")
            confidence = float(data.get("confidence", 0.0))
            return holder_id, confidence
        except Exception as exc:
            raise RuntimeError(f"Could not parse Gemini JSON: {exc}") from exc

    def _ask_openrouter_holder(self, frame):
        """One OpenRouter (Gemma 4) vision call — OpenAI-compatible API.

        Raises on any failure (timeout, network, bad JSON) so the caller
        can report "no holder identified" for this round.
        """
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not JPEG-encode frame for OpenRouter")
        data_url = ("data:image/jpeg;base64,"
                    + base64.b64encode(buf.tobytes()).decode("utf-8"))

        response = self._openrouter.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You identify the microphone holder in an "
                            "annotated frame. Return ONLY valid JSON."},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": _HOLDER_PROMPT},
                     {"type": "image_url",
                      "image_url": {"url": data_url}},
                 ]},
            ],
            temperature=0,
            max_tokens=OPENROUTER_MAX_TOKENS,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        data = json.loads(raw)  # raises -> caller reports no holder
        holder_id = data.get("microphone_holder_id")
        confidence = float(data.get("confidence", 0.0))
        return holder_id, confidence
