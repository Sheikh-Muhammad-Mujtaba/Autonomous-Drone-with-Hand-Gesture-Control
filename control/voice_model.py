"""
Standalone voice command pipeline (Steps 3 + 3b of the plan).

Flow:
    mic -> faster-whisper (local, offline) -> regex_parser -> execute
                                                  |
                                                  v (if no match)
                                            llm_parser (Groq, async, timeout)
                                                  |
                                                  v (if that also fails)
                                            safe_fallback()  -> hover

This script has ZERO drone dependency. It just prints the resulting
command dict + which path fired (regex / llm / safe_fallback), so you
can validate STT quality and regex coverage before wiring anything
into djitellopy.

Setup:
    pip install faster-whisper sounddevice numpy groq openai python-dotenv huggingface_hub opencv-python pyttsx3
    Create a .env file next to this script with:
        GROQ_API_KEY=your_key_here
    (get a free key, no credit card, at https://console.groq.com/keys)

Model storage:
    Whisper weights are downloaded ONCE into ./model/faster-whisper-<size>/
    inside this project folder (not the global HF cache under your user
    profile). After that first run it's fully offline.
"""

import os
import re
import json
import time
import queue
import base64
import threading
import collections
import concurrent.futures

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

import config

load_dotenv()

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000
RECORD_SECONDS = 4          # length of each listen window
WHISPER_MODEL_SIZE = "small"  # "tiny"/"small" for speed, "medium" for better accuracy
WHISPER_DEVICE = "cpu"        # change to "cuda" if you have a GPU set up for it
WHISPER_COMPUTE_TYPE = "int8"  # good CPU speed/accuracy tradeoff

# Store the Whisper weights inside the project's own "model" folder instead
# of the default global HF cache (C:\Users\<you>\.cache\huggingface\...).
# Keeps the repo self-contained / easy to zip for submission.
PROJECT_ROOT = str(config.PROJECT_ROOT)
WHISPER_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "model", f"faster-whisper-{WHISPER_MODEL_SIZE}"
)
WHISPER_REPO_ID = f"Systran/faster-whisper-{WHISPER_MODEL_SIZE}"

LLM_TIMEOUT_SECONDS = 2.5
# Groq: free tier, OpenAI-compatible API, no credit card.
# llama-3.1-8b-instant was deprecated by Groq — openai/gpt-oss-20b is their
# recommended replacement for fast, low-latency intent-style tasks.
GROQ_MODEL_NAME = "openai/gpt-oss-20b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MOVE_DISTANCE_CM = 50
DEFAULT_ROTATE_DEGREES = 90

# --------------------------------------------------------------------------
# VLM (Step 6b) + TTS config
# --------------------------------------------------------------------------
VLM_TIMEOUT_SECONDS = 4.0
# Groq's vision-capable free model as of testing. Their multimodal lineup
# changes fairly often — if this 404s, check console.groq.com/docs/models.
GROQ_VLM_MODEL_NAME = "qwen/qwen3.8-27b"
VLM_SYSTEM_PROMPT = (
    "You are a scene-description module for a journalism drone. "
    "In 1-2 short sentences, describe what's visible: is a person/reporter "
    "centered in frame, is a microphone visible, is the framing/lighting "
    "good for filming? Be concise and factual, no filler."
)
WEBCAM_INDEX = 0  # change if your drone/webcam isn't device 0
TTS_ENABLED = True  # set False to skip speaking the description aloud

# --------------------------------------------------------------------------
# Continuous listening (VAD) config — "ChatGPT voice mode" style, no wake
# word needed. Person-presence gating hooks into your existing MediaPipe
# pose module so the drone only listens when someone is actually there.
# --------------------------------------------------------------------------
VAD_FRAME_MS = 30                # webrtcvad requires 10/20/30ms frames
VAD_AGGRESSIVENESS = 2           # 0 (permissive) to 3 (aggressive filtering)
VAD_SILENCE_MS_TO_STOP = 800     # stop recording after this much silence
VAD_MIN_SPEECH_MS_TO_START = 150 # ignore tiny blips/coughs
VAD_MAX_UTTERANCE_SECONDS = 12   # hard cap so a stuck-open mic can't hang forever

# --------------------------------------------------------------------------
# 1. AUDIO CAPTURE
# --------------------------------------------------------------------------
def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Blocking mic record. Returns a mono float32 numpy array in [-1, 1]."""
    print(f"\n[mic] listening for {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


# --------------------------------------------------------------------------
# 1b. CONTINUOUS VAD LISTENING (no wake word — auto start/stop on speech)
# --------------------------------------------------------------------------
_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


def _default_person_present():
    """
    Always-on fallback. Replace this by passing your own callable into
    listen_for_utterance_vad(person_present_fn=...), e.g. a bound method
    on your existing PoseHandGate that returns True when MediaPipe sees
    a person in frame. Keeping listening gated behind presence avoids
    burning Whisper/LLM cycles (and API quota) when no one's around.
    """
    return True


def listen_for_utterance_vad(person_present_fn=_default_person_present,
                              sample_rate=SAMPLE_RATE):
    """
    Continuously streams mic audio and uses webrtcvad to detect natural
    speech start/stop, like ChatGPT's voice mode — no wake word, no fixed
    window. Only actively listens while person_present_fn() returns True.

    Returns a float32 numpy array of the captured utterance, or None if
    person_present_fn() never returned True (caller should just loop again).
    """
    frame_len = int(sample_rate * VAD_FRAME_MS / 1000)  # samples per frame
    ring_buffer = collections.deque(
        maxlen=int(VAD_SILENCE_MS_TO_STOP / VAD_FRAME_MS)
    )

    frames_collected = []
    triggered = False
    speech_ms = 0
    max_frames = int(VAD_MAX_UTTERANCE_SECONDS * 1000 / VAD_FRAME_MS)
    frame_count = 0

    audio_q = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    print("\n[vad] waiting for a person + speech...")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                         blocksize=frame_len, callback=_callback):
        while True:
            if not person_present_fn():
                # Drain queue so it doesn't build up while no one's there.
                while not audio_q.empty():
                    audio_q.get()
                time.sleep(0.1)
                continue

            frame = audio_q.get()
            frame_bytes = frame.tobytes()
            frame_count += 1

            is_speech = _vad.is_speech(frame_bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = sum(1 for _, s in ring_buffer if s)
                if num_voiced > 0.6 * ring_buffer.maxlen:
                    triggered = True
                    print("[vad] speech detected, recording...")
                    frames_collected = [f for f, _ in ring_buffer]
                    ring_buffer.clear()
            else:
                frames_collected.append(frame)
                if is_speech:
                    speech_ms = 0
                else:
                    speech_ms += VAD_FRAME_MS

                if speech_ms >= VAD_SILENCE_MS_TO_STOP:
                    print("[vad] silence detected, stopping.")
                    break
                if frame_count >= max_frames:
                    print("[vad] max utterance length reached, stopping.")
                    break

    if not frames_collected:
        return None

    audio_int16 = np.concatenate(frames_collected, axis=0).flatten()
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


# --------------------------------------------------------------------------
# 2. LOCAL STT (Whisper)
# --------------------------------------------------------------------------
def _ensure_local_whisper_model(model_dir, repo_id):
    """
    Downloads the model into `model_dir` only if it isn't already there.
    After the first run, this is a no-op and everything loads from disk
    — zero network calls, satisfies the offline-STT requirement.
    """
    marker = os.path.join(model_dir, "model.bin")
    if os.path.exists(marker):
        return model_dir

    print(f"[whisper] model not found locally, downloading '{repo_id}' "
          f"into: {model_dir}")
    from huggingface_hub import snapshot_download
    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=model_dir)
    return model_dir


print(f"[whisper] loading '{WHISPER_MODEL_SIZE}' model on {WHISPER_DEVICE}...")
_local_model_path = _ensure_local_whisper_model(WHISPER_MODEL_DIR, WHISPER_REPO_ID)
_whisper_model = WhisperModel(
    _local_model_path,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print(f"[whisper] model ready. (loaded from {_local_model_path})")


def transcribe(audio_array, sample_rate=SAMPLE_RATE):
    """Runs local Whisper on a numpy float32 array. Returns lowercase text."""
    segments, _info = _whisper_model.transcribe(
        audio_array,
        language="en",
        vad_filter=True,  # trims silence, helps short mic clips
    )
    text = " ".join(seg.text.strip() for seg in segments).strip().lower()
    return text


# --------------------------------------------------------------------------
# 3. REGEX / KEYWORD PARSER  (must cover 100% of rehearsed demo phrases)
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_number(text, default):
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    for word, val in NUMBER_WORDS.items():
        if word in text:
            return val
    return default


def regex_parser(text):
    """
    Fast, offline, deterministic. Returns a command dict or None.
    Extend this list with every phrase you plan to actually say live.
    """
    t = text.lower().strip()

    if not t:
        return None

    if re.search(r"\btake ?off\b|\blift ?off\b|\bstart flying\b|\blaunch\b", t):
        return {"action": "takeoff"}

    if re.search(r"\bland\b|\btouch ?down\b|\bcome down\b", t) and "return" not in t:
        return {"action": "land"}

    if re.search(
        r"\breturn\b.*\bhome\b|\bcome back\b|\bgo home\b|\bcome home\b|"
        r"\bfly back\b|\bhead back\b|\bgo back\b",
        t,
    ):
        return {"action": "return_home"}

    if re.search(
        r"\bstop\b|\bhover\b|\bhold\b|\bfreeze\b|\bhalt\b|\bpause\b|\bstay\b|\bbrake\b",
        t,
    ):
        return {"action": "hover"}

    if re.search(
        r"\bfollow me\b|\block on\b|\btrack me\b|\bchase me\b|\bcome with me\b|\bpursue\b",
        t,
    ):
        return {"action": "follow"}

    if re.search(
        r"\bdescribe\b.*\bscene\b|\bwhat.*see\b|\bdescribe\b|\blook around\b|"
        r"\bwhat's around\b|\bscene description\b",
        t,
    ):
        return {"action": "describe_scene"}

    # Rotate BEFORE move directions: "turn right" must rotate, not move.
    if re.search(r"\bturn\b|\brotate\b|\bspin\b|\byaw\b", t):
        degrees = _extract_number(t, DEFAULT_ROTATE_DEGREES)
        if re.search(
            r"\bleft\b|\bcounter ?clockwise\b|\bccw\b|\banti ?clockwise\b",
            t,
        ):
            return {"action": "rotate", "direction": "ccw", "degrees": degrees}
        return {"action": "rotate", "direction": "cw", "degrees": degrees}

    direction_map = {
        # forward
        "forward": "forward", "ahead": "forward", "go forward": "forward",
        "move forward": "forward", "onwards": "forward", "straight": "forward",
        "come": "forward", "come here": "forward",
        # back
        "back": "back", "backward": "back", "backwards": "back",
        "reverse": "back", "back up": "back", "retreat": "back",
        # left
        "left": "left", "go left": "left", "move left": "left",
        "to the left": "left",
        # right
        "right": "right", "go right": "right", "move right": "right",
        "to the right": "right",
        # up
        "up": "up", "ascend": "up", "climb": "up", "rise": "up",
        "go up": "up", "move up": "up", "higher": "up",
        # down
        "down": "down", "descend": "down", "drop": "down", "sink": "down",
        "go down": "down", "move down": "down", "lower": "down",
    }
    for keyword, direction in direction_map.items():
        if re.search(rf"\b{keyword}\b", t):
            distance = _extract_number(t, DEFAULT_MOVE_DISTANCE_CM)
            return {"action": "move", "direction": direction, "distance_cm": distance}

    return None  # nothing matched -> fall through to LLM


# --------------------------------------------------------------------------
# 4. LLM FALLBACK PARSER (Groq, OpenAI-compatible API, async with hard timeout)
# --------------------------------------------------------------------------
_groq_client = None
if GROQ_API_KEY:
    _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
else:
    print("[warn] GROQ_API_KEY not set — LLM fallback will be skipped.")

LLM_SYSTEM_PROMPT = """You convert a spoken drone command into STRICT JSON.
Return ONLY a JSON object, no markdown, no explanation, no code fences.

Valid actions and required fields:
- {"action": "takeoff"}
- {"action": "land"}
- {"action": "return_home"}
- {"action": "hover"}
- {"action": "follow"}
- {"action": "describe_scene"}
- {"action": "move", "direction": "forward"|"back"|"left"|"right"|"up"|"down", "distance_cm": <int>}
- {"action": "rotate", "direction": "cw"|"ccw", "degrees": <int>}

If the phrase does not clearly map to one of these, return: {"action": null}
"""


def _call_groq(text):
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f'Phrase: "{text}"'},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if not data or data.get("action") is None:
        return None
    return data


def llm_parser(text, timeout=LLM_TIMEOUT_SECONDS):
    """
    Runs the Groq call in a background thread with a hard timeout.
    Never raises — returns None on any failure (timeout, network, bad JSON).
    This is the function you'd later call from a background task so it
    never blocks the flight control loop.
    """
    if _groq_client is None:
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(_call_groq, text)
        except RuntimeError as e:
            # Interpreter is shutting down (user quit mid-utterance).
            print(f"[llm] skipped: {e}")
            return None
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[llm] timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"[llm] failed: {e}")
            return None


# --------------------------------------------------------------------------
# 5. SAFE FALLBACK
# --------------------------------------------------------------------------
def safe_fallback():
    return {"action": "hover"}


# --------------------------------------------------------------------------
# 5b. VLM SCENE DESCRIPTION (Step 6b) — same async/timeout pattern as LLM
# --------------------------------------------------------------------------
def capture_webcam_snapshot(save_path="snapshot.jpg", cam_index=WEBCAM_INDEX):
    """Grabs one frame from the webcam. Requires opencv-python."""
    import cv2
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture a frame from the webcam")
    cv2.imwrite(save_path, frame)
    return save_path


def _load_image_as_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _call_vlm(image_data_url):
    response = _groq_client.chat.completions.create(
        model=GROQ_VLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this scene."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.2,
        reasoning_effort="none",  # qwen3.8 defaults to "thinking" mode which
                                   # emits a slow <think>...</think> block —
                                   # not needed for a short scene description.
    )
    raw = response.choices[0].message.content.strip()
    # Belt-and-suspenders: strip any leftover <think> block just in case.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def describe_scene(image_path, timeout=VLM_TIMEOUT_SECONDS):
    """
    Runs the VLM call in a background thread with a hard timeout.
    Never raises — returns None on any failure. On failure the caller
    should just skip the response, never let it cascade into anything
    flight-critical.
    """
    if _groq_client is None:
        print("[vlm] no GROQ_API_KEY set, skipping.")
        return None

    try:
        image_data_url = _load_image_as_data_url(image_path)
    except Exception as e:
        print(f"[vlm] failed to load image: {e}")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_vlm, image_data_url)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[vlm] timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"[vlm] failed: {e}")
            return None


# --------------------------------------------------------------------------
# 5c. TTS — speak the scene description aloud (offline, no API)
# --------------------------------------------------------------------------
_tts_engine = None


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        import pyttsx3
        _tts_engine = pyttsx3.init()
    return _tts_engine


def speak(text):
    """Speaks text aloud via local TTS. Never raises — logs and skips on failure."""
    if not TTS_ENABLED or not text:
        return
    try:
        engine = _get_tts_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[tts] failed: {e}")


# --------------------------------------------------------------------------
# 6. MAIN LOOP (this is what you replace with the real dispatcher later)
# --------------------------------------------------------------------------
def resolve_command(transcript):
    """Runs the full regex -> llm -> safe_fallback chain. Returns (command, path)."""
    command = regex_parser(transcript)
    if command is not None:
        return command, "regex"

    command = llm_parser(transcript)
    if command is not None:
        return command, "llm"

    return safe_fallback(), "safe_fallback"


def handle_transcript(transcript):
    """Shared logic: resolve + execute + handle describe_scene. Used by
    both the press-Enter loop and the continuous VAD loop."""
    if not transcript:
        print("[result] no speech detected, skipping.")
        return

    command, path = resolve_command(transcript)
    print(f"[result] path={path} -> {json.dumps(command)}")

    if command.get("action") == "describe_scene":
        try:
            print("[cam] capturing snapshot...")
            image_path = capture_webcam_snapshot()
        except Exception as e:
            print(f"[cam] failed: {e}")
            return

        print("[vlm] sending to Groq vision model...")
        description = describe_scene(image_path)

        if description is None:
            print("[result] VLM call failed or timed out.")
        else:
            print(f"[result] scene description: \"{description}\"")
            speak(description)


def main():
    """Original press-Enter flow — untouched."""
    print("=== Voice Pipeline Standalone Test (press-Enter mode) ===")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            input("Press Enter to record a command...")
            audio = record_audio()
            transcript = transcribe(audio)
            print(f"[stt] transcript: \"{transcript}\"")
            handle_transcript(transcript)
    except KeyboardInterrupt:
        print("\nStopped.")


def main_continuous(person_present_fn=_default_person_present):
    """
    New: continuous listening, no wake word, no Enter key — like ChatGPT
    voice mode. Auto-detects speech start/stop via webrtcvad. Optionally
    gated by person_present_fn (wire this to your MediaPipe pose module
    so it only listens when someone is actually in frame).
    """
    print("=== Voice Pipeline — Continuous Listening (VAD mode) ===")
    print("Just start talking. Press Ctrl+C to stop.\n")
    try:
        while True:
            audio = listen_for_utterance_vad(person_present_fn=person_present_fn)
            if audio is None:
                continue
            transcript = transcribe(audio)
            print(f"[stt] transcript: \"{transcript}\"")
            handle_transcript(transcript)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    # Continuous listening, no wake word, no Enter key — like ChatGPT voice mode.
    # To gate it with your existing pose detector instead of always-on:
    #   from body_pose import PoseHandGate
    #   pose_gate = PoseHandGate(...)
    #   main_continuous(person_present_fn=pose_gate.is_present)
    main_continuous()
