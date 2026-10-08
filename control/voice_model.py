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
# "tiny"/"base"/"small" for speed, "medium" for accuracy (~3x slower on CPU).
# Override without editing code: set WHISPER_MODEL_SIZE=medium in .env.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
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
VAD_PREROLL_MS = 300             # audio kept from before the trigger (word onsets)
VAD_MIN_SPEECH_MS_TO_START = 150 # voiced audio inside the pre-roll that starts recording
VAD_SILENCE_MS_TO_STOP = 600     # stop recording after this much silence
VAD_MIN_UTTERANCE_SPEECH_MS = 200  # drop clicks/coughs shorter than this
VAD_MAX_UTTERANCE_SECONDS = 8    # hard cap so a stuck-open mic can't hang forever

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


def _never_stop():
    return False


def listen_for_utterance_vad(person_present_fn=_default_person_present,
                              sample_rate=SAMPLE_RATE,
                              should_stop_fn=_never_stop):
    """
    Continuously streams mic audio and uses webrtcvad to detect natural
    speech start/stop, like ChatGPT's voice mode — no wake word, no fixed
    window. Only actively listens while person_present_fn() returns True.

    Recording starts once VAD_MIN_SPEECH_MS_TO_START of voiced audio is
    seen inside the last VAD_PREROLL_MS (so one-word commands such as
    "land" trigger), and stops after VAD_SILENCE_MS_TO_STOP of silence.

    Returns a float32 numpy array of the captured utterance, or None when
    nothing usable was heard or should_stop_fn() returned True.
    """
    frame_len = int(sample_rate * VAD_FRAME_MS / 1000)  # samples per frame
    preroll = collections.deque(maxlen=max(1, VAD_PREROLL_MS // VAD_FRAME_MS))
    start_frames = max(1, VAD_MIN_SPEECH_MS_TO_START // VAD_FRAME_MS)
    max_frames = int(VAD_MAX_UTTERANCE_SECONDS * 1000 / VAD_FRAME_MS)

    frames_collected = []
    triggered = False
    silence_ms = 0
    voiced_ms = 0

    audio_q = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    print("\n[vad] waiting for a person + speech...")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                         blocksize=frame_len, callback=_callback):
        while not should_stop_fn():
            if not person_present_fn():
                # Drain queue so it doesn't build up while no one's there.
                while not audio_q.empty():
                    audio_q.get()
                time.sleep(0.1)
                continue

            try:
                frame = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            is_speech = _vad.is_speech(frame.tobytes(), sample_rate)

            if not triggered:
                preroll.append((frame, is_speech))
                if sum(1 for _, voiced in preroll if voiced) >= start_frames:
                    triggered = True
                    print("[vad] speech detected, recording...")
                    frames_collected = [f for f, _ in preroll]
                    voiced_ms = sum(VAD_FRAME_MS for _, v in preroll if v)
                    preroll.clear()
                continue

            frames_collected.append(frame)
            if is_speech:
                silence_ms = 0
                voiced_ms += VAD_FRAME_MS
            else:
                silence_ms += VAD_FRAME_MS

            if silence_ms >= VAD_SILENCE_MS_TO_STOP:
                print("[vad] silence detected, stopping.")
                break
            # Counted from the trigger only: idle time before speaking must
            # not eat into the utterance budget.
            if len(frames_collected) >= max_frames:
                print("[vad] max utterance length reached, stopping.")
                break

    if not triggered or voiced_ms < VAD_MIN_UTTERANCE_SPEECH_MS:
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


_whisper_model = None
_whisper_lock = threading.Lock()


def load_whisper():
    """Load (once) and return the Whisper model.

    Lazy so importing this module stays cheap; VoiceWorker calls it at the
    start of its thread, so the multi-second load happens in the background
    instead of delaying drone start-up.
    """
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            print(f"[whisper] loading '{WHISPER_MODEL_SIZE}' model on {WHISPER_DEVICE}...")
            model_path = _ensure_local_whisper_model(WHISPER_MODEL_DIR, WHISPER_REPO_ID)
            _whisper_model = WhisperModel(
                model_path,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            print(f"[whisper] model ready. (loaded from {model_path})")
    return _whisper_model


# No initial_prompt on purpose: a vocabulary prompt ("take off, land,
# hover, ...") was read back verbatim on propeller noise in flight tests,
# turning noise into a real takeoff/land command.

# Segments Whisper itself rates as probably-not-speech are dropped.
WHISPER_NO_SPEECH_MAX = 0.6

# Phrases Whisper invents from silence/noise. A transcript made only of
# these is dropped instead of being sent to the parser.
_WHISPER_HALLUCINATIONS = frozenset({
    "", "you", "thank you", "thanks", "thank you very much",
    "thanks for watching", "thank you for watching", "bye", "okay", "ok",
    "subtitles by the amara.org community",
})


def transcribe(audio_array, sample_rate=SAMPLE_RATE):
    """Runs local Whisper on a numpy float32 array. Returns lowercase text.

    Tuned for 1-3 second commands that webrtcvad has already trimmed:
    greedy decoding (beam_size=1) is ~2x faster than the default beam of 5
    on CPU with no loss on short phrases, and Silero VAD is off because it
    can clip one-word commands such as "land".
    """
    segments, _info = load_whisper().transcribe(
        audio_array,
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    text = " ".join(
        seg.text.strip() for seg in segments if seg.no_speech_prob <= WHISPER_NO_SPEECH_MAX
    ).strip().lower()
    if re.sub(r"[^a-z' ]", "", text).strip() in _WHISPER_HALLUCINATIONS:
        return ""
    return text


# --------------------------------------------------------------------------
# 3. REGEX / KEYWORD PARSER  (must cover 100% of rehearsed demo phrases)
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_NUMBER_RE = r"(\d+(?:\.\d+)?|half|" + "|".join(NUMBER_WORDS) + r")"
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ONES = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_COMPOUND_RE = re.compile(rf"\b({'|'.join(_TENS)})[ -]({'|'.join(_ONES)})\b")


def _join_compound_numbers(text):
    """ "forty five" -> "45", so it is not read as 40."""
    return _COMPOUND_RE.sub(
        lambda m: str(NUMBER_WORDS[m.group(1)] + NUMBER_WORDS[m.group(2)]), text
    )

# Safety bounds for spoken distances/angles (a misheard "500" must not
# send the drone across the room).
MIN_MOVE_DISTANCE_CM, MAX_MOVE_DISTANCE_CM = 10, 300
MIN_ROTATE_DEGREES, MAX_ROTATE_DEGREES = 5, 360

# Filler words Whisper keeps that would otherwise match a direction
# ("all right" -> move right) or add noise.
_FILLER_RE = re.compile(
    r"\b(all right|alright|right now|okay|ok|please|hey|drone|um+|uh+)\b"
)


def _parse_number(token):
    if token == "half":
        return 0.5
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])
    return float(token)


def _extract_number(text, default):
    """First number in `text` as a word or digits, whole words only.

    The old substring check matched "one" inside "someone" and "ten"
    inside "listen".
    """
    for match in re.finditer(rf"\b{_NUMBER_RE}\b", text):
        if match.group(1) not in ("a", "an"):  # "a"/"an" only count before a unit
            return _parse_number(match.group(1))
    return default


def _extract_distance_cm(text, default=DEFAULT_MOVE_DISTANCE_CM):
    """Distance in cm, honouring metres: "two meters" -> 200, "50" -> 50."""
    unit_re = rf"\b{_NUMBER_RE}\s*(?:a\s+)?(m|meters?|metres?|cm|centimet(?:er|re)s?)\b"
    match = re.search(unit_re, text)
    if match:
        value = _parse_number(match.group(1))
        if match.group(2).startswith("m"):
            value *= 100
    else:
        value = _extract_number(text, default)
    return int(min(max(value, MIN_MOVE_DISTANCE_CM), MAX_MOVE_DISTANCE_CM))


def _extract_degrees(text, default=DEFAULT_ROTATE_DEGREES):
    if re.search(r"\baround\b|\bhalf (?:a )?turn\b", text):
        return 180
    value = _extract_number(text, default)
    return int(min(max(value, MIN_ROTATE_DEGREES), MAX_ROTATE_DEGREES))


def regex_parser(text):
    """
    Fast, offline, deterministic. Returns a command dict or None.
    Extend this list with every phrase you plan to actually say live.
    """
    t = _FILLER_RE.sub(" ", text.lower())
    t = _join_compound_numbers(re.sub(r"[^a-z0-9.' -]", " ", t)).replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()

    if not t:
        return None

    wants_takeoff = re.search(r"\btake ?off\b|\blift ?off\b|\bstart flying\b|\blaunch\b", t)
    wants_land = re.search(r"\bland\b|\btouch ?down\b|\bcome down\b", t) and "return" not in t
    if wants_takeoff and wants_land:
        # Contradictory (e.g. Whisper echoing a list of commands): hold still.
        return {"action": "hover"}
    if wants_takeoff:
        return {"action": "takeoff"}
    if wants_land:
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
        degrees = _extract_degrees(t)
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
            distance = _extract_distance_cm(t)
            return {"action": "move", "direction": direction, "distance_cm": distance}

    return None  # nothing matched -> fall through to LLM


# --------------------------------------------------------------------------
# 4. LLM FALLBACK PARSER (Groq, OpenAI-compatible API, async with hard timeout)
# --------------------------------------------------------------------------
_groq_client = None
if GROQ_API_KEY:
    # Client-side timeout + no retries: the HTTP call itself gives up on
    # time, so a slow request can never pile up behind the next utterance.
    _groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        timeout=max(LLM_TIMEOUT_SECONDS, VLM_TIMEOUT_SECONDS),
        max_retries=0,
    )
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
        # gpt-oss is a reasoning model; "low" keeps a one-line intent
        # extraction well inside LLM_TIMEOUT_SECONDS.
        reasoning_effort="low",
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return validate_intent(json.loads(raw))


_SIMPLE_ACTIONS = frozenset({
    "takeoff", "land", "return_home", "hover", "follow", "describe_scene",
})
_MOVE_DIRECTIONS = frozenset({"forward", "back", "left", "right", "up", "down"})
_ROTATE_DIRECTIONS = frozenset({"cw", "ccw"})


def _bounded_int(value, default, low, high):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return int(min(max(number, low), high))


def validate_intent(data):
    """Return a clean intent dict, or None if `data` is not a valid command.

    LLM output is external input: only whitelisted actions/directions pass,
    and distances/angles are clamped to the same bounds as the regex path.
    """
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if action in _SIMPLE_ACTIONS:
        return {"action": action}
    direction = data.get("direction")
    if action == "move" and direction in _MOVE_DIRECTIONS:
        distance = _bounded_int(data.get("distance_cm"), DEFAULT_MOVE_DISTANCE_CM,
                                MIN_MOVE_DISTANCE_CM, MAX_MOVE_DISTANCE_CM)
        return {"action": "move", "direction": direction, "distance_cm": distance}
    if action == "rotate" and direction in _ROTATE_DIRECTIONS:
        degrees = _bounded_int(data.get("degrees"), DEFAULT_ROTATE_DEGREES,
                               MIN_ROTATE_DEGREES, MAX_ROTATE_DEGREES)
        return {"action": "rotate", "direction": direction, "degrees": degrees}
    return None


# One long-lived pool. A `with ThreadPoolExecutor()` block waits for its
# worker on exit, so the old per-call pool silently turned every "timeout"
# into "wait for the full HTTP request anyway".
_api_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="voice-api"
)


def _run_with_timeout(fn, arg, timeout, tag):
    """Run fn(arg) on the shared pool; None on timeout/any error. Never raises."""
    try:
        future = _api_executor.submit(fn, arg)
    except RuntimeError as e:
        # Interpreter is shutting down (user quit mid-utterance).
        print(f"[{tag}] skipped: {e}")
        return None
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"[{tag}] timed out after {timeout}s")
    except Exception as e:
        print(f"[{tag}] failed: {e}")
    return None


def llm_parser(text, timeout=LLM_TIMEOUT_SECONDS):
    """
    Runs the Groq call in a background thread with a hard timeout.
    Never raises — returns None on any failure (timeout, network, bad JSON).
    """
    if _groq_client is None:
        return None
    return _run_with_timeout(_call_groq, text, timeout, "llm")


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

    return _run_with_timeout(_call_vlm, image_data_url, timeout, "vlm")


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
