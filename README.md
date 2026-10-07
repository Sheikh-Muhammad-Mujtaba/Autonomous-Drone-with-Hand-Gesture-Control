# Autonomous Journalism Drone with Hand Gesture Control and Person Tracking

**An AI-powered journalism drone enabling hands-free operation without
wearable sensors or handheld controllers.** Using only an onboard camera,
it recognizes natural hand gestures, accurately identifies and tracks the
designated reporter, detects the reporter's microphone for optimal framing,
and integrates LLMs/VLMs for intelligent voice commands — delivering
autonomous, stable, and high-quality news coverage in dynamic reporting
environments.

## The problem

Live journalism often needs a second person — a camera operator or drone
pilot — to capture dynamic shots, which isn't feasible for solo reporters
or small teams. We built an autonomous interview drone that turns a
low-cost DJI Tello into a hands-free camera operator, controllable without
any wearable sensors or handheld controllers.

Using only its onboard camera, the drone identifies the reporter by
detecting who is holding a microphone, then tracks and obeys only that
person via hand gestures or natural voice commands, ignoring everyone else
in frame. Safety commands (land, hover, emergency stop) remain accessible
to any operator at all times.

Built with Python, OpenCV, and MediaPipe for real-time gesture recognition
at 30 Hz, a full voice pipeline (VAD → faster-whisper → LLM command
parser), and a "reporter guard" system combining YOLO + ByteTrack with a
vision-language model (Groq/Gemini) to lock onto the correct person.
Includes a dead-reckoning position tracker and hardened emergency-landing
logic.

This is a functional prototype validated through real flight tests on DJI
Tello hardware — not yet hardened for outdoor/production deployment. It's
built for journalists, content creators, and event broadcasters needing
accessible aerial coverage without a dedicated pilot.

## What it does today

- **Reporter guard** — YOLOv8 + ByteTrack track every person in frame; a
  vision LLM identifies the microphone holder and locks gesture/voice
  control to that person (press `o` to toggle the guard off for demos).
- **Hand-gesture control** — ten A–J hand signs (stop, up, down, left,
  right, flip, turn left, turn right, backward, forward). A MediaPipe
  pose gate runs the gesture engine only when a hand is raised.
- **Voice commands** — microphone → VAD → faster-whisper (offline) →
  regex/LLM parser: "take off", "move forward 50", "describe the scene"
  (the drone speaks the description back via TTS).
- **Face-tracking mode** — PID follow-the-face with distance estimation.
- **Dead-reckoning position map** — live 2D map of estimated position and
  yaw, plus 3-phase return-home.
- **Safety** — battery guard, emergency land, and a guaranteed clean
  shutdown (land + stream off) even if an error occurs mid-flight.

## Project layout

| Path | Purpose |
|---|---|
| `main.py` | Entry point — wiring + 30 Hz flight loop |
| `config.py` | Paths and tunables (project-root-relative) |
| `core/` | Video reader, worker threads, face-tracking PID |
| `perception/` | Gesture classifier, pose gate, reporter tracker, face detector |
| `control/` | Command dispatcher, position tracker, position map, voice pipeline |
| `ui/` | Unified tabbed window (tkinter), pygame key module |
| `scripts/` | Calibration tools + offline video tracking test |
| `test_script/` | No-drone tests: model verification + full webcam demo |
| `model/` | All model weights (gesture `.h5`, `yolov8s.pt`, cascade, whisper) |
| `custom_hand_gesture_model_training/` | Data collection, training notebook, and exported models |
| `assets/` | Architecture diagrams |

## Architecture

### System architecture
![System Architecture](assets/drone_architecture_simple.png)

### Training pipeline
![Training Pipeline](assets/training_pipeline_simple.png)

## Setup on any PC

1. **Install Python 3.10** (64-bit) from python.org — tick "Add to PATH".
   The project is developed and tested on 3.10.
2. **Copy the project folder** anywhere on the machine.
3. **Install dependencies** — open a terminal in the project folder:
   ```powershell
   py -3.10 -m pip install -r requirements.txt
   ```
   (If Python 3.10 is your default interpreter, plain `python -m pip ...`
   works too.)
4. **(Optional) API keys** — create a `.env` file in the project folder:
   ```ini
   GROQ_API_KEY=...   # free key at https://console.groq.com/keys
   GEMINI_API_KEY=... # https://aistudio.google.com/apikey
   OPEN_ROUTER_API_GAMMA_MODEL=... # https://openrouter.ai/settings/keys
   ```
   Without keys the drone still flies: the reporter guard disables itself
   and voice falls back to the built-in regex command parser.
5. **Model weights** — everything in `model/` ships with the project. The
   faster-whisper weights download automatically into
   `model/faster-whisper-medium/` on the first voice run (one-time, needs
   internet; fully offline afterwards).

## Run

From the project folder:

```powershell
py -3.10 main.py                        # fly the drone (full program)
py -3.10 test_script\test_models.py     # verify all models — no drone needed
py -3.10 test_script\demo_webcam.py     # full pipeline on your webcam — no drone
py -3.10 -m scripts.calibrate_and_map   # measure RC_TO_CM_S / RC_TO_DEG_S
py -3.10 -m scripts.caliberation        # measure face focal length
py -3.10 scripts\video_tracking.py      # offline reporter-ID test on a video
```

## Controls (in flight)

| Key | Action |
|---|---|
| Arrows | Move |
| w / s | Up / Down |
| a / d | Yaw |
| e | Takeoff |
| q | Land + quit |
| x | Emergency land |
| r | Return home |
| o | Toggle reporter guard (ON: reporter only, OFF: anyone) |

Gestures: raise a hand to enable; A–J hand signs map to stop / up / down /
left / right / flip / turn left / turn right / move backward / move forward.

Voice: speak naturally — "take off", "move forward 50", "describe the scene".

## Testing without a drone

`test_script\test_models.py` loads every production model exactly the way
the flight program loads them — same modules, same paths, same parameters —
and prints only the results. Nothing talks to a drone, a GUI, or the
microphone:

```powershell
py -3.10 test_script\test_models.py                          # all checks
py -3.10 test_script\test_models.py --skip-voice             # skip whisper load
py -3.10 test_script\test_models.py --video path\to\video.mp4  # + live YOLO demo on any video
```

Each check reports load status, load time, and one sample inference:
gesture engine, pose gate, YOLO person tracking, face cascade, and the
voice (whisper) pipeline.

### Full webcam demo (no drone)

`test_script\demo_webcam.py` runs the **entire production pipeline** with
your PC camera as the video source and a fake drone that records every
RC command instead of flying. Same workers, same debounce/latch/merge/
dispatch chain, same tabbed screen (Drone View / Hand Crop / Position
Map) as the real flight program — including the 3-phase return-home
(with simulated sensors) and face-tracking mode:

```powershell
py -3.10 test_script\demo_webcam.py              # webcam index 0
py -3.10 test_script\demo_webcam.py --camera 1   # another camera
py -3.10 test_script\demo_webcam.py --with-voice # + mic/whisper voice worker
```

Controls: press `e` to (fake) take off, arrows / `w` `s` / `a` `d` to
move, `r` to return home, `o` to toggle the reporter guard, `q` to land
and quit. The reporter guard needs API keys; without them it
auto-disables and the rest keeps running.
