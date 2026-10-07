# Setup on a New Machine (Windows)

Step-by-step guide to get the Autonomous Journalism Drone running on a fresh
Windows 10/11 PC: install, verify with the webcam (no drone), then fly the
real DJI Tello.

All commands are for **PowerShell**, run from the project root unless noted.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11, 64-bit | Tested on Windows 10 Pro 22H2 |
| Git | <https://git-scm.com/download/win> |
| GitHub CLI (`gh`) | Needed to clone the private repo: <https://cli.github.com> |
| `uv` | Python installer and package manager (step 2) |
| ~10 GB free disk | ~4.5 GB for packages, up to 1.5 GB per Whisper model, plus uv cache |
| Webcam + microphone | For the no-drone demo and voice commands |
| DJI Tello | Only for real flight (section 8) |

> **Keep the project path short.** Windows has a 260-character path limit and
> some packages (sounddevice, ultralytics) have deep file paths. A path like
> `D:\Projects\journalism-drone` is safe; a long nested path can fail with
> *"The filename or extension is too long"*. Alternatively enable long paths
> (admin PowerShell):
> `New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`

---

## 2. Install uv and Python 3.10

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Close and reopen PowerShell so `uv` is on PATH, then:
uv python install 3.10
```

Python **3.10** is required: the pinned TensorFlow 2.17 / MediaPipe 0.10.14
combination is what the project is tested against.

---

## 3. Get the code

```powershell
gh auth login                       # log in with an account that has access
gh repo clone mujios/journalism-drone D:\Projects\journalism-drone
cd D:\Projects\journalism-drone
```

---

## 4. Create the virtual environment and install packages

```powershell
# If the project and the uv cache are on different drives (e.g. D: and C:),
# hard-linking fails, so tell uv to copy files instead.
$env:UV_LINK_MODE = "copy"

# Check free space first (need ~10 GB across C: and the project drive)
Get-PSDrive C, D | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}

uv venv --python 3.10 .venv
.\.venv\Scripts\Activate.ps1
uv pip sync requirements-lock.txt
```

`requirements-lock.txt` pins every package to a version set that is verified
to pass all tests. Use `uv pip sync` (not `uv pip install -r`) — `sync`
installs exactly the listed packages, which matters because of the OpenCV
issue below.

If `Activate.ps1` is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Alternative: plain `python -m venv` + pip (no uv)

Use Python **3.10** and install the lock file with `--no-deps`. That flag is
pip's equivalent of `uv pip sync`: it installs exactly the 125 tested
packages and skips pulling in `opencv-python`.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip        # old pip (21.x) resolves badly
python -m pip install --no-deps -r requirements-lock.txt
```

If you install `requirements.txt` instead (without `--no-deps`), pip resolves
fine but adds `opencv-python`; fix it with the OpenCV-trap steps below,
using `pip` in place of `uv pip`
(`pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84`).

### Why these pins (don't upgrade them casually)

| Package | Pinned | Reason |
|---|---|---|
| `numpy` | 1.26.4 | TensorFlow 2.17 / MediaPipe break on NumPy 2 |
| `opencv-contrib-python` | 4.10.0.84 | Must be the **only** OpenCV package (see below) |
| `mediapipe` | 0.10.14 | Last version with the `mp.solutions` API used here |
| `protobuf` | 4.25.9 | Required by MediaPipe 0.10.14 |
| `tensorflow` | 2.17.1 | Loads `hand_gesture_model_improve.h5` |
| `openai` | ≥ 1.0 | Uses the `OpenAI()` client (Groq is OpenAI-compatible) |
| `webrtcvad-wheels` | — | Prebuilt replacement for `webrtcvad`, which fails to compile on Windows |
| `google-genai` | — | Provides `from google import genai` (Gemini). Do **not** install the unrelated `genai` package (it forces `openai<0.28`) or the old `google-generativeai` SDK (it needs protobuf ≥ 5, which breaks MediaPipe) |

`uv pip check` / `pip check` will report that `ultralytics` and `djitellopy`
want `opencv-python` — this is expected; `cv2` is provided by
`opencv-contrib-python`.

### The OpenCV trap

`ultralytics` depends on `opencv-python`, and installing it overwrites the
`cv2` module from `opencv-contrib-python`. If you ever install or upgrade
packages with `uv pip install`, run this afterwards:

```powershell
uv pip uninstall opencv-python
uv pip install --reinstall-package opencv-contrib-python "opencv-contrib-python==4.10.0.84"
uv pip list | Select-String opencv      # must show ONLY opencv-contrib-python
```

---

## 5. Verify the imports

```powershell
python -c "import cv2, mediapipe, tensorflow, webrtcvad, djitellopy; from openai import OpenAI; from google import genai; print('OK', cv2.__version__, hasattr(cv2,'CascadeClassifier'), hasattr(mediapipe,'solutions'))"
```

Expected (after TensorFlow log noise): `OK 4.10.0 True True`

---

## 6. API keys (`.env`)

Create a file named `.env` in the project root. It is git-ignored — never
commit it.

```dotenv
# Voice commands: LLM fallback for intents the regex parser doesn't catch
# Free key: https://console.groq.com/keys
GROQ_API_KEY=your_groq_key

# Reporter guard (identifies the person holding the microphone).
# Any one of these enables it; without them it auto-disables.
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

Everything else (gestures, pose gate, face tracking, keyboard, offline
speech-to-text) works without any keys.

---

## 7. Download the Whisper model and run the tests

### 7.1 Whisper weights

Whisper weights are **not** in the repo. They download automatically into
`model\faster-whisper-<size>\` on first run, but it's more reliable to fetch
them up front:

```powershell
# Disabling Xet avoids a download that can stall at 0 bytes on some networks
$env:HF_HUB_DISABLE_XET = "1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir=r'model\faster-whisper-small')"
```

**Choose the model size for your CPU.** Measured on a laptop CPU for a 3-second
voice command:

| Model | Download | Transcribe time | Recommendation |
|---|---|---|---|
| `tiny` | 75 MB | ~0.8 s | Fastest; fine for short commands |
| `small` | 460 MB | ~2.8 s | **Best balance on CPU** |
| `medium` | 1.5 GB | ~8 s | Too slow for live flight without a GPU |

Set the size in `control\voice_model.py`:

```python
WHISPER_MODEL_SIZE = "small"
```

If a model ever fails to load with *"invalid payload size"* or similar, the
`model.bin` is truncated — delete that `model\faster-whisper-<size>\` folder
and download it again.

### 7.2 Model test (no drone, no GUI, no mic)

```powershell
python test_script\test_models.py
```

Expected ending:

```
[PASS] model files ...
[PASS] gesture engine ...
[PASS] pose gate ...
[PASS] yolo person tracking ...
[PASS] face cascade ...
[PASS] voice pipeline (whisper) ...
RESULT: all 6 checks passed
```

Use `--skip-voice` to skip the Whisper load. A scikit-learn
`InconsistentVersionWarning` about `LabelEncoder` is expected and harmless.

### 7.3 Webcam demo (full pipeline, fake drone)

```powershell
python test_script\demo_webcam.py               # gestures, pose, face tracking
python test_script\demo_webcam.py --with-voice  # + microphone voice commands
python test_script\demo_webcam.py --camera 1    # if webcam 0 is the wrong camera
```

Nothing flies — a `FakeDrone` records the commands. What to check:

1. **Pose gate:** raise a hand → a green box appears around it.
2. **Gestures:** make a trained sign with the raised hand → the box label shows
   the gesture name; the **Hand Crop** tab shows the landmarks.
3. **Takeoff + movement:** press `e` (fake takeoff), then hold a gesture or use
   the keys; the **Position Map** tab draws the path.
4. **Face tracking:** while "flying", hold the *flip* gesture to toggle Face
   Tracking Mode; the console prints `Fb … yaw …` as it follows your face.
5. **Voice** (`--with-voice`): say e.g. *"move forward"* → the console shows
   `[voice] transcript: …` and the command is applied.
6. Press `q` to (fake) land and quit.

---

## 8. Flying the real Tello

> ⚠️ Fly indoors in an open space with no people within 2 m for the first
> flights. Keep a finger on `q` (land) / `x` (emergency land).

1. **Charge** the Tello battery fully and power it on.
2. **Connect the PC's Wi-Fi** to the `TELLO-XXXXXX` network.
   The Tello network has no internet, so cloud features (Groq/Gemini voice
   LLM and reporter guard) only work if the PC has a second connection
   (Ethernet or a USB Wi-Fi adapter). Gestures, face tracking and offline
   Whisper work without internet.
3. **Allow Python through Windows Firewall** (Private and Public) when
   prompted on first run — the drone uses UDP ports 8889 (commands) and
   11111 (video).
4. **Calibrate once per drone** (it takes off and flies short legs):

   ```powershell
   python scripts\calibrate_and_map.py
   ```

   Copy the printed constants into `control\position_tracker.py`:
   `RC_TO_CM_S` and `RC_TO_DEG_S`. These drive the position map and
   return-home accuracy. Re-run if you change drone or battery behaviour
   changes noticeably.
5. **Fly:**

   ```powershell
   python main.py
   ```

   It connects to the Tello immediately on start.

### Flight controls

| Input | Action |
|---|---|
| `e` | Take off (blocked if battery is too low) |
| Arrow keys | Move left / right / forward / back |
| `w` / `s` | Up / down |
| `a` / `d` | Yaw left / right |
| `q` | Land and quit the program |
| `x` | Emergency land |
| `r` | Return home (climb → fly back → descend); `x` aborts it |
| `f` | Toggle Face Tracking Mode (top-left label shows the mode). In face mode the drone follows your face and **ignores** arrow/w/s/a/d, gesture and voice movement — press `f` again to get manual control back |
| `o` | Toggle reporter guard (ON = only the mic holder can command) |
| Raise hand + sign | Gesture commands (up, down, left, right, move forward/backward, turn left/right, stop = land, flip = face-tracking toggle) |
| Voice (always on in `main.py`) | e.g. "go up", "move forward two meters", "turn left", "hover" |
| Close the window | Lands safely and exits |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `failed to hardlink files` / slow uv install | `$env:UV_LINK_MODE = "copy"` |
| `webrtcvad` fails to build | Use `webrtcvad-wheels` (already in the lock file) |
| pip: `Cannot install … openai==3.24.0 … genai depends on openai<0.28` | You have an old requirements/lock file listing `genai`. Pull the latest repo (it's removed) and use `pip install --no-deps -r requirements-lock.txt` |
| pip: `ResolutionImpossible` about `protobuf` | `google-generativeai` / `google-api-core` crept in — uninstall them; only `google-genai` is needed |
| `cv2` has no `CascadeClassifier`, or `module 'cv2' has no attribute …` | `opencv-python` got installed — see *The OpenCV trap* (section 4) |
| `mediapipe has no attribute 'solutions'` | Wrong MediaPipe version → `uv pip sync requirements-lock.txt` |
| `No module named 'google.genai'` | `uv pip install google-genai "protobuf==4.25.9"` |
| `The filename or extension is too long` / DLL load fails | Move the project to a shorter path or enable long paths (section 1) |
| Whisper download stuck at 0 bytes | `$env:HF_HUB_DISABLE_XET = "1"` and retry |
| Whisper `invalid payload size` / load error | Truncated `model.bin` → delete the model folder, re-download |
| `[Reporter] DISABLED: …` | Add a Gemini/Groq/OpenRouter key to `.env`, or ignore (optional feature) |
| `Could not open webcam` | Close other camera apps; try `--camera 1` |
| Tello connects but no video | Allow Python through the firewall (UDP 11111); stay close to the drone |
