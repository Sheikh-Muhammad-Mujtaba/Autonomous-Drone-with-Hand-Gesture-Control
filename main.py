"""
Drone Reporter — autonomous journalism drone for the DJI Tello.

Entry point.  Wires perception (gesture / pose / reporter), control
(command dispatcher, position tracking, voice) and UI (unified tabbed
window) into one 30 Hz flight loop:

    video -> pose gate -> gesture crop -> reporter guard
                                      \
                                       -> keyboard / gesture / voice merge
                                            -> execute_command -> RC

Face Tracking Mode runs its own PID that sends RC directly.  The
position tracker follows every RC actually sent so the live map stays
honest.  Press O to toggle the reporter guard (ON = reporter only,
OFF = anyone).

Run from the project root:
    python main.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import logging
import queue
import threading
import time

import cv2
from djitellopy import tello

from config import (
    DEBUG_TIMING,
    DISPLAY_H,
    DISPLAY_W,
    TARGET_FPS,
    TIMING_EVERY,
)
from control.command_dispatcher import (
    DispatcherState,
    execute_command,
    from_gesture,
    from_keyboard,
    from_voice,
    latch_gesture,
    merge_commands,
    safe_land,
)
from control.gesture_debouncer import NO_GESTURE, GestureDebouncer
from control.position_map import PositionMapRenderer
from control.position_tracker import PositionTracker, set_abort_key_source
from core.face_tracking import findFace, set_drone, trackFace
from core.gesture_worker import GestureWorker, hand_crop_from_raw
from core.pose_worker import PoseWorker
from core.video import LowLatencyFrameRead
from core.voice_worker import VoiceWorker
from perception.gesture_detector import last_timing_ms
from perception.reporter_tracker import ReporterTracker
from ui.unified_gui import UnifiedGUI

# Tello logs every send_rc_control; printing that 30x/s stalls Windows + OpenCV.
tello.Tello.LOGGER.setLevel(logging.ERROR)

# Must be set before VideoCapture: tell ffmpeg not to queue seconds of UDP video.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "fflags;nobuffer+discardcorrupt|flags;low_delay|max_delay;0"
)

gui = UnifiedGUI()

me = tello.Tello()
me.connect()

print("Battery:", me.get_battery())

me.streamon()
time.sleep(2)
# Do not use me.get_frame_read() (PyAV + PIL queue). Latest BGR frame only.
frame_read = LowLatencyFrameRead(me.get_udp_video_address())
set_drone(me)  # trackFace() sends its own RC — give it the drone handle
set_abort_key_source(gui)  # "x" aborts return-home via this window

dispatcher_state = DispatcherState()
tracker = PositionTracker()
dispatcher_state.tracker = tracker
map_renderer = PositionMapRenderer(canvas_size=(400, 400), cm_per_grid=50)
_prev_flying = False
voice_command_queue = queue.Queue()
voice_stop_event = threading.Event()


gesture1 = NO_GESTURE
gesture_debouncer = GestureDebouncer()
pError = 0

w, h = DISPLAY_W, DISPLAY_H

pid = [0.2, 0.2, 0]


# On-screen control legend (Drone View overlay). Keep in sync with
# from_keyboard() in command_dispatcher.py and the gesture pipeline.
_KEY_TABLE = [
    ("Arrows", "Move"),
    ("w / s", "Up / Down"),
    ("a / d", "Yaw"),
    ("e", "Takeoff"),
    ("q", "Land + quit"),
    ("x", "Emergency land"),
    ("r", "Return home"),
    ("f", "Face tracking"),
    ("o", "Reporter guard"),
    ("Hand up", "Enable gestures"),
]


def draw_key_table(img):
    """Draw a compact control legend onto the Drone View frame.

    Uses a semi-transparent panel so the text stays readable regardless
    of what the camera is looking at. Returns the (possibly new) frame
    because addWeighted produces a fresh array.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.35, 1
    line_h = 13
    x0 = 6
    top = 46
    panel_w = 132
    panel_h = line_h * (len(_KEY_TABLE) + 1) + 8

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 3, top - 12),
                  (x0 + panel_w, top + panel_h - 12), (0, 0, 0), cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    yy = top
    cv2.putText(img, "CONTROLS", (x0, yy), font, scale, (0, 255, 255), thickness)
    for key, action in _KEY_TABLE:
        yy += line_h
        cv2.putText(img, key, (x0, yy), font, scale, (0, 255, 0), thickness)
        cv2.putText(img, action, (x0 + 56, yy), font, scale, (255, 255, 255), thickness)
    return img


# FIX: wrap main loop so the drone always lands / disconnects cleanly,
# even if an exception happens mid-flight.
class _ReporterToggle:
    """Mutable ON/OFF switch for the reporter guard (O key toggles it)."""
    def __init__(self):
        self.enabled = True


gesture_worker = None
pose_worker = None
voice_worker = None
reporter_worker = None
reporter_toggle = _ReporterToggle()
_latched_voice_cmd = None
_voice_latch_until = 0.0
try:
    timing_sums = {
        "grab": 0.0,
        "mediapipe": 0.0,
        "inference": 0.0,
        "dispatch": 0.0,
        "display": 0.0,
        "loop": 0.0,
    }
    timing_n = 0
    last_battery_t = 0.0
    charge = 0
    last_gesture_seq = -1
    prev_loop_t0 = 0.0
    gesture_worker = GestureWorker()
    gesture_worker.start()
    pose_worker = PoseWorker()
    pose_worker.start()

    # Reporter (mic-holder) guard: YOLO + VLM on its own thread.  If it
    # cannot start (no API key / missing model), gesture + voice fall back
    # to the pre-reporter gating so the demo can still fly.
    reporter_worker = None
    try:
        reporter_worker = ReporterTracker()
        print("[Reporter] Mic-holder tracking ON — gesture/voice commands "
              "are accepted from the reporter only.")
    except Exception as exc:
        print(f"[Reporter] DISABLED: {exc}")
        print("[Reporter] Gesture/voice fall back to pre-reporter gating.")

    voice_worker = VoiceWorker(
        frame_read, pose_worker, reporter_worker, reporter_toggle,
        voice_stop_event, voice_command_queue,
    )
    voice_worker.start()

    gui.set_reporter_state(reporter_toggle.enabled,
                           available=reporter_worker is not None)

    while True:
        loop_t0 = time.perf_counter()
        loop_dt = loop_t0 - prev_loop_t0 if prev_loop_t0 > 0 else 1.0 / TARGET_FPS
        prev_loop_t0 = loop_t0

        # Reset map on takeoff: is_flying transition False → True
        if dispatcher_state.is_flying and not _prev_flying:
            map_renderer.clear()
        _prev_flying = dispatcher_state.is_flying

        t0 = time.perf_counter()
        raw = frame_read.frame
        grab_ms = (time.perf_counter() - t0) * 1000.0

        if raw is None:
            continue

        # Decoder already emits BGR. v1 used BGR2RGB only because djitellopy
        # delivered PIL RGB — that swap is not needed here.
        img = cv2.resize(raw, (w, h))

        # Reporter tracker consumes the full-res raw frame (best fidelity
        # for YOLO boxes and the VLM crop).
        if reporter_worker is not None and reporter_toggle.enabled:
            reporter_worker.submit_frame(raw)

        # Pose gate on the full frame: find a raised hand, then crop the
        # gesture model input around that hand. If no hand is up, the
        # gesture model is never submitted to (and never loaded).
        pose_worker.submit_frame(img)
        hand_info, pose_vis = pose_worker.latest()

        hand_bbox = None
        hand_allowed = True
        if hand_info is not None:
            hand_bbox = hand_info["bbox"]
            # Reporter guard: the pose gate (hand up) already ran; only the
            # mic holder's raised hand may drive the gesture model.
            if reporter_worker is not None and reporter_toggle.enabled:
                hand_allowed = reporter_worker.hand_belongs_to_reporter(
                    hand_bbox, w, h
                )
            if hand_allowed:
                crop, crop_box, raw_size = hand_crop_from_raw(raw, hand_bbox, (w, h))
                gesture_worker.submit_crop(crop, crop_box, raw_size)

        seq, gesture, processed_frame = gesture_worker.latest()
        if hand_bbox is None or not hand_allowed:
            gesture_debouncer.reset()
        elif seq != last_gesture_seq:
            gesture1 = gesture_debouncer.update(gesture, loop_t0)
        last_gesture_seq = seq

        t0 = time.perf_counter()
        kb_cmd = from_keyboard(gui)
        # O toggles the reporter guard: ON = reporter only, OFF = anyone.
        if gui.getKeyPressedOnce("o"):
            if reporter_worker is None:
                print("[Reporter] guard unavailable — the tracker did not start "
                      "(see the [Reporter] DISABLED line at start-up).")
            else:
                reporter_toggle.enabled = not reporter_toggle.enabled
                print("[Reporter] guard " + ("ENABLED" if reporter_toggle.enabled
                                             else "OFF (bypass — anyone)"))
                gui.set_reporter_state(reporter_toggle.enabled)
        held_gesture = latch_gesture(dispatcher_state, gesture1, t0)
        ges_cmd = from_gesture(held_gesture)
        command = merge_commands(kb_cmd, ges_cmd)

        # Drain voice command queue (non-blocking — never blocks loop 1).
        # Continuous movement commands (move/rotate) are latched so they
        # persist across ticks.  "hover" clears the latch.  Distance-based
        # commands expire after their calibrated duration.
        try:
            voice_intent = voice_command_queue.get_nowait()
            voice_cmd = from_voice(voice_intent)
            action_name = (voice_intent.get("action") or "").lower()
            if action_name in ("hover",):
                _latched_voice_cmd = None
                _voice_latch_until = 0.0
            else:
                _latched_voice_cmd = voice_cmd
                if voice_cmd.duration_s is not None:
                    _voice_latch_until = time.perf_counter() + voice_cmd.duration_s
                else:
                    _voice_latch_until = float('inf')
        except queue.Empty:
            pass

        if _latched_voice_cmd is not None:
            if time.perf_counter() < _voice_latch_until:
                command = merge_commands(command, _latched_voice_cmd)
            else:
                _latched_voice_cmd = None
                _voice_latch_until = 0.0

        sent_rc = execute_command(me, dispatcher_state, command)
        if sent_rc is not None:
            # sent_rc is the full (lr, fb, ud, yv) tuple actually sent.
            tracker.update(*sent_rc, loop_dt)

        # Face PID stays here (it sends its own RC). Manual RC is sent
        # inside execute_command only when flying and fmode is off.
        if dispatcher_state.is_flying and dispatcher_state.fmode:
            img2 = cv2.resize(img, (w, h))
            img, info = findFace(img2)
            face_result = trackFace(info, w, pid, pError)
            face_lr, face_fb, face_ud, face_yv, pError = face_result
            tracker.update(face_lr, face_fb, face_ud, face_yv, loop_dt)
        # Plot tracked position on the live map (only while airborne)
        if dispatcher_state.is_flying:
            map_renderer.add_point(tracker.x, tracker.y, tracker.yaw)
        dispatch_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        latest = frame_read.frame
        if latest is not None and not dispatcher_state.fmode:
            img = cv2.resize(latest, (w, h))
        mode_text = "Face Tracking Mode" if dispatcher_state.fmode else "Gesture Mode"
        cv2.putText(img, mode_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        now = time.perf_counter()
        if now - last_battery_t >= 1.0:
            charge = me.get_battery()
            last_battery_t = now
        bat_text = str(charge)
        (btw, _bth), _ = cv2.getTextSize(bat_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 1)
        cv2.putText(img, bat_text, (w - btw - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Reporter status badge — small, bottom-right corner (out of the way).
        if reporter_worker is None:
            rep_text, rep_color = "REPORTER: unavailable", (0, 140, 255)
        elif not reporter_toggle.enabled:
            rep_text, rep_color = "REPORTER: OFF (o)", (128, 128, 128)
        elif reporter_worker.reporter_track_id is None:
            rep_text, rep_color = "REPORTER: none", (0, 165, 255)
        else:
            rep_text = f"REPORTER: ID {reporter_worker.reporter_track_id}"
            rep_color = (0, 255, 0)
        (rtw, _rth), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.4, 1)
        cv2.putText(img, rep_text, (w - rtw - 8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, rep_color, 1)
        # No fixed gesture box anymore — draw the live hand bbox instead,
        # so it's visible exactly where gesture recognition is looking.
        # The recognized gesture is a label attached to the box, exactly
        # like the YOLO "ID n" labels.
        if hand_bbox is not None:
            if (reporter_worker is not None and reporter_toggle.enabled
                    and not hand_allowed):
                box_color = (0, 165, 255)  # raised hand, but NOT the reporter
            else:
                box_color = (0, 255, 0)
            hx1, hy1, hx2, hy2 = hand_bbox
            img = cv2.rectangle(img, (hx1, hy1), (hx2, hy2), box_color, 2)
            g_label = gesture if gesture and gesture != "None" else "hand up"
            cv2.putText(img, g_label, (hx1, max(hy1 - 5, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        gui.show("part View", processed_frame if hand_bbox is not None
                 else (pose_vis if pose_vis is not None else processed_frame))
        img = draw_key_table(img)
        # Draw YOLO person tracking boxes on the Drone View (if reporter
        # tracker is active).  Green = regular person, red = mic holder.
        if reporter_worker is not None and reporter_toggle.enabled:
            for person in reporter_worker.current_tracks_for(w, h):
                x1, y1, x2, y2 = person["box"]
                color = (0, 0, 255) if person["is_mic"] else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"ID {person['id']}"
                if person["is_mic"]:
                    label += " - MIC"
                cv2.putText(img, label, (x1, max(y1 - 5, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        gui.show("Drone View", img)

        map_img = map_renderer.render()
        gui.show("Position Map", map_img)

        gui.update()
        if gui.should_quit:
            break
        display_ms = (time.perf_counter() - t0) * 1000.0

        loop_ms = (time.perf_counter() - loop_t0) * 1000.0
        remain = (1.0 / TARGET_FPS) - (loop_ms / 1000.0)
        if remain > 0:
            time.sleep(remain)

        if DEBUG_TIMING:
            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            timing_sums["grab"] += grab_ms
            timing_sums["mediapipe"] += last_timing_ms["mediapipe"]
            timing_sums["inference"] += last_timing_ms["inference"]
            timing_sums["dispatch"] += dispatch_ms
            timing_sums["display"] += display_ms
            timing_sums["loop"] += loop_ms
            timing_n += 1
            if timing_n >= TIMING_EVERY:
                n = float(timing_n)
                fps = 1000.0 / (timing_sums["loop"] / n) if timing_sums["loop"] else 0.0
                print(
                    "timing_ms avg "
                    f"grab={timing_sums['grab']/n:.1f} "
                    f"mediapipe={timing_sums['mediapipe']/n:.1f} "
                    f"infer={timing_sums['inference']/n:.1f} "
                    f"dispatch={timing_sums['dispatch']/n:.1f} "
                    f"display={timing_sums['display']/n:.1f} "
                    f"loop={timing_sums['loop']/n:.1f} "
                    f"fps={fps:.1f}"
                )
                timing_sums = {k: 0.0 for k in timing_sums}
                timing_n = 0

        gesture1 = NO_GESTURE

finally:
    # FIX (safety): always attempt a clean shutdown, even after a crash/error
    print("Shutting down safely...")
    try:
        voice_stop_event.set()
        if voice_worker is not None:
            voice_worker.stop()
    except Exception:
        pass
    try:
        if reporter_worker is not None:
            reporter_worker.stop()
    except Exception:
        pass
    try:
        if gesture_worker is not None:
            gesture_worker.stop()
    except Exception:
        pass
    try:
        if pose_worker is not None:
            pose_worker.stop()
    except Exception:
        pass
    try:
        safe_land(me, dispatcher_state)
    except Exception as e:
        print("Error while landing:", e)
    try:
        frame_read.stop()
    except Exception:
        pass
    try:
        me.streamoff()
    except Exception as e:
        print("Error while stopping stream:", e)
    try:
        me.end()
    except Exception as e:
        print("Error while ending connection:", e)
    try:
        gui.destroy()
    except Exception:
        pass
    cv2.destroyAllWindows()
