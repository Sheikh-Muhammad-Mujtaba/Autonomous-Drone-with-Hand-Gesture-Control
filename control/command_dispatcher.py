"""
Unified command format for keyboard, gesture, and (future) voice.

RC merge matches the original gestureControl() pattern:

    if kp.getKey("LEFT") or gesture == "right":
        lr = -speed1
    elif kp.getKey("RIGHT") or gesture == "left":
        lr = speed1

Each adapter stores signed RC on that axis. Merge then applies the same
if/elif order as the original: the first matching branch wins (for lr/yv
that is the negative branch; for fb/ud that is the positive branch).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from control.position_tracker import PositionTracker

from control.position_tracker import RC_TO_CM_S, RC_TO_DEG_S


SPEED1 = 30  # same as gestureControl(); do not retune
MIN_TAKEOFF_BATTERY = 20


@dataclass
class DroneCommand:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yv: int = 0
    mode: bool = False  # desired fmode after this command is applied
    action: Optional[str] = None  # primary one-shot (takeoff/land/emergency/flip)
    # Same-frame multi-action (original could land AND takeoff AND emergency
    # in one iteration). execute_command runs these in original order.
    actions: Tuple[str, ...] = field(default_factory=tuple)
    # Voice timed move; ignored by execute_command until step 3/4.
    duration_s: Optional[float] = None


GESTURE_LATCH_S = 0.3  # how long a one-frame gesture pulse stays "held"


@dataclass
class DispatcherState:
    """Shared flight state — replaces scattered `global isFlying` / `fmode`."""

    is_flying: bool = False
    fmode: bool = False
    min_takeoff_battery: int = MIN_TAKEOFF_BATTERY
    last_rc_t: float = 0.0
    latched_gesture: Optional[str] = None
    latch_until: float = 0.0
    tracker: Optional[PositionTracker] = None

ONE_SHOT_GESTURES = {"flip", "stop"}


def latch_gesture(state: DispatcherState, gesture: str, now: float,
                   duration: float = GESTURE_LATCH_S) -> str:
    """
    Continuous gestures are latched briefly so one-frame detections
    survive the 30 Hz RC rate limit.

    One-shot gestures such as flip/stop are NOT latched because
    replaying them would execute the action multiple times.
    """
    if gesture and gesture != "None":
        # One-shot actions: execute exactly once.
        if gesture in ONE_SHOT_GESTURES:
            state.latched_gesture = None
            state.latch_until = 0.0
            return gesture

        # Continuous movement gestures: latch normally.
        state.latched_gesture = gesture
        state.latch_until = now + duration
        return gesture

    # Continue a previously latched movement gesture.
    if state.latched_gesture and now < state.latch_until:
        return state.latched_gesture

    state.latched_gesture = None
    return "None"

def clear_gesture_latch(state: DispatcherState) -> None:
    state.latched_gesture = None
    state.latch_until = 0.0


def _actions_of(command: DroneCommand) -> Tuple[str, ...]:
    if command.actions:
        return command.actions
    if command.action:
        return (command.action,)
    return ()


def _with_actions(*flags: str) -> Tuple[Optional[str], Tuple[str, ...]]:
    actions = tuple(a for a in flags if a)
    primary = actions[-1] if actions else None
    return primary, actions


def from_keyboard(kp_module) -> DroneCommand:
    """Keyboard only. Must be called once per loop (getKey pumps pygame)."""
    lr, fb, ud, yv = 0, 0, 0, 0

    if kp_module.getKey("LEFT"):
        lr = -SPEED1
    elif kp_module.getKey("RIGHT"):
        lr = SPEED1

    if kp_module.getKey("UP"):
        fb = SPEED1
    elif kp_module.getKey("DOWN"):
        fb = -SPEED1

    if kp_module.getKey("w"):
        ud = SPEED1
    elif kp_module.getKey("s"):
        ud = -SPEED1

    if kp_module.getKey("a"):
        yv = -SPEED1
    elif kp_module.getKey("d"):
        yv = SPEED1

    # Call every one-shot reader every frame so edge detection stays in sync.
    want_land = kp_module.getKeyPressedOnce("q")
    want_takeoff = kp_module.getKeyPressedOnce("e")
    want_emergency = kp_module.getKeyPressedOnce("x")
    want_return_home = kp_module.getKeyPressedOnce("r")
    # f toggles Face Tracking Mode (same as the "flip" gesture), so there
    # is always a keyboard way out of face mode, where manual RC is ignored.
    want_face_toggle = kp_module.getKeyPressedOnce("f")

    primary, actions = _with_actions(
        "land" if want_land else "",
        "flip" if want_face_toggle else "",
        "takeoff" if want_takeoff else "",
        "emergency" if want_emergency else "",
        "return_home" if want_return_home else "",
    )
    return DroneCommand(lr=lr, fb=fb, ud=ud, yv=yv, action=primary, actions=actions)


def from_gesture(gesture: str) -> DroneCommand:
    """Gesture string only (already debounced by the main loop)."""
    lr, fb, ud, yv = 0, 0, 0, 0

    # Mirrored vs keyboard: "right" → leftward RC, "left" → rightward RC
    # (same as original `gesture == "right"` on the LEFT branch).
    if gesture == "right":
        lr = -SPEED1
    elif gesture == "left":
        lr = SPEED1

    if gesture == "move farward":
        fb = SPEED1
    elif gesture == "move backward":
        fb = -SPEED1

    if gesture == "up":
        ud = SPEED1
    elif gesture == "down":
        ud = -SPEED1

    if gesture == "turn right":
        yv = -SPEED1
    elif gesture == "turn left":
        yv = SPEED1

    primary, actions = _with_actions(
        "land" if gesture == "stop" else "",
        "flip" if gesture == "flip" else "",
    )
    return DroneCommand(lr=lr, fb=fb, ud=ud, yv=yv, action=primary, actions=actions)


def from_voice(intent: dict) -> DroneCommand:
    """
    Map a parsed voice intent onto the same RC / action format.

    Expected shapes (from voice_model.resolve_command):
        {"action": "takeoff"|"land"|"hover"|"follow"|"return_home"|"describe_scene"}
        {"action": "move", "direction": "forward"|..., "distance_cm": 50}
        {"action": "rotate", "direction": "cw"|"ccw", "degrees": 90}

    Voice directions are semantic (same as keyboard), NOT mirrored like
    gesture labels. Timed travel is stored on duration_s; execute_command
    does not wait on it yet (continuous RC only, same as today).
    """
    if not intent:
        return DroneCommand()

    action_name = (intent.get("action") or "").lower()
    direction = (intent.get("direction") or "").lower()

    # voice_model.py uses distance_cm; accept distance_m as legacy fallback.
    distance_cm = intent.get("distance_cm")
    if distance_cm is None:
        distance_cm = intent.get("distance_m")

    lr, fb, ud, yv = 0, 0, 0, 0
    duration_s = None
    actions: Tuple[str, ...] = ()

    if action_name in ("takeoff", "land", "emergency", "flip", "return_home",
                        "follow", "hover"):
        actions = (action_name,)
    elif action_name == "describe_scene":
        # Handled entirely in the voice thread — never reaches the queue.
        # Return a no-op so the merge path is safe if it ever does arrive.
        pass
    elif action_name in ("move", "go", ""):
        # Keyboard-semantic axes (not gesture-mirrored).
        dir_to_rc = {
            "forward": (0, SPEED1, 0, 0),
            "farward": (0, SPEED1, 0, 0),
            "back": (0, -SPEED1, 0, 0),  # canonical key emitted by voice_model.py
            "backward": (0, -SPEED1, 0, 0),  # legacy alias (LLM safety)
            "left": (-SPEED1, 0, 0, 0),
            "right": (SPEED1, 0, 0, 0),
            "up": (0, 0, SPEED1, 0),
            "down": (0, 0, -SPEED1, 0),
            "turn left": (0, 0, 0, SPEED1),
            "turn right": (0, 0, 0, -SPEED1),
        }
        if direction in dir_to_rc:
            lr, fb, ud, yv = dir_to_rc[direction]
        if distance_cm is not None:
            try:
                speed_cm_s = SPEED1 * RC_TO_CM_S
                duration_s = float(distance_cm) / speed_cm_s if speed_cm_s > 0 else None
            except (TypeError, ValueError):
                duration_s = None
    elif action_name == "rotate":
        # voice_model: {"action": "rotate", "direction": "cw"|"ccw", "degrees": 90}
        degrees = intent.get("degrees", 90)
        if direction == "ccw":
            yv = -SPEED1
        elif direction == "cw":
            yv = SPEED1
        if degrees is not None:
            try:
                speed_deg_s = SPEED1 * RC_TO_DEG_S
                duration_s = float(degrees) / speed_deg_s if speed_deg_s > 0 else None
            except (TypeError, ValueError):
                duration_s = None

    primary = actions[-1] if actions else None
    return DroneCommand(
        lr=lr,
        fb=fb,
        ud=ud,
        yv=yv,
        action=primary,
        actions=actions,
        duration_s=duration_s,
    )


def _merge_axis(values, prefer_negative: bool) -> int:
    """
    Reproduce one original if/elif pair.

    lr / yv first branch is negative (LEFT, yaw-a / turn-right).
    fb / ud first branch is positive (UP, w / farward / up).
    """
    vals = list(values)
    has_neg = any(v < 0 for v in vals)
    has_pos = any(v > 0 for v in vals)
    if prefer_negative:
        if has_neg:
            return -SPEED1
        if has_pos:
            return SPEED1
    else:
        if has_pos:
            return SPEED1
        if has_neg:
            return -SPEED1
    return 0


def merge_commands(*commands: DroneCommand) -> DroneCommand:
    """
    Combine simultaneous sources.

    Per-axis first-branch wins (same as gestureControl if/elif):
      lr: LEFT or gesture "right"  (negative) before RIGHT / "left"
      fb: UP or "move farward"     (positive) before DOWN / "move backward"
      ud: w or "up"                (positive) before s / "down"
      yv: a or "turn right"        (negative) before d / "turn left"

    Actions: concatenated in source order; execute_command applies them in
    the original gestureControl sequence (land → flip → takeoff → emergency).
    """
    if not commands:
        return DroneCommand()

    lr = _merge_axis((c.lr for c in commands), prefer_negative=True)
    fb = _merge_axis((c.fb for c in commands), prefer_negative=False)
    ud = _merge_axis((c.ud for c in commands), prefer_negative=False)
    yv = _merge_axis((c.yv for c in commands), prefer_negative=True)

    merged_actions: list[str] = []
    for c in commands:
        for a in _actions_of(c):
            if a not in merged_actions:
                merged_actions.append(a)

    duration_s = next((c.duration_s for c in commands if c.duration_s), None)
    primary = merged_actions[-1] if merged_actions else None
    return DroneCommand(
        lr=lr,
        fb=fb,
        ud=ud,
        yv=yv,
        action=primary,
        actions=tuple(merged_actions),
        duration_s=duration_s,
    )


def safe_takeoff(me, state: DispatcherState) -> None:
    if state.is_flying:
        print("Already flying - ignoring takeoff request.")
        return
    battery = me.get_battery()
    if battery < state.min_takeoff_battery:
        print(f"Battery too low ({battery}%) - takeoff blocked.")
        return
    me.takeoff()
    state.is_flying = True
    # Reset dead-reckoning position to origin so every flight starts fresh.
    if state.tracker is not None:
        state.tracker.reset()
        print("[Tracker] Position reset to origin (0, 0, 0, 0).")


def safe_land(me, state: DispatcherState) -> None:
    if not state.is_flying:
        return
    me.land()
    state.is_flying = False


def execute_command(me, state: DispatcherState, command: DroneCommand) -> Optional[Tuple[int, int, int, int]]:
    """
    Apply one-shot actions (original order), then send RC if flying
    and not in face-tracking mode. Face PID stays in the main loop.

    Returns the (lr, fb, ud, yv) tuple that was actually sent this tick,
    or None if no RC was sent (rate-limited, not flying, or fmode active).
    The caller should feed this into PositionTracker.update() so dead
    reckoning stays in sync with real send_rc_control calls.
    """
    actions = _actions_of(command)

    # Original order inside gestureControl():
    #   q/stop land → flip toggle → e takeoff → x emergency (controlled land)
    if "land" in actions:
        safe_land(me, state)
        clear_gesture_latch(state)  # don't replay a stale "up" after re-takeoff

    if "flip" in actions:
        state.fmode = not state.fmode

    if "takeoff" in actions:
        safe_takeoff(me, state)

    if "follow" in actions:
        state.fmode = True
        clear_gesture_latch(state)

    if "hover" in actions:
        state.fmode = False
        clear_gesture_latch(state)

    if "emergency" in actions:
        print("EMERGENCY LAND triggered")
        safe_land(me, state)  # guarded wrapper — no-op on the ground, no
                              # TelloException; controlled me.land() in flight
        state.is_flying = False  # shared state — not a local variable
        clear_gesture_latch(state)

    # return_home takes over the control loop entirely — it blocks until
    # the three-phase sequence completes (or times out).  Skip normal RC
    # send for this tick.
    if "return_home" in actions:
        if state.tracker is not None:
            # Lazy import to avoid circular dependency at module-load time.
            from control.position_tracker import return_home

            return_home(me, state, state.tracker)
        else:
            print("[RTH] No position tracker available — cannot return home.")
        return None

    command.mode = state.fmode

    # duration_s is unused here on purpose (continuous RC, same as today).
    # Tello SDK: keep ~30 Hz RC. Faster floods WiFi and delays the video UDP stream
    # (v1 felt live because model.predict kept the loop near 10 Hz).
    if state.is_flying and not state.fmode:
        now = time.perf_counter()
        if now - state.last_rc_t >= (1.0 / 30.0):
            me.send_rc_control(command.lr, command.fb, command.ud, command.yv)
            state.last_rc_t = now
            return (command.lr, command.fb, command.ud, command.yv)
    return None
