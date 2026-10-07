"""
Dead-reckoning position tracker for DJI Tello.

RC values are used as a velocity proxy — this is NOT measured velocity.
Drift is expected and acceptable for short flights (typical hackathon
duration of 2–5 minutes). The tracker is reset to origin on every takeoff
so each flight starts fresh.

The return_home() routine uses three altitude-aware phases:
  1. Climb to safe cruise height (if too low)
  2. Horizontal return to origin using dead-reckoning vector
  3. Descend to landing approach height, then hand off to safe_land()

Every phase has a hard timeout so a bad dead-reckoning estimate can't
cause infinite flight — the drone will land wherever it ended up.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import cv2
from ui import keypress as kp

if TYPE_CHECKING:
    from control.command_dispatcher import DispatcherState


# Where return_home() reads the emergency-abort key ("x").  The flight loop
# registers its UnifiedGUI here (its getKeyPressedOnce also pumps the window,
# keeping it responsive during the blocking RTH sequence).  When nothing is
# registered, the pygame keypress module is used (calibration scripts).
_abort_key_source = None


def set_abort_key_source(source) -> None:
    """Register an object with getKeyPressedOnce(key) for the RTH abort key."""
    global _abort_key_source
    _abort_key_source = source


def _abort_pressed() -> bool:
    source = _abort_key_source if _abort_key_source is not None else kp
    return source.getKeyPressedOnce("x")


# ---------------------------------------------------------------------------
# Calibration constants — all in cm / degrees / seconds
# ---------------------------------------------------------------------------

# Measured on this specific drone with calibrate_and_map.py Test A
# (forward at RC=30 for 2.0 s).  The old 8.0 was a spec-sheet estimate
# that was ~7x too high.  Re-measure if the drone, battery, or flight
# environment changes significantly.
RC_TO_CM_S = 1.4050    # measured (average with earlier 1.1333 if you want more confidence)
RC_TO_DEG_S = 0.4833   # measured (average with earlier 0.7333 if you want more confidence)
                      # (sign was already correct, magnitude was ~36% high)

# Return-to-home geometry
SAFE_HEIGHT_MIN_CM = 152        # ~5 ft — minimum safe altitude before crossing
SAFE_CRUISE_HEIGHT_CM = 183     # ~6 ft — target climb height for return leg
LANDING_APPROACH_HEIGHT_CM = 60  # descend to this before handing off to safe_land

# Timeout safeties — prevent infinite flight on bad dead-reckoning data
MAX_ASCEND_TIME_SEC = 5
MAX_RETURN_TIME_SEC = 15
MAX_DESCEND_TIME_SEC = 5

# ±20 cm counts as "arrived" at origin for phase 2
ARRIVAL_TOLERANCE_CM = 20.0

# RC speeds used during return-to-home phases
RTH_UP_SPEED = 30      # same order of magnitude as SPEED1 for predictability
RTH_DOWN_SPEED = 25    # slightly gentler descent
RTH_CRUISE_SPEED = 40  # faster horizontal return than normal cruise
RTH_YAW_THRESHOLD_DEG = 15  # if yaw error > this, rotate first; else fly+trim


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

@dataclass
class PositionTracker:
    """Dead-reckoning position relative to takeoff point (0, 0, 0, 0).

    x/y are world-frame horizontal coordinates in cm.  z is NOT integrated
    from RC — the Tello's barometric height sensor is more reliable, so z
    is updated externally via set_height() from the drone's own sensor.
    yaw is the world-frame heading in degrees (0 = takeoff heading).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    # ------------------------------------------------------------------
    def update(self, lr: int, fb: int, ud: int, yv: int, dt: float) -> None:
        """Integrate one tick of commanded RC into the position estimate.

        lr/fb are in the drone's body frame — we rotate by current yaw
        to get world-frame displacement.  yaw is integrated directly from
        the yaw rate (yv).  ud is intentionally ignored for z because the
        barometer is more reliable than RC integration for altitude.
        """
        if dt <= 0:
            return

        # Yaw integration (world-frame heading)
        yaw_rate_deg_s = yv * RC_TO_DEG_S
        self.yaw = (self.yaw + yaw_rate_deg_s * dt) % 360.0

        # Body-frame velocity → world-frame displacement
        # fb: forward (+) / backward (-) in body frame
        # lr: left  (+) / right  (-) in body frame
        vx_body = fb * RC_TO_CM_S
        vy_body = lr * RC_TO_CM_S

        yaw_rad = math.radians(self.yaw)
        # Rotate body-frame velocity into world frame
        dx = vx_body * math.cos(yaw_rad) - vy_body * math.sin(yaw_rad)
        dy = vx_body * math.sin(yaw_rad) + vy_body * math.cos(yaw_rad)

        self.x += dx * dt
        self.y += dy * dt

    # ------------------------------------------------------------------
    def set_height(self, height_cm: float) -> None:
        """Override z from a sensor reading (barometer / ToF)."""
        self.z = height_cm

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all axes to origin — call on every takeoff."""
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

    # ------------------------------------------------------------------
    @property
    def distance_to_origin(self) -> float:
        """Horizontal distance from takeoff point in cm."""
        return math.hypot(self.x, self.y)


# ---------------------------------------------------------------------------
# Height helper
# ---------------------------------------------------------------------------

def get_current_height_cm(me) -> float:
    """Best-effort height reading in cm.

    Tello's get_height() returns barometric height in dm, which is coarse
    near ground.  get_distance_tof() is more reliable below ~2 m but
    returns 0/None when out of range.  We try barometer first, then fall
    back to ToF.
    """
    try:
        h = me.get_height()
        if h is not None and h > 0:
            return float(h)
    except Exception:
        pass
    try:
        tof = me.get_distance_tof()
        if tof is not None and tof > 0:
            return float(tof)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Return-to-home
# ---------------------------------------------------------------------------

def return_home(me, state: DispatcherState, tracker: PositionTracker) -> None:
    """Three-phase altitude-aware return-to-home.

    PHASE 1 — Climb to safe cruise height if too low.
    PHASE 2 — Horizontal return to (x=0, y=0), holding altitude.
    PHASE 3 — Descend to landing approach height, then hand off to safe_land.

    Each phase has a hard timeout so bad dead-reckoning data doesn't
    cause infinite flight.  On any timeout we log a warning and proceed
    to the next phase rather than hanging.
    """
    if not state.is_flying:
        print("[RTH] Not flying — skipping return-to-home.")
        return

    print(
        "[RTH] Starting return-to-home from "
        f"(x={tracker.x:.0f}, y={tracker.y:.0f}) "
        f"dist={tracker.distance_to_origin:.0f}cm yaw={tracker.yaw:.0f}°"
    )

    # ------------------------------------------------------------------
    # PHASE 1 — Climb to safe altitude
    # ------------------------------------------------------------------
    height = get_current_height_cm(me)
    print(f"[RTH] Phase 1 — current height: {height:.0f} cm")

    if height < SAFE_HEIGHT_MIN_CM:
        print(f"[RTH] Climbing to {SAFE_CRUISE_HEIGHT_CM} cm...")
        t_start = time.perf_counter()
        while True:
            cv2.waitKey(1)  # drain OpenCV events so the window stays responsive
            if _abort_pressed():
                print("EMERGENCY LAND triggered during return-home")
                from control.command_dispatcher import safe_land

                safe_land(me, state)  # guarded: no-op when already grounded
                state.is_flying = False
                from control.command_dispatcher import clear_gesture_latch

                clear_gesture_latch(state)
                return  # exit return_home() immediately, skip remaining phases
            height = get_current_height_cm(me)
            if height >= SAFE_CRUISE_HEIGHT_CM:
                break
            if time.perf_counter() - t_start > MAX_ASCEND_TIME_SEC:
                print(
                    f"[RTH] WARNING: ascend timeout ({MAX_ASCEND_TIME_SEC}s), "
                    f"height={height:.0f}cm — proceeding anyway"
                )
                break
            me.send_rc_control(0, 0, RTH_UP_SPEED, 0)
            time.sleep(0.05)
        # Stop vertical movement before transitioning
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)
        print(f"[RTH] Phase 1 complete — height: {get_current_height_cm(me):.0f} cm")
    else:
        print(
            f"[RTH] Phase 1 skipped — already at {height:.0f} cm "
            f"(>= {SAFE_HEIGHT_MIN_CM})"
        )

    # ------------------------------------------------------------------
    # PHASE 2 — Horizontal return to origin
    # ------------------------------------------------------------------
    print(
        f"[RTH] Phase 2 — returning to origin, "
        f"dist={tracker.distance_to_origin:.0f}cm"
    )
    t_start = time.perf_counter()
    while tracker.distance_to_origin > ARRIVAL_TOLERANCE_CM:
        cv2.waitKey(1)  # drain OpenCV events so the window stays responsive
        if _abort_pressed():
            print("EMERGENCY LAND triggered during return-home")
            from control.command_dispatcher import safe_land

            safe_land(me, state)  # guarded: no-op when already grounded
            state.is_flying = False
            from control.command_dispatcher import clear_gesture_latch

            clear_gesture_latch(state)
            return  # exit return_home() immediately, skip remaining phases
        if time.perf_counter() - t_start > MAX_RETURN_TIME_SEC:
            print(
                f"[RTH] WARNING: return timeout ({MAX_RETURN_TIME_SEC}s), "
                f"remaining dist={tracker.distance_to_origin:.0f}cm "
                f"— proceeding to land"
            )
            break

        # Steer toward origin.  The tracker's (x, y) is the drone's
        # position relative to origin, so the vector from drone to origin
        # is (-x, -y).
        target_yaw = math.degrees(math.atan2(-tracker.y, -tracker.x))
        yaw_error = (target_yaw - tracker.yaw + 180) % 360 - 180

        # If yaw error is large, rotate to face origin; otherwise fly
        # forward with a gentle yaw correction.
        if abs(yaw_error) > RTH_YAW_THRESHOLD_DEG:
            yv = RTH_CRUISE_SPEED if yaw_error > 0 else -RTH_CRUISE_SPEED
            fb = 0
            lr = 0
        else:
            yv = int(0.3 * yaw_error)  # gentle correction while moving
            fb = RTH_CRUISE_SPEED
            lr = 0

        me.send_rc_control(lr, fb, 0, yv)
        time.sleep(0.05)

        # Keep the tracker in sync with the movement we just commanded
        tracker.update(lr, fb, 0, yv, 0.05)

    # Stop horizontal movement
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(0.1)
    print(f"[RTH] Phase 2 complete — dist={tracker.distance_to_origin:.0f}cm")

    # ------------------------------------------------------------------
    # PHASE 3 — Descend & land
    # ------------------------------------------------------------------
    height = get_current_height_cm(me)
    print(f"[RTH] Phase 3 — descending from {height:.0f} cm")

    if height > LANDING_APPROACH_HEIGHT_CM:
        t_start = time.perf_counter()
        while True:
            cv2.waitKey(1)  # drain OpenCV events so the window stays responsive
            if _abort_pressed():
                print("EMERGENCY LAND triggered during return-home")
                from control.command_dispatcher import safe_land

                safe_land(me, state)  # guarded: no-op when already grounded
                state.is_flying = False
                from control.command_dispatcher import clear_gesture_latch

                clear_gesture_latch(state)
                return  # exit return_home() immediately, skip remaining phases
            height = get_current_height_cm(me)
            if height <= LANDING_APPROACH_HEIGHT_CM:
                break
            if time.perf_counter() - t_start > MAX_DESCEND_TIME_SEC:
                print(
                    f"[RTH] WARNING: descend timeout ({MAX_DESCEND_TIME_SEC}s), "
                    f"height={height:.0f}cm — landing anyway"
                )
                break
            me.send_rc_control(0, 0, -RTH_DOWN_SPEED, 0)
            time.sleep(0.05)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)

    print("[RTH] Phase 3 complete — handing off to safe_land")
    # Lazy import to avoid circular dependency at module-load time
    from control.command_dispatcher import safe_land as _safe_land
    from control.command_dispatcher import clear_gesture_latch as _clear_latch

    _safe_land(me, state)
    # safe_land bypasses execute_command's land branch (the normal place
    # that clears the latch), so clear it here too — otherwise a gesture
    # latched right before RTH could replay on the next takeoff.
    _clear_latch(state)
    print("[RTH] Return-to-home finished.")
