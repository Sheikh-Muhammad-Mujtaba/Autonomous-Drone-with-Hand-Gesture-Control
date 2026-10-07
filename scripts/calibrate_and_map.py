"""
Standalone calibration + live-map diagnostic for DJI Tello.

Connects to the drone, runs three sequential tests, and prints the
measured constants to paste into control/position_tracker.py.

Safety:
  - Battery check before takeoff (same MIN_TAKEOFF_BATTERY = 20).
  - try/finally guarantees landing on any exception.
  - Keyboard override ('x' / 'q') during the live square-flight test.
  - Keyboard override ('x') also works during return-to-home via the
    same pattern already wired in control/position_tracker.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly (python scripts/calibrate_and_map.py) — put the
# project root on sys.path so package imports (ui, control) resolve.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import time
from typing import Optional

import cv2
import djitellopy
from ui import keypress as kp

from control.position_map import PositionMapRenderer
from control.position_tracker import PositionTracker, RC_TO_CM_S, RC_TO_DEG_S

# ---------------------------------------------------------------------------
# Constants (same as control/command_dispatcher.py + position_tracker.py)
# ---------------------------------------------------------------------------
SPEED1 = 30
MIN_TAKEOFF_BATTERY = 20
DURATION_S = 2.0  # per-leg duration for calibration moves
TEST_C_LEG_DISTANCE_CM = 150  # target distance per forward leg in Test C
TEST_C_TURN_DEGREES = 90      # target turn angle per corner in Test C
MAX_LAND_ATTEMPTS = 3         # safe_land retries before giving up


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def connect_tello() -> djitellopy.Tello:
    """Connect, start video stream, and return ready Tello object."""
    me = djitellopy.Tello()
    me.connect()
    me.streamon()
    battery = me.get_battery()
    print(f"[CALIB] Connected.  Battery: {battery}%  (min: {MIN_TAKEOFF_BATTERY}%)")
    if battery < MIN_TAKEOFF_BATTERY:
        raise RuntimeError(
            f"Battery too low ({battery}% < {MIN_TAKEOFF_BATTERY}%) — "
            "charge or use a fresh battery."
        )
    time.sleep(2)  # let the video stream settle
    return me


# ---------------------------------------------------------------------------
# Safe takeoff / land helpers (no DispatcherState dependency)
# ---------------------------------------------------------------------------

def safe_takeoff(me: djitellopy.Tello) -> None:
    """Take off with a battery check (mirrors command_dispatcher guardian)."""
    bat = me.get_battery()
    if bat < MIN_TAKEOFF_BATTERY:
        print(f"[CALIB] Battery too low ({bat}%) — cannot take off.")
        return
    me.takeoff()
    print("[CALIB] Takeoff OK.")


def _drone_on_ground(me: djitellopy.Tello) -> bool:
    """True if the drone's height sensor reports it is on the ground."""
    try:
        h = me.get_height()
        return h is not None and h <= 1  # ≤1 dm (≈10 cm) = grounded
    except Exception:
        return False


def safe_land(me: djitellopy.Tello) -> None:
    """Land and CONFIRM the drone actually touched down.

    djitellopy's land() masks 'error' responses as 'ok', so the raw
    response is inspected via send_command_with_return() instead.
    Retries up to MAX_LAND_ATTEMPTS times on 'error' with a short
    pause, swallows a second Ctrl+C so an interrupt can't abort the
    landing, and warns loudly if landing could not be confirmed so the
    user checks the drone manually.
    """
    if _drone_on_ground(me):
        print("[CALIB] Already on the ground — no landing needed.")
        return

    for attempt in range(1, MAX_LAND_ATTEMPTS + 1):
        try:
            resp = me.send_command_with_return("land")
            if resp and "ok" in str(resp).lower():
                print("[CALIB] Landed.")
                return
            print(f"[CALIB] Land response '{resp}' (attempt {attempt}/"
                  f"{MAX_LAND_ATTEMPTS}) — retrying…")
            time.sleep(0.5)
            # The response may lag a touchdown that already happened
            # (e.g. auto-landed a moment ago) — the height sensor
            # settles quickly.
            if _drone_on_ground(me):
                print("[CALIB] Landed (height sensor confirms touchdown).")
                return
        except KeyboardInterrupt:
            # A second Ctrl+C must not abort a landing in progress.
            print("[CALIB] Ctrl+C ignored during landing — continuing…")
            time.sleep(0.3)
            continue
        except Exception as exc:
            print(f"[CALIB] Land command failed (attempt {attempt}/"
                  f"{MAX_LAND_ATTEMPTS}): {exc}")
            time.sleep(0.5)
            continue

    print("\n[CALIB] WARNING: landing could NOT be confirmed after "
          f"{MAX_LAND_ATTEMPTS} attempts.")
    print("[CALIB] Check the drone manually — if it is still airborne,")
    print("[CALIB] cut the motors with the battery button and inspect.")


# ---------------------------------------------------------------------------
# Test A — Forward distance calibration
# ---------------------------------------------------------------------------

def test_forward_calibration(me: djitellopy.Tello) -> Optional[float]:
    """Measure rc→cm/s by flying forward at SPEED1 for DURATION_S.

    Returns the measured constant (cm/s per RC unit), or None if the
    user skips / enters an invalid measurement.
    """
    print("\n" + "=" * 58)
    print("  TEST A — Forward distance calibration")
    print("=" * 58)
    print("")
    print(" 0. IMPORTANT: the Tello flies toward ITS OWN camera nose —")
    print("    align the camera with the direction you want 'forward'.")
    print(" 1. Place the drone at a known start line on the floor.")
    print(" 2. I'll fly forward for exactly 2 seconds at speed 30.")
    print(" 3. Mark where the drone ends up and measure the distance.")
    print("")
    print(f"    Current heading (get_yaw): {me.get_yaw()}°")
    print("")
    print("→ Mark your current position now.")
    input("   Press ENTER when ready to fly… ".rjust(55))

    print("\n[TEST A] Flying forward…")
    me.send_rc_control(0, SPEED1, 0, 0)
    time.sleep(DURATION_S)
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(0.3)

    # Re-ask until a valid number, or 'skip' to keep the previous value
    while True:
        distance_str = input("   Measure the distance travelled (cm) and enter it: ")
        try:
            distance_cm = float(distance_str)
            break
        except ValueError:
            if distance_str.strip() == "skip":
                print("[TEST A] Skipped — keeping previous RC_TO_CM_S.")
                return None
            print("[TEST A] Please enter a number, or 'skip' to keep the previous value.")

    measured = distance_cm / (SPEED1 * DURATION_S)
    print(f"\n   Measured RC_TO_CM_S = {distance_cm:.0f} / ({SPEED1} × {DURATION_S})")
    print(f"        → RC_TO_CM_S = {measured:.4f}")
    return measured


# ---------------------------------------------------------------------------
# Test B — Yaw calibration + sign verification
# ---------------------------------------------------------------------------

def test_yaw_calibration(me: djitellopy.Tello) -> float:
    """Measure rc→deg/s by yawing at SPEED1 for DURATION_S.

    The SIGN of the result is critical — it tells us whether the sign
    convention in control/position_tracker.py's RC_TO_DEG_S matches the
    real Tello sensor.  Prints the exact constant to paste.
    """
    print("\n" + "=" * 58)
    print("  TEST B — Yaw calibration & sign verification")
    print("=" * 58)
    print("")
    print(" I'll yaw the drone for exactly 2 seconds at yv=30.")
    print(" The onboard yaw sensor will tell us the real rotation.")
    print("")
    print("→ Ensure the drone has enough space to yaw safely.")
    input("   Press ENTER to start the yaw test… ".rjust(55))

    yaw_before = me.get_yaw()
    print(f"\n   Yaw before: {yaw_before}°")

    me.send_rc_control(0, 0, 0, SPEED1)  # positive yv
    time.sleep(DURATION_S)
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)  # let the IMU reading settle

    yaw_after = me.get_yaw()
    print(f"   Yaw after:  {yaw_after}°")

    raw_delta = yaw_after - yaw_before
    # Normalise to [-180, 180] (handle 0/360 wraparound)
    normalised = (raw_delta + 180) % 360 - 180

    measured = normalised / (SPEED1 * DURATION_S)

    direction = "CLOCKWISE" if normalised < 0 else "COUNTER-CLOCKWISE"
    sign_str = f"{measured:.4f}"  # sign is already from normalised

    print(f"\n   Raw delta:   {raw_delta:+.1f}°  (signed difference)")
    print(f"   Normalised:  {normalised:+.1f}°  (wrapped to [-180, 180])")
    print("")
    print(f"   Positive yv = {SPEED1} produced a yaw change of "
          f"{normalised:.1f}°")
    print(f"   → The drone rotated {direction}.")
    print("")
    print(f"   Measured RC_TO_DEG_S = {normalised:.1f} / "
          f"({SPEED1} × {DURATION_S})")
    print(f"        → RC_TO_DEG_S = {sign_str}")
    print("")
    if normalised >= 0:
        print("   VERDICT: raw_delta is POSITIVE / ZERO — positive yv")
        print("            matches increasing yaw.  The sign in")
        print("            control/position_tracker.py is already correct.")
    else:
        print("   VERDICT: raw_delta is NEGATIVE — positive yv produces")
        print("            decreasing yaw.  RC_TO_DEG_S in control/")
        print("            position_tracker.py currently uses +0.7333 (wrong sign);")
        print("            the measured value above has the correct sign.")
    print("")
    return measured


# ---------------------------------------------------------------------------
# Test C — Live map sanity check (square pattern)
# ---------------------------------------------------------------------------

def test_live_map(me: djitellopy.Tello) -> None:
    """Fly a square pattern while the map renderer shows dead-reckoning vs sensor yaw.

    Press 'x' during the test for an emergency land, or 'q' to cut the
    pattern short and land normally.
    """
    # Derive forward-leg and turn-leg durations from the MEASURED scale
    # constants, so distances and angles stay accurate after re-calibration.
    forward_leg_duration_s = TEST_C_LEG_DISTANCE_CM / (SPEED1 * RC_TO_CM_S)
    turn_leg_duration_s = TEST_C_TURN_DEGREES / (SPEED1 * RC_TO_DEG_S)

    print("\n" + "=" * 58)
    print("  TEST C — Live map sanity check")
    print("=" * 58)
    print("")
    print(f" Each forward leg: {forward_leg_duration_s:.1f}s at speed {SPEED1}")
    print(f" (targeting ~{TEST_C_LEG_DISTANCE_CM}cm)")
    print(f" Each turn: {turn_leg_duration_s:.1f}s at yv={SPEED1}")
    print(f" (targeting ~{TEST_C_TURN_DEGREES}°)")
    print("")
    print(" I'll fly a square-ish pattern.  Watch the map window:")
    print("  • GREEN arrows = dead-reckoning yaw (tracker)")
    print("  • RED   arrows = onboard sensor yaw (me.get_yaw())")
    print(" If they diverge after a turn, the yaw sign is wrong.")
    print("")
    print(" Controls during test:")
    print("   q  — land now (cut short)")
    print("   x  — emergency land")
    print("")
    input("   Press ENTER to start the square flight… ".rjust(55))

    tracker = PositionTracker()
    renderer = PositionMapRenderer()
    window_name = "Calibration Map"
    cv2.namedWindow(window_name)

    # Square pattern:  4 × (forward ~4.4 s  →  yaw ~4.1 s  →  stop)
    # Forward and turn durations are computed from the measured
    # constants so each leg covers ~150 cm and each corner ~90°.
    legs: list[tuple[int, int, int, int, float]] = [
        # (lr, fb, ud, yv, duration_s)
        (0,  SPEED1,  0,  0, forward_leg_duration_s),  # forward
        (0,  0,       0,  0, 0.3),   # pause
        (0,  0,       0,  SPEED1, turn_leg_duration_s),   # yaw right
        (0,  0,       0,  0, 0.3),   # pause
        (0,  SPEED1,  0,  0, forward_leg_duration_s),  # forward
        (0,  0,       0,  0, 0.3),   # pause
        (0,  0,       0,  SPEED1, turn_leg_duration_s),   # yaw right
        (0,  0,       0,  0, 0.3),   # pause
        (0,  SPEED1,  0,  0, forward_leg_duration_s),  # forward
        (0,  0,       0,  0, 0.3),   # pause
        (0,  0,       0,  SPEED1, turn_leg_duration_s),   # yaw right
        (0,  0,       0,  0, 0.3),   # pause
        (0,  SPEED1,  0,  0, forward_leg_duration_s),  # forward
        (0,  0,       0,  0, 0.3),   # pause
        (0,  0,       0,  SPEED1, turn_leg_duration_s),   # yaw right (back to start heading)
        (0,  0,       0,  0, 0.3),   # final pause
    ]

    try:
        for lr, fb, ud, yv, dur in legs:
            # ---- Keyboard abort checks ----
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[TEST C] Window closed — landing.")
                break

            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                print("[TEST C] 'q' pressed — landing now.")
                break
            if kp.getKeyPressedOnce("x"):
                print("[TEST C] 'x' — emergency land.")
                safe_land(me)
                return  # exit test_c, skip remaining legs

            # ---- Send RC ----
            me.send_rc_control(lr, fb, ud, yv)

            if dur > 0:
                # Sub-step: update tracker + render at ~20 Hz so the map
                # animates smoothly during a leg
                dt_leg = 0.05
                elapsed = 0.0
                while elapsed < dur:
                    cv2.waitKey(1)

                    # Periodic abort re-check
                    if kp.getKeyPressedOnce("x"):
                        me.send_rc_control(0, 0, 0, 0)
                        print("[TEST C] 'x' — emergency land mid-leg.")
                        safe_land(me)
                        return

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        me.send_rc_control(0, 0, 0, 0)
                        break

                    # Update dead-reckoning
                    tracker.update(lr, fb, ud, yv, dt_leg)
                    elapsed += dt_leg

                    # Sample real heading after yaw legs
                    real_yaw: Optional[float] = me.get_yaw() \
                        if fb == 0 and lr == 0 and yv != 0 else None
                    renderer.add_point(tracker.x, tracker.y, tracker.yaw,
                                       real_yaw)

                    map_img = renderer.render()
                    cv2.imshow(window_name, map_img)

                    if key == ord("q"):
                        break
                    time.sleep(dt_leg * 0.8)  # ≈ real-time-ish

                # After a leg, also sample+render for the stationary pause
                me.send_rc_control(0, 0, 0, 0)
                # Get yaw after movement stops (post-yaw settling)
                real_yaw_stop = me.get_yaw() if yv != 0 else None
                tracker.update(lr, fb, ud, yv, 0.0)  # no time elapsed
                renderer.add_point(tracker.x, tracker.y, tracker.yaw,
                                   real_yaw_stop)
                cv2.imshow(window_name, renderer.render())
                cv2.waitKey(int(dur * 1000) if dur <= 0.5 else 300)

                if key == ord("q"):
                    break

        # ---- Square done — show final state ----
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        final_yaw = me.get_yaw()
        renderer.add_point(tracker.x, tracker.y, tracker.yaw, final_yaw)
        final_img = renderer.render()

        # Overlay final stats
        info_lines = [
            f"Tracker: x={tracker.x:.0f}  y={tracker.y:.0f}  yaw={tracker.yaw:.0f}°",
            f"Sensor:  yaw={final_yaw:.0f}°",
            f"Dist from origin: {tracker.distance_to_origin:.0f} cm",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                final_img, line,
                (10, (len(info_lines) - i) * 18 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
            )
        cv2.imshow(window_name, final_img)
        cv2.waitKey(3000)

        print("\n" + "-" * 58)
        print("  TEST C complete — final state:")
        for line in info_lines:
            print(f"   {line}")
        print("")
        print("  Does the drone appear to be near its start position?")
        print("  If not, the dead-reckoning constants need tuning.")

    finally:
        cv2.destroyWindow(window_name)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all three calibration tests in sequence."""
    print("=" * 58)
    print("  Tello Calibration Tool — scripts/calibrate_and_map.py")
    print("=" * 58)
    print("")
    print(" This script connects to the Tello, takes off, and runs three")
    print(" sequential diagnostic tests.  Read each prompt carefully.")
    print(" Target: SPEED1 = 30,  DURATION = 2.0 s / leg")
    print("")
    print(" ORIENTATION: the Tello always flies toward ITS OWN camera")
    print(" nose — point the camera where you want 'forward' to be.")
    print("")
    input("   Press ENTER to connect to the Tello… ".rjust(55))

    kp.init()  # needed for getKeyPressedOnce during emergency checks

    me: Optional[djitellopy.Tello] = None
    rc_to_cm_s = 1.1333  # last measured value; fallback if Test A skipped
    rc_to_deg_s = 0.7333  # last measured value; fallback if Test B skipped

    cleanup_done = False

    def cleanup() -> None:
        """Tear down the connection EXACTLY once, on every exit path.

        Guarded by cleanup_done so a double trigger (an except handler
        plus the finally, or a stray call) can never re-send land/end
        commands to an already-closed connection.
        """
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if me is None:
            return
        safe_land(me)  # confirms touchdown; retries on 'error'
        try:
            me.streamoff()
        except Exception:
            pass
        try:
            # end() would call land() again while djitellopy still
            # thinks the drone is flying — tell it the truth first.
            me.is_flying = False
            me.end()
        except Exception:
            pass
        # __del__ fires at interpreter exit and re-runs end() on the
        # torn-down connection (KeyError/OSError noise).  Shadow the
        # method so the destructor becomes a no-op.
        try:
            me.end = lambda: None  # type: ignore[method-assign]
        except Exception:
            pass

    try:
        me = connect_tello()
        safe_takeoff(me)

        # ---- Test A ----
        input("\n   TEST A — Press ENTER when ready… ".rjust(55))
        measured_a = test_forward_calibration(me)
        if measured_a is not None:
            rc_to_cm_s = measured_a

        # ---- Test B ----
        input("\n   TEST B — Press ENTER when ready… ".rjust(55))
        rc_to_deg_s = test_yaw_calibration(me)

        # ---- Test C ----
        if me.is_flying:
            test_live_map(me)
        else:
            print("\n[TEST C] Not flying — skipping live map test.")
    except KeyboardInterrupt:
        print("\n[CALIB] Interrupted by user.")
    except Exception as e:
        print(f"\n[CALIB] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

    # ---- Final instructions ----
    print("\n" + "=" * 58)
    print("  CALIBRATION COMPLETE — paste these into")
    print("  control/position_tracker.py:")
    print("=" * 58)
    print("")
    # Format with the measured sign preserved
    print(f"  RC_TO_CM_S = {rc_to_cm_s:.4f}")
    print(f"  RC_TO_DEG_S = {rc_to_deg_s:.4f}")
    print("")
    print("  (These replace the RC_TO_CM_S / RC_TO_DEG_S constants.)")
    print("")
    print("  If RC_TO_DEG_S sign is NEGATIVE, also double-check the")
    print("  yaw rotation matrix sign in PositionTracker.update() —")
    print("  the sensor agrees with the tracker, or it doesn't.")
    print("")


if __name__ == "__main__":
    main()
