"""Gesture debouncer — turns noisy per-frame classifier output into commands.

Replaces the per-gesture counter chain that used to live in main.py and
test_script/demo_webcam.py. That chain had three problems:

  * stop / flip needed 20 consecutive hits (1.5-2 s at the classifier's
    real rate), so the land gesture felt unresponsive;
  * the stop / flip / up / down counters were never reset by a different
    gesture, so stray single-frame detections piled up over minutes and
    eventually fired a spurious land or face-mode toggle;
  * movement gestures were emitted on hits 5..10 and then had to re-confirm
    from zero, so a held gesture stuttered.

Rules here:
  * A gesture is confirmed after N consecutive classifier results with the
    same label (N per gesture, see config.GESTURE_CONFIRM_*).
  * An unclassified result (None: hand visible, confidence too low) neither
    breaks nor extends a streak; a DIFFERENT label restarts it.
  * Streaks expire when no result arrives for GESTURE_STREAK_TIMEOUT_S, and
    reset() clears them when the hand is lowered.
  * Continuous gestures keep emitting on every hit once confirmed.
  * One-shot gestures fire once per hold and then honour a cooldown, so
    holding "flip" does not toggle face mode back and forth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import (
    GESTURE_CONFIRM_MOVE,
    GESTURE_CONFIRM_STOP,
    GESTURE_CONFIRM_TOGGLE,
    GESTURE_ONE_SHOT_COOLDOWN_S,
    GESTURE_STREAK_TIMEOUT_S,
)

NO_GESTURE = "None"

CONTINUOUS_GESTURES = frozenset({
    "up", "down", "left", "right",
    "turn left", "turn right",
    "move farward", "move backward",
})
ONE_SHOT_GESTURES = frozenset({"stop", "flip"})
KNOWN_GESTURES = CONTINUOUS_GESTURES | ONE_SHOT_GESTURES

_CONFIRM_COUNT = {
    "stop": GESTURE_CONFIRM_STOP,
    "flip": GESTURE_CONFIRM_TOGGLE,
}


@dataclass
class _Streak:
    label: Optional[str] = None
    count: int = 0
    last_t: float = 0.0
    fired: bool = False


class GestureDebouncer:
    """Feed one classifier result per update(); returns the gesture to act on."""

    def __init__(self) -> None:
        self._streak = _Streak()
        self._cooldown_until = 0.0

    def reset(self) -> None:
        """Forget the current streak (call when the hand is lowered)."""
        self._streak = _Streak()

    def update(self, gesture: Optional[str], now: float) -> str:
        """Process one NEW classifier result.

        Args:
            gesture: label from PredictClass, or None when nothing was
                classified confidently.
            now: monotonic timestamp in seconds.

        Returns:
            The confirmed gesture to send this tick, or NO_GESTURE.
        """
        streak = self._streak
        if streak.label is not None and now - streak.last_t > GESTURE_STREAK_TIMEOUT_S:
            self.reset()
            streak = self._streak

        if gesture is None or gesture not in KNOWN_GESTURES:
            return NO_GESTURE

        if gesture != streak.label:
            self._streak = streak = _Streak(label=gesture)
        streak.count += 1
        streak.last_t = now

        if streak.count < _CONFIRM_COUNT.get(gesture, GESTURE_CONFIRM_MOVE):
            return NO_GESTURE

        if gesture in CONTINUOUS_GESTURES:
            return gesture

        # One-shot: once per hold, and not again until the cooldown passes.
        if streak.fired or now < self._cooldown_until:
            return NO_GESTURE
        streak.fired = True
        self._cooldown_until = now + GESTURE_ONE_SHOT_COOLDOWN_S
        return gesture
