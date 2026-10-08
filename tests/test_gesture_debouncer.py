"""Unit tests for control.gesture_debouncer (no models, no drone).

Run: uv run python -m unittest discover -s tests
"""

import unittest

from config import (
    GESTURE_CONFIRM_MOVE,
    GESTURE_CONFIRM_STOP,
    GESTURE_CONFIRM_TOGGLE,
    GESTURE_ONE_SHOT_COOLDOWN_S,
    GESTURE_STREAK_TIMEOUT_S,
)
from control.gesture_debouncer import NO_GESTURE, GestureDebouncer

STEP_S = 0.05  # classifier result spacing (~20 Hz)


def feed(debouncer, labels, start=0.0):
    """Feed labels at STEP_S spacing; return the outputs."""
    return [debouncer.update(label, start + i * STEP_S) for i, label in enumerate(labels)]


class GestureDebouncerTest(unittest.TestCase):
    def test_move_confirms_after_n_hits_and_keeps_emitting(self):
        # Arrange
        debouncer = GestureDebouncer()
        # Act
        out = feed(debouncer, ["up"] * (GESTURE_CONFIRM_MOVE + 3))
        # Assert
        self.assertEqual(out[:GESTURE_CONFIRM_MOVE - 1], [NO_GESTURE] * (GESTURE_CONFIRM_MOVE - 1))
        self.assertTrue(all(g == "up" for g in out[GESTURE_CONFIRM_MOVE - 1:]))

    def test_different_label_restarts_streak(self):
        debouncer = GestureDebouncer()
        out = feed(debouncer, ["up"] * (GESTURE_CONFIRM_MOVE - 1) + ["down"])
        self.assertEqual(out[-1], NO_GESTURE)

    def test_unclassified_frame_does_not_break_streak(self):
        debouncer = GestureDebouncer()
        labels = ["left"] * (GESTURE_CONFIRM_MOVE - 1) + [None, "left"]
        self.assertEqual(feed(debouncer, labels)[-1], "left")

    def test_stray_stop_hits_do_not_accumulate(self):
        # Old counter chain never reset "stop", so scattered hits eventually landed.
        debouncer = GestureDebouncer()
        labels = (["stop"] + ["up"] * 3) * (GESTURE_CONFIRM_STOP * 3)
        self.assertNotIn("stop", feed(debouncer, labels))

    def test_stop_fires_once_per_hold(self):
        debouncer = GestureDebouncer()
        out = feed(debouncer, ["stop"] * (GESTURE_CONFIRM_STOP + 10))
        self.assertEqual(out.count("stop"), 1)
        self.assertEqual(out.index("stop"), GESTURE_CONFIRM_STOP - 1)

    def test_flip_respects_cooldown_after_reset(self):
        debouncer = GestureDebouncer()
        first = feed(debouncer, ["flip"] * GESTURE_CONFIRM_TOGGLE)
        debouncer.reset()
        again = feed(debouncer, ["flip"] * GESTURE_CONFIRM_TOGGLE, start=1.0)
        later = feed(debouncer, ["flip"] * GESTURE_CONFIRM_TOGGLE,
                     start=1.0 + GESTURE_ONE_SHOT_COOLDOWN_S + 1.0)
        self.assertEqual(first.count("flip"), 1)
        self.assertEqual(again.count("flip"), 0)
        self.assertEqual(later.count("flip"), 1)

    def test_streak_expires_after_gap(self):
        debouncer = GestureDebouncer()
        feed(debouncer, ["up"] * (GESTURE_CONFIRM_MOVE - 1))
        resumed = debouncer.update("up", GESTURE_STREAK_TIMEOUT_S + 1.0)
        self.assertEqual(resumed, NO_GESTURE)

    def test_unknown_label_is_ignored(self):
        debouncer = GestureDebouncer()
        self.assertEqual(feed(debouncer, ["Q"] * 10), [NO_GESTURE] * 10)


if __name__ == "__main__":
    unittest.main()
