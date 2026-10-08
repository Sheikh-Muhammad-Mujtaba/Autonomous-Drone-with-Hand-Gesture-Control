"""VoiceCommandLatch: one-shot voice actions must apply exactly once.

Regression for the flight log where a voice "take off" was replayed every
tick ("Already flying - ignoring takeoff request." x hundreds).

Run: uv run python -m unittest discover -s tests -t .
"""

import unittest

from control.command_dispatcher import VOICE_MOVE_MAX_S, VoiceCommandLatch

TICK_S = 1.0 / 30.0


def run_ticks(latch, start, count):
    return [latch.tick(start + i * TICK_S) for i in range(count)]


class VoiceCommandLatchTest(unittest.TestCase):
    def test_takeoff_is_applied_on_one_tick_only(self):
        # Arrange
        latch = VoiceCommandLatch()
        latch.push({"action": "takeoff"}, 0.0)
        # Act
        out = run_ticks(latch, 0.0, 60)
        # Assert
        self.assertEqual(out[0].actions, ("takeoff",))
        self.assertTrue(all(cmd is None for cmd in out[1:]))

    def test_land_is_not_replayed_after_a_new_takeoff(self):
        latch = VoiceCommandLatch()
        latch.push({"action": "land"}, 0.0)
        run_ticks(latch, 0.0, 1)
        self.assertIsNone(latch.tick(5.0))  # keyboard takeoff later: no stale land

    def test_move_is_held_for_its_duration_then_released(self):
        latch = VoiceCommandLatch()
        latch.push({"action": "move", "direction": "up", "distance_cm": 50}, 0.0)
        first = latch.tick(0.0)
        self.assertGreater(first.ud, 0)
        self.assertIsNone(latch.tick(first.duration_s + 0.01))

    def test_move_without_distance_is_capped(self):
        latch = VoiceCommandLatch()
        latch.push({"action": "move", "direction": "left"}, 0.0)
        self.assertIsNotNone(latch.tick(VOICE_MOVE_MAX_S - 0.1))
        self.assertIsNone(latch.tick(VOICE_MOVE_MAX_S + 0.1))

    def test_hover_cancels_held_move_and_applies_once(self):
        latch = VoiceCommandLatch()
        latch.push({"action": "move", "direction": "forward", "distance_cm": 200}, 0.0)
        latch.tick(0.0)
        latch.push({"action": "hover"}, 0.5)
        self.assertEqual(latch.tick(0.5).actions, ("hover",))
        self.assertIsNone(latch.tick(0.6))

    def test_describe_scene_holds_nothing(self):
        latch = VoiceCommandLatch()
        latch.push({"action": "describe_scene"}, 0.0)
        self.assertIsNone(latch.tick(0.0))


if __name__ == "__main__":
    unittest.main()
