"""Voice command parsing: regex path and LLM-output validation.

Imports control.voice_model without loading Whisper (load is lazy) and
makes no network calls.

Run: uv run python -m unittest discover -s tests
"""

import unittest

from control.voice_model import (
    MAX_MOVE_DISTANCE_CM,
    regex_parser,
    validate_intent,
)


class RegexParserTest(unittest.TestCase):
    def assertParsed(self, phrase, expected):
        self.assertEqual(regex_parser(phrase), expected, phrase)

    def test_simple_actions(self):
        self.assertParsed("Take off.", {"action": "takeoff"})
        self.assertParsed("land", {"action": "land"})
        self.assertParsed("Stop!", {"action": "hover"})
        self.assertParsed("return home", {"action": "return_home"})
        self.assertParsed("describe the scene", {"action": "describe_scene"})

    def test_metres_are_converted_to_centimetres(self):
        self.assertParsed("move forward two meters",
                          {"action": "move", "direction": "forward", "distance_cm": 200})
        self.assertParsed("go left half a meter",
                          {"action": "move", "direction": "left", "distance_cm": 50})
        self.assertParsed("move up 1.5 m",
                          {"action": "move", "direction": "up", "distance_cm": 150})

    def test_plain_number_is_centimetres(self):
        self.assertParsed("move forward 50",
                          {"action": "move", "direction": "forward", "distance_cm": 50})

    def test_number_words_need_word_boundaries(self):
        # "someone" used to parse as 1 cm, "listen" as 10.
        self.assertParsed("someone go right",
                          {"action": "move", "direction": "right", "distance_cm": 50})

    def test_article_before_number_is_skipped(self):
        self.assertParsed("turn a little left 30",
                          {"action": "rotate", "direction": "ccw", "degrees": 30})

    def test_all_right_is_not_a_direction(self):
        self.assertIsNone(regex_parser("all right"))
        self.assertParsed("alright, take off", {"action": "takeoff"})

    def test_contradictory_takeoff_and_land_hovers(self):
        # Whisper echoed its old vocabulary prompt during a flight test.
        self.assertParsed("drone commands take off, land, hover, stop,", {"action": "hover"})

    def test_compound_number_words(self):
        self.assertParsed("turn left forty five degrees",
                          {"action": "rotate", "direction": "ccw", "degrees": 45})
        self.assertParsed("move up twenty-five",
                          {"action": "move", "direction": "up", "distance_cm": 25})

    def test_turn_around_is_half_turn(self):
        self.assertParsed("turn around", {"action": "rotate", "direction": "cw", "degrees": 180})

    def test_distance_is_clamped(self):
        cmd = regex_parser("move forward 900")
        self.assertEqual(cmd["distance_cm"], MAX_MOVE_DISTANCE_CM)


class ValidateIntentTest(unittest.TestCase):
    def test_accepts_known_shapes(self):
        self.assertEqual(validate_intent({"action": "land"}), {"action": "land"})
        self.assertEqual(
            validate_intent({"action": "rotate", "direction": "ccw", "degrees": "45"}),
            {"action": "rotate", "direction": "ccw", "degrees": 45},
        )

    def test_rejects_unknown_or_malformed(self):
        self.assertIsNone(validate_intent({"action": None}))
        self.assertIsNone(validate_intent({"action": "self_destruct"}))
        self.assertIsNone(validate_intent({"action": "move", "direction": "sideways"}))
        self.assertIsNone(validate_intent(["land"]))

    def test_clamps_and_defaults_numbers(self):
        cmd = validate_intent({"action": "move", "direction": "up", "distance_cm": 10_000})
        self.assertEqual(cmd["distance_cm"], MAX_MOVE_DISTANCE_CM)
        cmd = validate_intent({"action": "move", "direction": "up", "distance_cm": "lots"})
        self.assertEqual(cmd["distance_cm"], 50)


if __name__ == "__main__":
    unittest.main()
