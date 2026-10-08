"""Crop geometry and landmark mapping for the gesture classifier.

The key property: landmarks found on a mirrored hand crop must map to the
same numbers the training script produced on the whole mirrored frame.

Run: uv run python -m unittest discover -s tests
"""

import unittest
from types import SimpleNamespace

import numpy as np

from config import GESTURE_CROP_SIZE
from core.gesture_worker import hand_crop_from_raw, square_box_in_frame
from perception.gesture_detector import _landmark_features

RAW_W, RAW_H = 960, 720


def as_landmarks(points):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y) for x, y in points])


class SquareBoxTest(unittest.TestCase):
    def test_box_inside_frame_stays_centred(self):
        self.assertEqual(square_box_in_frame(500, 300, 200, RAW_W, RAW_H), (400, 200, 600, 400))

    def test_box_at_edge_is_shifted_not_shrunk(self):
        x1, y1, x2, y2 = square_box_in_frame(950, 10, 200, RAW_W, RAW_H)
        self.assertEqual((x2 - x1, y2 - y1), (200, 200))
        self.assertEqual((x2, y1), (RAW_W, 0))

    def test_box_larger_than_frame_is_capped(self):
        x1, y1, x2, y2 = square_box_in_frame(480, 360, 5000, RAW_W, RAW_H)
        self.assertEqual((x1, y1, x2, y2), (120, 0, 840, 720))


class HandCropFromRawTest(unittest.TestCase):
    def test_display_bbox_maps_to_square_raw_crop(self):
        raw = np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8)
        crop, box, size = hand_crop_from_raw(raw, (100, 50, 160, 110), (360, 240))
        x1, y1, x2, y2 = box
        self.assertEqual(crop.shape, (GESTURE_CROP_SIZE, GESTURE_CROP_SIZE, 3))
        self.assertEqual(x2 - x1, y2 - y1)
        self.assertEqual(size, (RAW_W, RAW_H))
        # Centre preserved: display (130, 80) * (960/360, 720/240) = (346.7, 240)
        self.assertAlmostEqual((x1 + x2) / 2, 346, delta=2)
        self.assertAlmostEqual((y1 + y2) / 2, 240, delta=2)


class LandmarkMappingTest(unittest.TestCase):
    def test_crop_landmarks_match_full_frame_training_space(self):
        # Arrange: hand points in raw (unmirrored) pixels, and a crop around them.
        rng = np.random.default_rng(0)
        pts_px = np.column_stack([rng.uniform(420, 520, 21), rng.uniform(150, 260, 21)])
        box = (380, 120, 580, 320)
        x1, y1, x2, y2 = box
        side_w, side_h = x2 - x1, y2 - y1
        # What MediaPipe reports on the MIRRORED crop.
        crop_pts = [(1.0 - (px - x1) / side_w, (py - y1) / side_h) for px, py in pts_px]
        # What the training script recorded on the MIRRORED full frame.
        expected = np.column_stack([1.0 - pts_px[:, 0] / RAW_W, pts_px[:, 1] / RAW_H])

        # Act
        features = _landmark_features(as_landmarks(crop_pts), box, (RAW_W, RAW_H))

        # Assert
        self.assertEqual(features.shape, (1, 42))
        np.testing.assert_allclose(features.reshape(21, 2), expected, atol=1e-5)

    def test_without_geometry_falls_back_to_crop_coordinates(self):
        features = _landmark_features(as_landmarks([(0.25, 0.75)] * 21), None, None)
        np.testing.assert_allclose(features.reshape(21, 2)[0], [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
