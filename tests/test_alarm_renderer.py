import unittest
from unittest.mock import patch

import numpy as np

from src.ee_cctv_ai.alarm_renderer import _color_for, draw_detections

class AlarmRendererTests(unittest.TestCase):
    def setUp(self):
        self.detection = {
            "class": 0, "class_name": "person", "model_id": "model_1",
            "conf": 0.92, "box": (5, 5, 40, 40),
        }

    def test_renderer_copies_frame_and_supports_labels_disabled(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        rendered = draw_detections(image, [self.detection],
                                   {"show_label": False, "show_confidence": False})
        self.assertEqual(rendered.shape, image.shape)
        self.assertIsNot(rendered, image)
        self.assertFalse(np.any(image))
        self.assertTrue(np.any(rendered))

    def test_auto_color_is_stable(self):
        self.assertEqual(_color_for(self.detection, {"color_mode": "auto"}),
                         _color_for(self.detection, {"color_mode": "auto"}))

    def test_custom_class_color_uses_bgr_tuple(self):
        color = _color_for(self.detection, {
            "color_mode": "custom", "class_colors": {"person": [0, 0, 255]},
        })
        self.assertEqual(color, (0, 0, 255))

    def test_custom_mode_uses_auto_for_unmapped_classes(self):
        color = _color_for(self.detection, {
            "color_mode": "custom", "class_colors": {"machine": [0, 0, 255]},
        })
        self.assertEqual(color, _color_for(self.detection, {"color_mode": "auto"}))

    def test_model_mode_uses_one_explicit_color_for_all_classes(self):
        cfg = {"color_mode": "model", "model_color": [255, 128, 0]}
        self.assertEqual(_color_for(self.detection, cfg), (255, 128, 0))
        other = dict(self.detection, class_name="obstacle")
        self.assertEqual(_color_for(other, cfg), (255, 128, 0))

    def test_model_mode_generates_one_stable_color_per_model(self):
        cfg = {"color_mode": "model"}
        first = _color_for(self.detection, cfg)
        other = dict(self.detection, class_name="obstacle")
        self.assertEqual(first, _color_for(other, cfg))

    def test_invalid_custom_color_falls_back_to_auto(self):
        color = _color_for(self.detection, {
            "color_mode": "custom", "class_colors": {"person": "red"},
        })
        self.assertEqual(color, _color_for(self.detection, {"color_mode": "auto"}))

    def test_renderer_uses_per_model_color_settings(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cfg = {"models": {"model_1": {
            "color_mode": "custom", "class_colors": {"person": [0, 0, 255]},
        }}}
        with patch("src.ee_cctv_ai.alarm_renderer.cv2.rectangle") as rectangle:
            draw_detections(image, [self.detection], cfg)
        self.assertEqual(rectangle.call_args.args[3], (0, 0, 255))

if __name__ == "__main__":
    unittest.main()
