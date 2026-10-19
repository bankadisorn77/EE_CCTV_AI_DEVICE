import unittest

import numpy as np

from src.ee_cctv_ai.alarm_renderer import draw_detections


class AlarmRendererTests(unittest.TestCase):
    def test_renderer_copies_frame_and_supports_labels_disabled(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        rendered = draw_detections(image, [
            {"class": 0, "class_name": "person", "conf": 0.92, "box": (5, 5, 40, 40)}
        ], {"show_label": False, "show_confidence": False})
        self.assertEqual(rendered.shape, image.shape)
        self.assertIsNot(rendered, image)
        self.assertFalse(np.any(image))
        self.assertTrue(np.any(rendered))


if __name__ == "__main__":
    unittest.main()
