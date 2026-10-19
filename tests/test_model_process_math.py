"""Geometry / payload tests for the batched inference helpers (no GPU needed)."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.ee_cctv_ai import model_process_ultralytics as mp


def reference_letterbox(img, size):
    """Re-implementation of ultralytics LetterBox(auto=False, center=True) with cv2."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (size - new_unpad[0]) / 2, (size - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, left, top


class LetterboxTests(unittest.TestCase):
    def test_params_match_reference_for_many_shapes(self):
        for h, w in [(1080, 1920), (720, 1280), (360, 640), (480, 640), (1440, 2560), (333, 777), (640, 640)]:
            img = np.zeros((h, w, 3), np.uint8)
            ref, r, left, top = reference_letterbox(img, 640)
            rr, nw, nh, l, t = mp.letterbox_params(h, w, 640)
            self.assertEqual((l, t), (left, top), (h, w))
            self.assertAlmostEqual(rr, r)
            self.assertLessEqual(l + nw, 640)
            self.assertLessEqual(t + nh, 640)
            self.assertEqual(ref.shape[:2], (640, 640))

    def test_box_roundtrip_through_letterbox(self):
        h, w = 1080, 1920
        r, nw, nh, left, top = mp.letterbox_params(h, w, 640)
        box = np.array([[400.0, 300.0, 900.0, 700.0, 0.9, 1.0]], np.float32)
        net = box.copy()
        net[:, [0, 2]] = box[:, [0, 2]] * r + left
        net[:, [1, 3]] = box[:, [1, 3]] * r + top
        back = mp.map_boxes(net, (r, left, top), (h, w), (0, 0), (h, w))
        np.testing.assert_allclose(back[:, :4], box[:, :4], atol=1e-3)
        self.assertEqual(back[0, 4], np.float32(0.9))

    def test_roi_offset_and_clipping(self):
        # crop 200x100 located at col1=50,row1=20 inside a 300x200 frame
        net = np.array([[-5.0, -5.0, 250.0, 150.0, 0.8, 0.0]], np.float32)  # spills outside crop
        out = mp.map_boxes(net, None, (100, 200), (50, 20), (200, 300))
        np.testing.assert_allclose(out[0, :4], [50.0, 20.0, 250.0, 120.0])

    def test_map_boxes_empty(self):
        self.assertEqual(mp.map_boxes(np.zeros((0, 6), np.float32), None, (10, 10), (0, 0), (10, 10)).shape, (0, 6))

    def test_preprocess_roi_is_a_view(self):
        img = np.zeros((200, 300, 3), np.uint8)
        cfg = {"roi": {"row1": 20, "row2": 120, "col1": 50, "col2": 250}}
        with patch.object(mp, "USE_ROI_CROP", True):
            crop, off = mp.cv2_preprocess_fast(img, cfg)
        self.assertEqual(crop.shape[:2], (100, 200))
        self.assertEqual(off, (50, 20))
        self.assertTrue(np.shares_memory(crop, img))
        with patch.object(mp, "USE_ROI_CROP", True):
            crop2, off2 = mp.cv2_preprocess_fast(img, {"roi": {"row1": None, "col1": None, "row2": None, "col2": None}})
        self.assertEqual(off2, (0, 0))
        self.assertEqual(crop2.shape, img.shape)

    def test_payload_and_draw_does_not_modify_original(self):
        dets = np.array([[10, 10, 50, 50, 0.876, 1], [60, 60, 90, 90, 0.5, 0]], np.float32)
        payload = mp.build_payload(dets, {"machineCount": 2})
        self.assertEqual(payload["machineCount"], 2)
        self.assertEqual(payload["detection"][0], {"class": 1, "conf": 0.88, "box": (10, 10, 50, 50)})
        img = np.zeros((100, 100, 3), np.uint8)
        out = mp.draw_detections(img, payload["detection"])
        self.assertEqual(int(img.sum()), 0)       # original untouched
        self.assertGreater(int(out.sum()), 0)     # copy has boxes
        self.assertEqual(mp.build_payload(mp._EMPTY, None)["machineCount"], 0)


if __name__ == "__main__":
    unittest.main()
