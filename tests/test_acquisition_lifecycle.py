"""Hardware-independent tests for Acquisition's lifecycle."""
import unittest
from unittest.mock import patch

from src.ee_cctv_ai.acquistion import Acquisition


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *args):
        pass

    def release(self):
        self.released = True


class AcquisitionTests(unittest.TestCase):
    def test_constructor_does_not_open_camera(self):
        with patch("src.ee_cctv_ai.acquistion.cv2.VideoCapture") as video_capture:
            camera = Acquisition("rtsp://example.invalid/stream", key="cam1")
            video_capture.assert_not_called()
            self.assertFalse(camera.connected)
            self.assertEqual(camera.frame_id, 0)

    def test_failed_open_releases_capture(self):
        fake = FakeCapture(opened=False)
        with patch("src.ee_cctv_ai.acquistion.cv2.VideoCapture", return_value=fake):
            camera = Acquisition("rtsp://example.invalid/stream", key="cam1")
            with self.assertRaises(RuntimeError):
                camera.initial()
            self.assertTrue(fake.released)


if __name__ == "__main__":
    unittest.main()
