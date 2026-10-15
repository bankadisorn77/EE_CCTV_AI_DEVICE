"""Tests for health-stat formatting without Jetson hardware."""
import unittest
from unittest.mock import patch, Mock

from src.ee_cctv_ai.status_service import get_jetson_stats


class StatusServiceTests(unittest.TestCase):
    @patch("src.ee_cctv_ai.status_service.psutil.virtual_memory")
    def test_stats_include_memory_status_and_device_state(self, memory_mock):
        memory_mock.return_value = Mock(available=2 * 1024 * 1024, used=3 * 1024 * 1024)
        shared = Mock()
        shared.get_active_device.return_value = 1
        result = get_jetson_stats("active", shared)
        self.assertEqual(result["runningStatus"], "active")
        self.assertEqual(result["memoryFree"], 2)
        self.assertEqual(result["memoryUsed"], 3)
        self.assertEqual(result["activeDevice"], 1)
        self.assertEqual(result["aiVersion"], "v1.12.3")


if __name__ == "__main__":
    unittest.main()
