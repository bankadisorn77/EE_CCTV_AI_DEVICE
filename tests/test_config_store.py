"""Tests for atomic config storage."""
import json
import tempfile
import unittest
from pathlib import Path

from src.ee_cctv_ai.config_store import load_config, save_config


class ConfigStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = {"cameras": [{"name": "cam1"}], "confidence_threshold": 0.7}
            save_config(value, path)
            self.assertEqual(load_config(path), value)

    def test_malformed_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{not-json", encoding="utf-8")
            value = load_config(path)
            self.assertEqual(value["cameras"], [])
            self.assertEqual(value["confidence_threshold"], 0.5)

    def test_save_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config({"first": True}, path)
            save_config({"second": True}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"second": True})


if __name__ == "__main__":
    unittest.main()
