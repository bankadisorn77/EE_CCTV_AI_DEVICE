import unittest

from src.ee_cctv_ai.alarm_rules import AlarmRuleEngine


class AlarmRuleEngineTests(unittest.TestCase):
    def test_class_detected_rules_work_for_arbitrary_model_ids(self):
        engine = AlarmRuleEngine({"models": [
            {"id": "person_v3", "alarm_rules": [
                {"id": "person", "type": "class_detected", "class_name": "person",
                 "cooldown_sec": 3, "alarm_type": "person_detected"}
            ]}
        ]})
        payload = {"person_v3": {"detection": [
            {"class": 0, "class_name": "person", "conf": 0.91, "box": (1, 2, 3, 4)}
        ]}}
        events = engine.evaluate("cam1", payload, now=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["alarm_type"], "person_detected")
        self.assertEqual(events[0]["model_id"], "person_v3")
        self.assertEqual(engine.evaluate("cam1", payload, now=11), [])

    def test_count_below_duration_and_reset(self):
        engine = AlarmRuleEngine({"models": [
            {"id": "machine", "alarm_rules": [
                {"id": "missing", "type": "count_below", "expected_count": 2,
                 "duration_sec": 2, "cooldown_sec": 5}
            ]}
        ]})
        payload = {"machine": {"detection": [{"class": 0, "conf": .8, "box": (0, 0, 1, 1)}]}}
        self.assertEqual(engine.evaluate("cam1", payload, now=10), [])
        self.assertEqual(len(engine.evaluate("cam1", payload, now=12)), 1)
        full = {"machine": {"detection": [
            {"class": 0, "conf": .8, "box": (0, 0, 1, 1)},
            {"class": 0, "conf": .8, "box": (1, 1, 2, 2)}
        ]}}
        engine.evaluate("cam1", full, now=13)
        self.assertEqual(engine.evaluate("cam1", payload, now=14), [])
        self.assertEqual(len(engine.evaluate("cam1", payload, now=16)), 1)

    def test_image_modes_and_per_model_cooldown_are_independent(self):
        engine = AlarmRuleEngine({"models": [
            {"id": "a", "render": {"mode": "both"}, "alarm_rules": [
                {"id": "hit", "type": "class_detected", "class_id": 1}
            ]},
            {"id": "b", "render": {"mode": "model"}, "alarm_rules": [
                {"id": "hit", "type": "class_detected", "class_id": 1}
            ]}
        ]})
        one = {"detection": [{"class": 1, "conf": .8, "box": (0, 0, 1, 1)}]}
        events = engine.evaluate("cam1", {"a": one, "b": one}, now=1)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["image_mode"], "both")
        self.assertEqual(events[1]["image_mode"], "model")
        self.assertEqual(engine.evaluate("cam1", {"a": one, "b": one}, now=2), [])

    def test_legacy_fallback_preserves_model_1_rules(self):
        engine = AlarmRuleEngine({}, legacy_miss_seconds=2, legacy_cooldown_seconds=5)
        payload = {"model_1": {"detection": [], "machineCount": 1}}
        self.assertEqual(engine.evaluate("cam1", payload, now=10), [])
        events = engine.evaluate("cam1", payload, now=12)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["alarm_type"], "machine missing")


if __name__ == "__main__":
    unittest.main()
