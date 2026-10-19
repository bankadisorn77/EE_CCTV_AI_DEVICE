"""Config-driven, model-agnostic alarm rule evaluation.

Supported rule types: class_detected and count_below. State is isolated by
(camera, model, rule), so cooldowns and missing-duration timers never collide.
"""
import time


class AlarmRuleEngine:
    def __init__(self, config=None, legacy_miss_seconds=2.0, legacy_cooldown_seconds=5.0):
        self.config = config or {}
        self.legacy_miss_seconds = float(legacy_miss_seconds)
        self.legacy_cooldown_seconds = float(legacy_cooldown_seconds)
        self._since = {}
        self._last_fired = {}

    def update_config(self, config):
        self.config = config or {}

    @staticmethod
    def _class_matches(det, rule):
        if "class_id" in rule and rule["class_id"] is not None:
            return det.get("class") == int(rule["class_id"])
        target = rule.get("class_name")
        return target is None or str(det.get("class_name", "")).lower() == str(target).lower()

    def _rules_by_model(self, models_payload):
        configured = self.config.get("models")
        if isinstance(configured, list):
            return {
                str(item.get("id") or item.get("name")): item
                for item in configured
                if isinstance(item, dict) and item.get("enabled", True)
                and (item.get("id") or item.get("name"))
            }
        # Backward compatibility with the old model_1 machineCount/class==1 logic.
        if "model_1" not in models_payload:
            return {}
        return {
            "model_1": {
                "enabled": True,
                "render": self.config.get("alarm_image", {"mode": "combined"}),
                "alarm_rules": [
                    {
                        "id": "legacy_machine_missing", "type": "count_below",
                        "expected_count_from_camera": True,
                        "duration_sec": self.legacy_miss_seconds,
                        "cooldown_sec": self.legacy_cooldown_seconds,
                        "alarm_type": "machine missing",
                        "message": "can not detect machine",
                    },
                    {
                        "id": "legacy_alarm_open", "type": "class_detected",
                        "class_id": 1, "cooldown_sec": self.legacy_cooldown_seconds,
                        "alarm_type": "alarm open", "message": "open detected",
                    },
                ],
            }
        }

    @staticmethod
    def _as_render_dict(value):
        if isinstance(value, str):
            return {"mode": value}
        return value if isinstance(value, dict) else {}

    def evaluate(self, cam_id, models_payload, now=None):
        now = time.time() if now is None else float(now)
        if not models_payload:
            return []
        configured = self._rules_by_model(models_payload)
        all_detections = [
            dict(det, model_id=model_id)
            for model_id, data in models_payload.items()
            for det in data.get("detection", [])
        ]
        global_render = self._as_render_dict(self.config.get("alarm_image", {}))
        model_render_configs = {}
        for configured_id, item in configured.items():
            model_render = self._as_render_dict(item.get("render", {}))
            model_render_configs[configured_id] = {**global_render, **model_render}

        events = []
        for model_id, model_cfg in configured.items():
            data = models_payload.get(model_id)
            if data is None:
                continue
            detections = data.get("detection", [])
            for index, rule in enumerate(model_cfg.get("alarm_rules", [])):
                if not isinstance(rule, dict) or rule.get("enabled", True) is False:
                    continue
                rule_id = str(rule.get("id") or "rule_{}".format(index + 1))
                state_key = (cam_id, model_id, rule_id)
                kind = rule.get("type")
                matched = [d for d in detections if self._class_matches(d, rule)]
                active = False
                if kind == "class_detected":
                    active = bool(matched)
                elif kind == "count_below":
                    expected = rule.get("expected_count")
                    if rule.get("expected_count_from_camera"):
                        expected = data.get("machineCount", 0)
                    if expected is None:
                        expected = 0
                    actual = len(matched) if ("class_id" in rule or "class_name" in rule) else len(detections)
                    active = actual < max(0, int(expected))
                else:
                    continue

                if not active:
                    self._since.pop(state_key, None)
                    continue
                self._since.setdefault(state_key, now)
                duration = max(0.0, float(rule.get("duration_sec", 0)))
                if now - self._since[state_key] < duration:
                    continue
                cooldown = max(0.0, float(rule.get("cooldown_sec", self.legacy_cooldown_seconds)))
                if now - self._last_fired.get(state_key, float("-inf")) < cooldown:
                    continue
                self._last_fired[state_key] = now

                selected = matched if kind == "class_detected" else detections
                confidence = max((float(d.get("conf", 0.0)) for d in matched), default=0.0)
                # Merge defaults -> model settings -> rule settings, so rule-level
                # colors/styles override only the triggering model for this event.
                rule_render = self._as_render_dict(rule.get("render", {}))
                effective_render = {**model_render_configs.get(model_id, {}), **rule_render}
                mode = effective_render.get("mode", "combined")
                if mode not in {"combined", "model", "both"}:
                    mode = "combined"
                effective_render["mode"] = mode
                render_config = dict(effective_render)
                render_config["models"] = dict(model_render_configs)
                render_config["models"][model_id] = effective_render

                events.append({
                    "camera_id": cam_id,
                    "model_id": model_id,
                    "rule_id": rule_id,
                    "alarm_type": rule.get("alarm_type", rule_id),
                    "message": rule.get("message", rule.get("alarm_type", rule_id)),
                    "confidence": confidence,
                    "detections": all_detections if mode in {"combined", "both"} else [
                        dict(det, model_id=model_id) for det in selected
                    ],
                    "model_detections": [dict(det, model_id=model_id) for det in selected],
                    "image_mode": mode,
                    "render": render_config,
                })
        return events
