# Multi-model alarm rules

The worker keeps the existing RTSP acquisition, per-model batch inference, and asynchronous alert queue. Alarm behavior is configured in `config/config.json` under `models`; model IDs must match the generated IDs (`model_1`, `model_2`, …) or the IDs assigned by the model loader.

## Rule types

- `class_detected`: fires while a detection matches `class_id` or `class_name`.
- `count_below`: fires when the total number of detections (or detections matching a configured class) is below `expected_count`. Use `expected_count_from_camera: true` to use that camera's legacy `machineCount` setting.
- `duration_sec`: how long a condition must remain true before firing.
- `cooldown_sec`: minimum time between fires for this camera/model/rule. Timers are isolated per camera, model, and rule.
- `enabled`: disable a model or individual rule without deleting its configuration.
- `alarm_type` and `message`: values sent in the existing alarm payload.

Example:

```json
{
  "alarm_image": {
    "mode": "both",
    "show_label": true,
    "show_confidence": true
  },
  "models": [
    {
      "id": "model_1",
      "enabled": true,
      "render": { "mode": "both", "show_label": true, "show_confidence": true },
      "alarm_rules": [
        {
          "id": "person_detected",
          "type": "class_detected",
          "class_name": "person",
          "cooldown_sec": 5,
          "alarm_type": "person detected",
          "message": "person detected in camera"
        },
        {
          "id": "machine_missing",
          "type": "count_below",
          "expected_count": 2,
          "duration_sec": 2,
          "cooldown_sec": 5,
          "alarm_type": "machine missing",
          "message": "expected machine count not met"
        }
      ]
    }
  ]
}
```

## Alarm image modes

Set `render.mode` per model or per rule (rule-level setting wins):

- `combined`: `imageBase64` contains detections from all configured models.
- `model`: `imageBase64` contains detections from the model whose rule fired.
- `both`: `imageBase64` is the combined image and the additive `modelImageBase64` field contains the triggering model's image.

Legacy configuration without a `models` list continues to use the old `model_1` missing-machine and class-ID-1 alarm rules. Existing alarm payload fields are preserved; `modelId`, `ruleId`, and `modelImageBase64` are additive metadata.
