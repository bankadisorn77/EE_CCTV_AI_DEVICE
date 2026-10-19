# Multi-model alarm rules

The worker keeps the existing RTSP acquisition, per-model batch inference, and asynchronous alert queue. Alarm behavior is configured in `config/config.json` under `models`; model IDs must match the generated IDs (`model_1`, `model_2`, …) or the IDs assigned by the model loader.

## Rule types

- `class_detected`: fires while a detection matches `class_id` or `class_name`.
- `count_below`: fires when the total number of detections (or detections matching a configured class) is below `expected_count`. Use `expected_count_from_camera: true` to use that camera’s legacy `machineCount` setting.
- `duration_sec`: how long a condition must remain true before firing.
- `cooldown_sec`: minimum time between fires for this camera/model/rule. Timers are isolated per camera, model, and rule.
- `enabled`: disable a model or individual rule without deleting its configuration.
- `alarm_type` and `message`: values sent in the existing alarm payload.

## Alarm image modes

Set `render.mode` per model or per rule (rule-level setting wins):

- `combined`: `imageBase64` contains detections from all configured models.
- `model`: `imageBase64` contains detections from the model whose rule fired.
- `both`: `imageBase64` is the combined image and the additive `modelImageBase64` field contains the triggering model’s image.

## Bounding-box color configuration

Put color settings in `alarm_image` for system defaults, `models[].render` for model-specific behavior, or `models[].alarm_rules[].render` for one alarm rule. Rule-level render settings take precedence over model settings when the rule supplies its own render object. Colors are **OpenCV BGR**, not RGB.

### Supported keys

- `color_mode: "auto"`: stable generated color for each model/class pair. This is the default. It intentionally ignores `class_colors`.
- `color_mode: "custom"`: use `class_colors` for classes listed there; unlisted classes fall back to stable auto colors.
- `color_mode: "model"`: use one color for all detections from a model. Set `model_color` to choose the color; if omitted, a stable color based on model ID is generated.
- `class_colors`: mapping from exact class names to BGR arrays, for example `"person": [0, 0, 255]` means red.
- `model_color`: BGR array used when `color_mode` is `"model"`.
- `thickness`, `show_label`, `show_confidence`, `font_scale`, and `text_thickness`: visual drawing settings.

### Example: custom colors for some classes, auto for the rest

```json
{
  "alarm_image": {
    "mode": "both",
    "color_mode": "auto",
    "show_label": true,
    "show_confidence": true,
    "thickness": 2
  },
  "models": [
    {
      "id": "model_1",
      "enabled": true,
      "render": {
        "mode": "both",
        "color_mode": "custom",
        "class_colors": {
          "machine": [0, 0, 255],
          "alarm_open": [0, 255, 0]
        }
      },
      "alarm_rules": []
    },
    {
      "id": "model_2",
      "enabled": true,
      "render": {
        "mode": "model",
        "color_mode": "auto"
      },
      "alarm_rules": []
    }
  ]
}
```

In this example, model 1 draws `machine` in red and `alarm_open` in green. Other classes on model 1 use auto colors. Model 2 uses auto colors for each class. Class names must exactly match the `class_name` values emitted by the model.

Example for a single color across all classes in a model:

```json
{
  "render": {
    "color_mode": "model",
    "model_color": [255, 128, 0]
  }
}
```

Legacy configuration without a `models` list continues to use the old `model_1` missing-machine and class-ID-1 alarm rules. Existing alarm payload fields are preserved; `modelId`, `ruleId`, and `modelImageBase64` are additive metadata.
