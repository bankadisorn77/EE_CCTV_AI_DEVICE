"""Configurable detection renderer for alarm snapshots.

Color values use OpenCV BGR order.
- auto: stable generated color per model and class.
- custom: use class_colors for listed classes, auto colors for the rest.
- model: use one model_color for all detections from a model; if omitted,
  generate one stable color based on the model ID.
"""
import hashlib
import cv2

_VALID_COLOR_MODES = {"auto", "custom", "model"}

def _normalize_color(value):
    """Validate and clamp a BGR color; return None if invalid."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(max(0, min(255, int(channel))) for channel in value)
    except (TypeError, ValueError, OverflowError):
        return None

def _auto_color(model_id, class_key):
    """Generate a stable, reasonably bright BGR color."""
    digest = hashlib.md5("{}:{}".format(model_id, class_key).encode("utf-8")).digest()
    return (64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192)

def _color_for(det, render_config):
    """Choose a detection color from the configured color mode."""
    cfg = render_config if isinstance(render_config, dict) else {}
    model_id = str(det.get("model_id", ""))
    class_key = str(det.get("class_name", det.get("class", "unknown")))
    mode = str(cfg.get("color_mode", "auto")).lower()
    if mode not in _VALID_COLOR_MODES:
        mode = "auto"
    if mode == "custom":
        colors = cfg.get("class_colors", {})
        if isinstance(colors, dict):
            color = _normalize_color(colors.get(class_key))
            if color is not None:
                return color
        return _auto_color(model_id, class_key)
    if mode == "model":
        color = _normalize_color(cfg.get("model_color"))
        return color if color is not None else _auto_color(model_id, "model")
    return _auto_color(model_id, class_key)

def draw_detections(image, detections, render_config=None):
    """Draw detections on a copy, supporting per-model render settings."""
    cfg = render_config if isinstance(render_config, dict) else {}
    out = image.copy()
    model_configs = cfg.get("models", {})
    for det in detections or []:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        det_cfg = model_configs.get(str(det.get("model_id")), cfg) if isinstance(model_configs, dict) else cfg
        if not isinstance(det_cfg, dict):
            det_cfg = cfg
        x1, y1, x2, y2 = (int(v) for v in box)
        color = _color_for(det, det_cfg)
        thickness = max(1, int(det_cfg.get("thickness", cfg.get("thickness", 2))))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        parts = []
        if det_cfg.get("show_label", cfg.get("show_label", True)):
            parts.append(str(det.get("class_name", det.get("class", "object"))))
        if det_cfg.get("show_confidence", cfg.get("show_confidence", True)) and det.get("conf") is not None:
            parts.append("{:.2f}".format(float(det["conf"])))
        if parts:
            cv2.putText(out, " ".join(parts), (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        float(det_cfg.get("font_scale", cfg.get("font_scale", 0.5))), color,
                        max(1, int(det_cfg.get("text_thickness", cfg.get("text_thickness", 1)))), cv2.LINE_AA)
    return out
