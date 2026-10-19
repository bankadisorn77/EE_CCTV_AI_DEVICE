"""Configurable detection renderer used for alarm snapshots."""
import hashlib

import cv2


def _color_for(det, render_config):
    colors = render_config.get("class_colors", {}) if isinstance(render_config, dict) else {}
    key = str(det.get("class_name", det.get("class", "unknown")))
    color = colors.get(key)
    if color is not None and isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(max(0, min(255, c))) for c in color)
    digest = hashlib.md5(f"{det.get('model_id', '')}:{key}".encode("utf-8")).digest()
    return (64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192)


def draw_detections(image, detections, render_config=None):
    """Draw detections on a copy. Config supports show_label/show_confidence/class_colors."""
    cfg = render_config or {}
    out = image.copy()
    for det in detections or []:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        color = _color_for(det, cfg)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(cfg.get("thickness", 2)))
        parts = []
        if cfg.get("show_label", True):
            parts.append(str(det.get("class_name", det.get("class", "object"))))
        if cfg.get("show_confidence", True) and det.get("conf") is not None:
            parts.append(f"{float(det['conf']):.2f}")
        if parts:
            text = " ".join(parts)
            y = max(14, y1 - 5)
            cv2.putText(out, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                        float(cfg.get("font_scale", 0.5)), color,
                        int(cfg.get("text_thickness", 1)), cv2.LINE_AA)
    return out
