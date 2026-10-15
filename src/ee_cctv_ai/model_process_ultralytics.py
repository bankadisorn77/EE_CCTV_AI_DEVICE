import os
import time
import numpy as np
import cv2
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_V6.engine"
MODEL = YOLO(str(MODEL_PATH), task="detect")

# ลด thread pool สำหรับ CPU ไม่ให้แย่งคอร์กับ GStreamer/Jetson Core
cpu_executor = ThreadPoolExecutor(max_workers=2)
_inference_lock = threading.Lock()
SAVE_DEBUG_IMAGES = os.getenv("EE_CCTV_SAVE_DEBUG_IMAGES", "0").lower() in {"1", "true", "yes"}


def _predict_fast(cropped_img, size, conf):
    """รัน TensorRT Engine พร้อมล็อกเธรดเพื่อความปลอดภัยของ Context"""
    with _inference_lock:
        return MODEL.predict(cropped_img, imgsz=size, conf=conf, verbose=False)


def process_image_fast(result_obj, img, roi_offset, config_cam, conf=0.5):
    col1, row1 = roi_offset
    count = config_cam.get("machineCount", 0) if config_cam else 0
    h_orig, w_orig = img.shape[:2]

    detection_class = []
    boxes = result_obj.boxes

    if boxes is not None and len(boxes) > 0:
        # ดึง Tensor ออกมาเป็น Numpy Array ก้อนเดียว [N, 6] -> x1, y1, x2, y2, conf, cls
        data = boxes.data.cpu().numpy()
        
        # กรองเฉพาะค่า conf ที่ผ่านเกณฑ์
        valid_mask = data[:, 4] >= conf
        valid_boxes = data[valid_mask]

        for item in valid_boxes:
            x1, y1, x2, y2, conf_score, cls_id = item
            cls_id = int(cls_id)

            rx1 = max(0, int(x1) + col1)
            ry1 = max(0, int(y1) + row1)
            rx2 = min(w_orig, int(x2) + col1)
            ry2 = min(h_orig, int(y2) + row1)

            # วาดกรอบเฉพาะเมื่อต้องการใช้งาน
            color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)

            detection_class.append({
                "class": cls_id,
                "conf": round(float(conf_score), 2),
            })

    result = {
        "detection": detection_class,
        "machineCount": count if count is not None else 0,
    }
    return img, result


def cv2_preprocess_fast(img, config_cam):
    """ทำ ROI Crop แบบ View (ไม่สั่ง .copy() เปลือง RAM)"""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    h_orig, w_orig = img.shape[:2]
    row1, col1 = 0, 0
    cropped_img = img

    roi = config_cam.get("roi") if config_cam is not None else None
    if roi and isinstance(roi, dict):
        try:
            r1 = max(0, int(roi.get("row1") or 0))
            r2 = min(h_orig, int(roi.get("row2") or h_orig))
            c1 = max(0, int(roi.get("col1") or 0))
            c2 = min(w_orig, int(roi.get("col2") or w_orig))

            if (r2 > r1) and (c2 > c1):
                # ใช้ slice view โดยไม่ .copy()
                cropped_img = img[r1:r2, c1:c2]
                row1, col1 = r1, c1
        except Exception as e:
            print(f"Error in fast crop: {e}")

    return cropped_img, (col1, row1)


async def predict_model(img, config_cam, size=640, conf=0.5):
    loop = asyncio.get_running_loop()

    # Preprocess ตัดภาพใน executor
    cropped_img, roi_offset = await loop.run_in_executor(
        cpu_executor, cv2_preprocess_fast, img, config_cam
    )

    try:
        prediction_future = loop.run_in_executor(
            cpu_executor, _predict_fast, cropped_img, size, conf
        )
        results_obj = await asyncio.shield(prediction_future)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        cam_name = config_cam.get("name", "unknown") if config_cam else "unknown"
        print(f"Inference Error on {cam_name}: {e}")
        return img, {"detection": [], "machineCount": config_cam.get("machineCount", 0) if config_cam else 0}

    # Post-process วาดกรอบและสรุปผล
    img, payload = await loop.run_in_executor(
        cpu_executor, process_image_fast, results_obj[0], img, roi_offset, config_cam, conf
    )

    return img, payload