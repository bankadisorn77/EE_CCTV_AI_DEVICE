import os
import time
import numpy as np
import cv2
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_V6.pt"
MODEL = YOLO(str(MODEL_PATH), task="detect")
cpu_executor = ThreadPoolExecutor(max_workers=4)
# Shared YOLO/GPU instance: serialize prediction until target-device tests prove
# concurrent calls are safe and improve throughput.
_inference_lock = threading.Lock()
SAVE_DEBUG_IMAGES = os.getenv("EE_CCTV_SAVE_DEBUG_IMAGES", "0").lower() in {"1", "true", "yes"}


def _predict_sync(cropped_img, size, conf):
    """Run the shared YOLO model under a thread lock, independent of asyncio loops."""
    with _inference_lock:
        return MODEL.predict(cropped_img, imgsz=size, conf=conf, verbose=False)

def process_image_ultralytics(
    result_obj, img, roi_offset, config_cam, conf=0.5
):
    col1, row1 = roi_offset
    count = config_cam.get("machineCount", 0) if config_cam else 0
    h_orig, w_orig = img.shape[:2]

    detection_class = []

    boxes = result_obj.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            conf_score = float(box.conf[0].cpu().numpy())
            if conf_score < conf:
                continue

            cls_id = int(box.cls[0].cpu().numpy())
            # xyxy พิกัดบนพื้นที่ cropped_img
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            rx1 = int(x1) + col1
            ry1 = int(y1) + row1
            rx2 = int(x2) + col1
            ry2 = int(y2) + row1

            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w_orig, rx2), min(h_orig, ry2)

            # วาดกรอบ
            color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)

            detection_class.append(
                {
                    "class": cls_id,
                    "conf": round(conf_score, 2),
                }
            )

    result = {
        "detection": detection_class,
        "machineCount": count if count is not None else 0,
    }
    return img, result

async def predict_model(img, config_cam, size=640, conf=0.5):
    cropped_img, croped_shape, roi_offset = await preprocess(
        img, config_cam, size
    )

    loop = asyncio.get_running_loop()
    try:
        # The lock lives in a worker thread, not an asyncio event loop. This avoids
        # "Future attached to a different loop" when the pipeline is restarted.
        prediction_future = loop.run_in_executor(
            cpu_executor, _predict_sync, cropped_img, size, conf
        )
        try:
            # Do not let cancellation abandon a still-running CUDA inference call.
            results_obj = await asyncio.shield(prediction_future)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(prediction_future)
            except Exception:
                pass
            raise

        # Extract results [x1, y1, x2, y2, conf, cls]
        results = results_obj[0].boxes.data.cpu().numpy()
    except Exception as e:
        cam_name = config_cam.get("name", "unknown_cam") if config_cam else "unknown_cam"
        print(f"Ultralytics Inference Error on {cam_name}: {e}")
        results = None

    if results is None or len(results) == 0:
        return img, {
            "detection": [],
            "machineCount": config_cam.get("machineCount", 0) if config_cam else 0,
        }

    img, payload = await loop.run_in_executor(
        cpu_executor,
        process_image_ultralytics,
        results_obj[0],
        img,
        roi_offset,
        config_cam,
        conf,
    )
    if SAVE_DEBUG_IMAGES:
        os.makedirs("test/alam", exist_ok=True)
        cam_name = config_cam.get("name", "unknown") if config_cam else "unknown"
        cv2.imwrite(f"test/alam/test_output_{time.time()}_{cam_name}.jpg", img)
    return img, payload


def cv2_preprocess(img, config_cam, size):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    h_orig, w_orig = img.shape[:2]
    row1, col1 = 0, 0
    cropped_img = img.copy()

    roi = config_cam.get("roi") if config_cam is not None else None
    if roi is None or roi == {}:
        print("No ROI provided. Skipping crop.")
    else:
        try:
            row1_val = roi.get("row1", 0)
            row2_val = roi.get("row2", h_orig)
            col1_val = roi.get("col1", 0)
            col2_val = roi.get("col2", w_orig)

            row1 = 0 if row1_val is None else int(row1_val)
            row2 = h_orig if row2_val is None else int(row2_val)
            col1 = 0 if col1_val is None else int(col1_val)
            col2 = w_orig if col2_val is None else int(col2_val)

            row1, row2 = max(0, row1), min(h_orig, row2)
            col1, col2 = max(0, col1), min(w_orig, col2)

            if (row2 > row1) and (col2 > col1):
                cropped_img = img[row1:row2, col1:col2].copy()
            else:
                print("Warning: Invalid ROI. Skipping crop.")
        except Exception as e:
            print(f"Error in cropping image: {e}")

    croped_shape = cropped_img.shape[:2]
    roi_offset = (col1, row1)
    # print(f"{config_cam.get('name', 'unknown')} Original image shape: {img.shape}, Cropped image shape: {cropped_img.shape}")
    # print(f"DEBUG [{config_cam.get('name')}] ROI input from config: {roi}")
    if SAVE_DEBUG_IMAGES:
        os.makedirs("test/alam", exist_ok=True)
        cam_name = config_cam.get("name", "unknown") if config_cam is not None else "unknown"
        cv2.imwrite(f"test/alam/test_cropped_img{cam_name}.jpg", cropped_img)
    return cropped_img, croped_shape, roi_offset

async def preprocess(image, config_cam, size):
    loop = asyncio.get_running_loop()
    img, croped_shape, letterbox_info = await loop.run_in_executor(
        cpu_executor, 
        cv2_preprocess, 
        image, config_cam, size
    )
    return img, croped_shape, letterbox_info

if __name__ == "__main__":
    import os
    os.makedirs("test/alam", exist_ok=True)

    async def main():
        video_paths = [
            "./test/C01.mp4",
        ]

        caps = [cv2.VideoCapture(p) for p in video_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Error: Could not open one or more video files.")
            return

        mock_configs = [{'name': f'cam{i+1}', 'roi': None, 'machineCount': 0} for i in range(len(video_paths))]

        try:
            frame_idx = 0
            while True:
                frames = [cap.read() for cap in caps]
                
                # ตรวจสอบว่ามีวิดีโอไหนจบหรืออ่านเฟรมไม่ได้ไหม
                if any(not ret or frame is None for ret, frame in frames):
                    print("A video has ended or invalid frame. Exiting.")
                    break

                # รันทีละกล้องเพื่อป้องกันปัญหา Thread/CUDA ชนกัน
                for i, (ret, frame) in enumerate(frames):
                    config = mock_configs[i]
                    # เรียก predict แบบรอทีละกล้อง
                    processed_img, payload = await predict_model(frame, config)
                    
                    # บันทึกรูปภาพเก็บไว้เช็คผลลัพธ์แทนการใช้ cv2.imshow
                    out_path = f"test/alam/cam{i+1}_frame_{frame_idx}.jpg"
                    cv2.imwrite(out_path, processed_img)
                    print(f"Processed Cam {i+1} -> Saved to {out_path}")

                frame_idx += 1
                
                # ทดสอบรันสัก 10 เฟรมพอสำหรับเทส (หรือลบเงื่อนไขนี้ออกถ้าจะรันยาวๆ)
                # if frame_idx >= 10:
                #     print("Test finished 10 frames.")
                #     break

        finally:
            for cap in caps:
                cap.release()
            print("All video captures released.")

    asyncio.run(main())