import time 
import numpy as np
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ultralytics import YOLO

# Load the Ultralytics model from the engine file
# Using forward slashes for better path compatibility.
MODEL = YOLO('models/best_V6.pt', task='detect')

cpu_executor = ThreadPoolExecutor(max_workers=4)

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

    loop = asyncio.get_event_loop()
    try:
        predict_func = partial(MODEL.predict, cropped_img, imgsz=size, verbose=False)
        results_obj = await loop.run_in_executor(cpu_executor, predict_func)

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
    # print(payload)
    cv2.imwrite(f"test/alam/test_output_{time.time()}_{config_cam.get('name', 'unknown')}.jpg", img)
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
    cam_name = config_cam.get("name", "unknown") if config_cam is not None else "unknown"
    cv2.imwrite(f"test/alam/test_cropped_img{cam_name}.jpg", cropped_img)
    return cropped_img, croped_shape, roi_offset

async def preprocess(image, config_cam, size):
    loop = asyncio.get_event_loop()
    img, croped_shape, letterbox_info = await loop.run_in_executor(
        cpu_executor, 
        cv2_preprocess, 
        image, config_cam, size
    )
    return img, croped_shape, letterbox_info

if __name__ == "__main__":
    async def main():
        """
        Asynchronous function to run real-time inference on multiple video files and display the results.
        """
        # List of video paths for testing
        video_paths = [
            ".\\test\\D01.mp4",
            ".\\test\\D02.mp4",
            ".\\test\\D03.mp4",
            ".\\test\\D04.mp4",
        ]

        # Open video capture for each path
        caps = [cv2.VideoCapture(p) for p in video_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Error: Could not open one or more video files.")
            return

        # Create mock camera configurations for testing
        mock_configs = [{'name': f'cam{i+1}', 'roi': None, 'machineCount': 0} for i in range(len(video_paths))]

        try:
            while True:
                # Read a frame from each video
                frames = [cap.read()[1] for cap in caps]

                # If any frame is None (video ended), break the loop
                if any(frame is None for frame in frames):
                    print("A video has ended. Exiting.")
                    break

                # Create a list of concurrent prediction tasks
                tasks = [predict_model(frames[i], mock_configs[i]) for i in range(len(frames))]
                results = await asyncio.gather(*tasks)

                # Display the processed frames
                for i, (processed_img, payload) in enumerate(results):
                    cv2.imshow(f"Real-time Detection - Cam {i+1}", processed_img)
                    # print(f'CAM {i+1} Payload: {payload}')

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Clean up resources
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()

    asyncio.run(main())
