"""Blocking initial-frame capture used during device registration.

Call this function through an executor from async code so camera I/O never blocks
the FastAPI event loop.
"""
import base64
import time

import cv2


def init_cctv_ai_frame(rtsp_urls, max_retries=5):
    for attempt in range(1, max_retries + 1):
        initial_frames = {}
        try:
            for i, url in enumerate(rtsp_urls):
                cam_id = f"cam{i + 1}"
                if not url:
                    print(f"[{cam_id}] No stream URL configured.")
                    initial_frames[cam_id] = None
                    continue

                print(f"Capturing initial frame for {cam_id}...")
                cap = None
                frame = None
                try:
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        for _ in range(50):
                            ret, candidate = cap.read()
                            if ret and candidate is not None:
                                frame = candidate
                                break
                    if frame is None:
                        print(f"[{cam_id}] Initial frame unavailable.")
                except Exception as exc:
                    print(f"[{cam_id}] Initial capture error: {exc}")
                finally:
                    if cap is not None:
                        cap.release()
                initial_frames[cam_id] = frame

            encoded = {}
            for cam_id, frame in initial_frames.items():
                if frame is None:
                    encoded[cam_id] = None
                    continue
                success, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                encoded[cam_id] = (
                    f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                    if success else None
                )
            return encoded
        except Exception as exc:
            print(f"Initial frame capture failed ({attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                time.sleep(5)

    print("FATAL: Failed to initialize initial camera frames.")
    return None
