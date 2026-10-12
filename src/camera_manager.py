import sys
from pathlib import Path
import torch
import cv2
import asyncio
import json
import atexit
import time
import base64
# Removed Triton gRPC import

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ee_cctv_ai import model_process_ultralytics as model_process
from src.ee_cctv_ai.acquistion import Acquisition
from src.ee_cctv_ai.alert_service import AlertService
from turbojpeg import TurboJPEG

class CCTV_AI_Ultralytics: 
    def __init__(self, rtsp_urls, shared_data, API_KEY, nodeJS_URL):
        self.stop_process = False
        self._cleanup_done = False
        self._tasks = []
        self.shared_data = shared_data
        self.API_KEY = API_KEY
        self.nodeJS_URL = nodeJS_URL
        self.urls = rtsp_urls
        self.read_config()

        self.miss_counter   = {f"cam{i+1}": 0 for i in range(len(rtsp_urls))}
        self.alarm_cooldown = {f"cam{i+1}": 0 for i in range(len(rtsp_urls))}
        self.MISS_THRESHOLD = 25
        self.COOLDOWN_SEC   = 5
        self.start_time     = None

        self.fps_stats = {
            f"cam{i+1}": {
                "capture": 0.0,
                "process": 0.0,
                "capture_count": 0,
                "process_count": 0,
                "capture_last_time": time.time(),
                "process_last_time": time.time(),
            }
            for i in range(len(rtsp_urls))
        }

        # Bound per-camera latency and memory: only keep the freshest pending frame.
        self.frame_queues = {f"cam{i+1}": asyncio.Queue(maxsize=1) for i in range(len(rtsp_urls))}
        self.alert_service = AlertService(self.nodeJS_URL, self.API_KEY)

        # Readers reconnect independently; one camera must not abort pipeline startup.
        self.cameras = [
            Acquisition(rtsp_urls[i], key=f"cam{i+1}")
            for i in range(len(rtsp_urls))
        ]
        atexit.register(self.cleanup)

    def read_config(self):
        try:
            config_path = PROJECT_ROOT / "config" / "config.json"
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.shared_data.set_config(config_data.get("cameras", []))
                self.shared_data.set_confidence_threshold(config_data.get("confidence_threshold", 0.5))
                print("Unified config loaded successfully.")
        except Exception as e:
            print(f"Config error (config.json): {e}")
            self.shared_data.set_config([{"name": f"cam{i+1}", "roi": None} for i in range(len(self.urls))])
            self.shared_data.set_confidence_threshold(0.5)

    async def producer(self, cam_idx):
        camera = self.cameras[cam_idx]
        camera.start()
        queue = self.frame_queues[camera.name]
        interval = 1.0 / 30.0
        last_frame_id = -1

        while not self.stop_process:
            frame = camera.image
            frame_id = camera.frame_id
            if frame is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                if queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put({"frame": frame, "cam_key": camera.name})
            await asyncio.sleep(interval)
    async def worker(self, cam_id):
        queue = self.frame_queues[cam_id]
        print(f"Started Ultralytics Worker for {cam_id}")

        while not self.stop_process:
            item = None
            try:
                item = await queue.get()
                frame = item["frame"]
                cam_key = item["cam_key"]
                cam_config = self.shared_data.get_config_by_cam(cam_key)
                confidence = self.shared_data.get_confidence_threshold()
                processed_img, detection_results = await model_process.predict_model(
                    frame, cam_config, size=640, conf=confidence
                )
                if processed_img is not None:
                    await self.logic(processed_img, cam_key, detection_results)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Worker Error on {cam_id}: {e}")
                self.shared_data.set_LIFE("error")
                await asyncio.sleep(0.1)
            finally:
                if item is not None:
                    queue.task_done()

    def update_capture_fps(self, cam_key):
        stats = self.fps_stats[cam_key]
        now = time.time()
        stats["capture_count"] += 1
        elapsed = now - stats["capture_last_time"]
        if elapsed >= 1.0:
            stats["capture"] = stats["capture_count"] / elapsed
            stats["capture_count"] = 0
            stats["capture_last_time"] = now
            print(f"[Camera FPS] {cam_key}: {stats['capture']:.1f}")

    def update_process_fps(self, cam_key):
        stats = self.fps_stats[cam_key]
        now = time.time()
        stats["process_count"] += 1
        elapsed = now - stats["process_last_time"]
        if elapsed >= 1.0:
            stats["process"] = stats["process_count"] / elapsed
            stats["process_count"] = 0
            stats["process_last_time"] = now
            print(f"[Inference FPS] {cam_key}: {stats['process']:.1f}")

    def get_camera_fps(self, cam_key=None):
        if cam_key:
            return self.fps_stats.get(cam_key, {})
        return self.fps_stats

    async def run(self):
        self.start_time = time.time()
        print("CCTV_AI System Running (Ultralytics Mode)...")
        await self.alert_service.start()
        self._tasks = [
            asyncio.create_task(self.producer(i), name=f"camera-producer-{i+1}")
            for i in range(len(self.cameras))
        ] + [
            asyncio.create_task(self.worker(f"cam{i+1}"), name=f"inference-worker-{i+1}")
            for i in range(len(self.cameras))
        ]
        try:
            await asyncio.gather(*self._tasks)
        finally:
            self.cleanup()
            await self.alert_service.stop()

    async def alarm_detect(self, camid, detection_class, message, confidence, image):
        image_base64_str = ""
        if image is not None:
            loop = asyncio.get_event_loop()
            image_base64_str = await loop.run_in_executor(
                model_process.cpu_executor,
                self._encode_image,
                image
            )
        payload = {
            "alarmType": detection_class,
            "message": message,
            "cameraId": f"{camid[:3].upper()}-0{camid[3:]}",
            "confidence": confidence * 100,
            "imageBase64": f"data:image/jpeg;base64,{image_base64_str}" if image_base64_str else None
        }

        # Queue the alert and return immediately; delivery must not block detection.
        await self.alert_service.enqueue(payload)
        return payload
    
    async def logic(self, image, cam_id, cam_data):
        try:
            cam_data = cam_data if cam_data is not None else {"detection": [], "machineCount": 0}
            now = time.time()
            # print(cam_data)

            if len(cam_data["detection"]) < cam_data["machineCount"]:
                self.miss_counter[cam_id] += 1
                if self.miss_counter[cam_id] >= self.MISS_THRESHOLD:
                    if now - self.alarm_cooldown[cam_id] >= self.COOLDOWN_SEC:
                        self.alarm_cooldown[cam_id] = now
                        # print(f"Detection on {cam_id}: Class=machine missing, miss={self.miss_counter[cam_id]} frames")
                        await self.alarm_detect(cam_id, "machine missing", "can not detect machine", 0.0, image)
            else:
                self.miss_counter[cam_id] = 0
                class1_det = next((det for det in cam_data["detection"] if det["class"] == 1), None)
                if class1_det and now - self.alarm_cooldown[cam_id] >= self.COOLDOWN_SEC:
                    self.alarm_cooldown[cam_id] = now
                    confidence = class1_det["conf"]
                    # print(f"Detection on {cam_id}: Class=alarm open, Confidence={confidence * 100:.1f}%")
                    await self.alarm_detect(cam_id, "alarm open", "open detected", confidence, image)

        except Exception as e:
            print(f"Error in logic: {e}")


    def _encode_image(self, image):
        try:
            jpeg_data = TurboJPEG().encode(image, quality=80)
            return base64.b64encode(jpeg_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8') if success else ""

    def cleanup(self):
        if self._cleanup_done:
            return
        self._cleanup_done = True
        self.stop_process = True
        print("Cleaning up CCTV_AI (Ultralytics)...")
        for task in self._tasks:
            if not task.done():
                task.cancel()
        for cam in getattr(self, "cameras", []):
            try:
                cam.stop()
            except Exception as exc:
                print(f"Camera stop warning: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.shared_data.set_LIFE("active")
        print("CCTV_AI (Ultralytics) cleanup complete.")
