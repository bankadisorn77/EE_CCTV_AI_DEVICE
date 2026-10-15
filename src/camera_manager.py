import asyncio
import atexit
import base64
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ee_cctv_ai import model_process_ultralytics as model_process
from src.ee_cctv_ai.acquistion import Acquisition
from src.ee_cctv_ai.alert_service import AlertService
from src.ee_cctv_ai.config_store import load_config

try:  # optional: CI / dev machines may not have torch or libturbojpeg
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    from turbojpeg import TurboJPEG
except Exception:  # pragma: no cover
    TurboJPEG = None


class CCTV_AI_Ultralytics:

    def __init__(self, rtsp_urls, shared_data, API_KEY, nodeJS_URL, camera_adapters=None):
        self.stop_process = False
        self._cleanup_done = False
        self._tasks = []
        self.shared_data = shared_data
        self.API_KEY = API_KEY
        self.nodeJS_URL = nodeJS_URL
        self.urls = rtsp_urls
        self.read_config()

        names = [f"cam{i+1}" for i in range(len(rtsp_urls))]
        self.cam_names = names
        self.miss_since = {n: None for n in names}
        self.alarm_cooldown = {n: 0 for n in names}
        # "machine missing" must persist this long (was 25 inferred frames, i.e. it
        # depended on inference FPS). Override with EE_CCTV_MISS_SECONDS.
        self.MISS_SECONDS = float(os.getenv("EE_CCTV_MISS_SECONDS", "2.0"))
        self.COOLDOWN_SEC = 5
        self.start_time = None

        self.fps_stats = {
            n: {
                "capture": 0.0,
                "process": 0.0,
                "capture_count": 0,
                "process_count": 0,
                "capture_last_time": time.time(),
                "process_last_time": time.time(),
            }
            for n in names
        }

        # Newest not-yet-processed frame per camera (older ones are overwritten = dropped).
        self._pending = {}
        self._last_ids = {n: -1 for n in names}
        self._rr = 0  # round-robin start so no camera starves when batch < #cameras
        self._loop = None
        self.frame_event = None
        self._result_queue = None
        self._prof = self._new_prof()

        self.alert_service = AlertService(self.nodeJS_URL, self.API_KEY)

        # Readers reconnect independently; one camera must not abort pipeline startup.
        self.cameras = list(camera_adapters) if camera_adapters is not None else [
            Acquisition(rtsp_urls[i], key=names[i]) for i in range(len(rtsp_urls))
        ]
        if len(self.cameras) != len(names):
            raise ValueError("camera_adapters must match the configured RTSP camera count")

        self.jpeg = None
        if TurboJPEG is not None:
            # explicit ARM64/Jetson path avoids the slow fallback lookup
            try:
                self.jpeg = TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
            except Exception:
                try:
                    self.jpeg = TurboJPEG()
                except Exception:
                    self.jpeg = None

        atexit.register(self.cleanup)

    # ------------------------------------------------------------------ config
    def read_config(self):
        try:
            config_data = load_config(PROJECT_ROOT / "config" / "config.json")
            self.shared_data.set_config(config_data.get("cameras", []))
            self.shared_data.set_confidence_threshold(config_data.get("confidence_threshold", 0.5))
            print("Unified config loaded successfully.")
        except Exception as e:
            print(f"Config error (config.json): {e}")
            self.shared_data.set_config([{"name": f"cam{i+1}", "roi": None} for i in range(len(self.urls))])
            self.shared_data.set_confidence_threshold(0.5)

    # ----------------------------------------------------------- frame intake
    def _on_camera_frame(self, cam_key):
        """Runs in the camera's reader thread for every decoded frame."""
        self.update_capture_fps(cam_key)  # counts EVERY decoded frame, not just processed ones
        loop = self._loop
        event = self.frame_event
        if loop is None or event is None or event.is_set():
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:  # loop already closed during shutdown
            pass

    def _collect_new_frames(self):
        for cam in self.cameras:
            frame, fid = cam.take_new_frame(self._last_ids[cam.name])
            if frame is not None:
                self._last_ids[cam.name] = fid
                self._pending[cam.name] = frame  # overwrite -> stale frame dropped

    def _take_batch(self):
        limit = max(1, model_process.max_batch())
        names = self.cam_names
        order = names[self._rr:] + names[:self._rr]
        items = []
        for name in order:
            if name in self._pending:
                items.append({"cam_key": name, "frame": self._pending.pop(name)})
                if len(items) >= limit:
                    break
        if items:
            self._rr = (names.index(items[-1]["cam_key"]) + 1) % len(names)
        return items

    # -------------------------------------------------------------- inference
    async def _inference_loop(self):
        loop = self._loop
        print("Started batched inference loop")
        while not self.stop_process:
            await self.frame_event.wait()
            self.frame_event.clear()
            self._collect_new_frames()
            items = self._take_batch()
            if not items:
                continue
            if self._pending:          # more cameras waiting than fit in this batch
                self.frame_event.set()

            try:
                cfgs = [self.shared_data.get_config_by_cam(it["cam_key"]) for it in items]
                conf = self.shared_data.get_confidence_threshold()
                dets, timing = await loop.run_in_executor(
                    model_process.gpu_executor,
                    model_process.predict_batch_sync,
                    [it["frame"] for it in items],
                    cfgs,
                    model_process.INPUT_SIZE,
                    conf,
                )
                await self._result_queue.put((items, cfgs, dets, timing))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Inference error: {e!r}")
                self.shared_data.set_LIFE("error")
                await asyncio.sleep(0.1)

    async def _result_loop(self):
        while not self.stop_process:
            items, cfgs, dets, timing = await self._result_queue.get()
            try:
                for it, cfg, d in zip(items, cfgs, dets):
                    payload = model_process.build_payload(d, cfg)
                    await self.logic(it["frame"], it["cam_key"], payload)
                    self.update_process_fps(it["cam_key"])
                self._profile(timing)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Result handling error: {e!r}")
            finally:
                self._result_queue.task_done()

    # -------------------------------------------------------------- profiling
    @staticmethod
    def _new_prof():
        return {"t": time.time(), "batches": 0, "frames": 0, "pre": 0.0, "infer": 0.0, "post": 0.0, "speed": None}

    def _profile(self, timing):
        if not model_process.PROFILE:
            return
        p = self._prof
        p["batches"] += 1
        p["frames"] += timing["n"]
        p["pre"] += timing["pre_ms"]
        p["infer"] += timing["infer_ms"]
        p["post"] += timing["post_ms"]
        p["speed"] = timing.get("speed", p["speed"])
        p["shape"] = timing.get("frame_shape")
        now = time.time()
        if now - p["t"] >= 5.0 and p["batches"]:
            b = p["batches"]
            print(
                f"[Profile] batches/s={b/(now-p['t']):.1f} avg_batch={p['frames']/b:.2f} "
                f"pre={p['pre']/b:.1f}ms infer={p['infer']/b:.1f}ms post={p['post']/b:.1f}ms "
                f"frame_hw={p.get('shape')} ultralytics_speed={p['speed']}"
            )
            self._prof = self._new_prof()

    # -------------------------------------------------------------------- FPS
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

    # -------------------------------------------------------------------- run
    async def run(self):
        self.start_time = time.time()
        self._loop = asyncio.get_running_loop()
        self.frame_event = asyncio.Event()
        self._result_queue = asyncio.Queue(maxsize=2)  # back-pressure: GPU runs at most 2 ahead
        print("CCTV_AI System Running (Ultralytics batched mode)...")
        await self.alert_service.start()

        # Start readers first so RTSP connects while the engine loads / warms up.
        for cam in self.cameras:
            cam.on_frame = self._on_camera_frame
            cam.start()
        try:
            await model_process.ensure_loaded()
            self._tasks = [
                asyncio.create_task(self._inference_loop(), name="inference-loop"),
                asyncio.create_task(self._result_loop(), name="result-loop"),
            ]
            await asyncio.gather(*self._tasks)
        finally:
            self.cleanup()
            await self.alert_service.stop()

    # ----------------------------------------------------------------- alarms
    def _render_alarm_image(self, image, detections):
        """Draw boxes on a COPY (frames are shared with the camera) and JPEG-encode."""
        return self._encode_image(model_process.draw_detections(image, detections))

    async def alarm_detect(self, camid, detection_class, message, confidence, image, detections=None):
        image_base64_str = ""
        if image is not None:
            loop = asyncio.get_running_loop()
            image_base64_str = await loop.run_in_executor(
                model_process.cpu_executor,
                self._render_alarm_image,
                image,
                detections,
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
            detections = cam_data["detection"]

            if len(detections) < cam_data["machineCount"]:
                if self.miss_since[cam_id] is None:
                    self.miss_since[cam_id] = now
                if now - self.miss_since[cam_id] >= self.MISS_SECONDS:
                    if now - self.alarm_cooldown[cam_id] >= self.COOLDOWN_SEC:
                        self.alarm_cooldown[cam_id] = now
                        await self.alarm_detect(
                            cam_id, "machine missing", "can not detect machine", 0.0, image, detections
                        )
            else:
                self.miss_since[cam_id] = None
                class1_det = next((det for det in detections if det["class"] == 1), None)
                if class1_det and now - self.alarm_cooldown[cam_id] >= self.COOLDOWN_SEC:
                    self.alarm_cooldown[cam_id] = now
                    await self.alarm_detect(
                        cam_id, "alarm open", "open detected", class1_det["conf"], image, detections
                    )

        except Exception as e:
            print(f"Error in logic: {e}")

    def _encode_image(self, image):
        if self.jpeg is not None:
            try:
                jpeg_data = self.jpeg.encode(image, quality=80)
                return base64.b64encode(jpeg_data).decode('utf-8')
            except Exception:
                pass
        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8') if success else ""

    # ---------------------------------------------------------------- cleanup
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
            cam.on_frame = None
            try:
                cam.stop()
            except Exception as exc:
                print(f"Camera stop warning: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.shared_data.set_LIFE("active")
        print("CCTV_AI (Ultralytics) cleanup complete.")
