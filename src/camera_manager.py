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
from src.ee_cctv_ai.alarm_rules import AlarmRuleEngine
from src.ee_cctv_ai.alarm_renderer import draw_detections as render_detections
from src.ee_cctv_ai.config_store import load_config

try:  # optional: CI / dev machines may not have torch or libturbojpeg
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    from turbojpeg import TurboJPEG
except Exception:  # pragma: no cover
    TurboJPEG = None


class CCTV_AI:

    def __init__(self, rtsp_urls, shared_data, API_KEY, nodeJS_URL):
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
        self.MISS_SECONDS = float(os.getenv("EE_CCTV_MISS_SECONDS", 2))
        self.COOLDOWN_SEC = 5
        self.rule_engine = AlarmRuleEngine(
            getattr(self, "app_config", {}),
            legacy_miss_seconds=self.MISS_SECONDS,
            legacy_cooldown_seconds=self.COOLDOWN_SEC,
        )
        self.start_time = None


        self.target_fps = max(1.0, float(os.getenv("EE_CCTV_TARGET_FPS", 10)))
        self.min_frame_interval = 1.0 / self.target_fps
        self._last_collect_time = {n: 0.0 for n in names}

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


        self._pending = {}
        self._last_ids = {n: -1 for n in names}
        self._rr = 0  
        self._loop = None
        self.frame_event = None
        self._prep_queue = None
        self._result_queue = None
        self._prof = self._new_prof()

        self.alert_service = AlertService(self.nodeJS_URL, self.API_KEY)

    
        self.cameras = [Acquisition(rtsp_urls[i], key=names[i]) for i in range(len(rtsp_urls))]

        self.jpeg = None
        if TurboJPEG is not None:
            try:
                self.jpeg = TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
            except Exception:
                try:
                    self.jpeg = TurboJPEG()
                except Exception:
                    self.jpeg = None

        atexit.register(self.cleanup)

    # ------------------------------------------ config -------------------------------------------------------
    def read_config(self):
        try:
            config_data = load_config(PROJECT_ROOT / "config" / "config.json")
            self.app_config = config_data
            self.shared_data.set_config(config_data.get("cameras", []))
            self.shared_data.set_confidence_threshold(config_data.get("confidence_threshold", 0.5))
            print("Unified config loaded successfully.")
        except Exception as e:
            self.app_config = {}
            print(f"Config error (config.json): {e}")
            self.shared_data.set_config([{"name": f"cam{i+1}", "roi": None} for i in range(len(self.urls))])
            self.shared_data.set_confidence_threshold(0.5)

    # ---------------------------------------------- frame intake -------------------------------------------------------
    def _on_camera_frame(self, cam_key):
        self.update_capture_fps(cam_key)  
        #if (time.time() - self._last_collect_time.get(cam_key, 0.0)) < self.min_frame_interval:
        #    return
        loop = self._loop
        event = self.frame_event
        if loop is None or event is None or event.is_set():
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:  # loop already closed during shutdown
            pass

    def _collect_new_frames(self):
        now = time.time()
        for cam in self.cameras:
            #if (now - self._last_collect_time[cam.name]) < self.min_frame_interval:
            #    continue
            frame, fid = cam.take_new_frame(self._last_ids[cam.name])
            if frame is not None:
                self._last_ids[cam.name] = fid
                self._pending[cam.name] = frame  # overwrite -> stale frame dropped
                self._last_collect_time[cam.name] = now

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

    # -------------------------------------------------------------- inference -------------------------------------------------------
    async def _prepare_loop(self):
        loop = self._loop
        print("Started batched inference pipeline")
        while not self.stop_process:
            try:
                await asyncio.wait_for(self.frame_event.wait(), timeout=self.min_frame_interval)
                self.frame_event.clear()
            except asyncio.TimeoutError:
                pass

            self._collect_new_frames()
            items = self._take_batch()
            if not items:
                continue
            if self._pending:          # more cameras waiting than fit in this batch
                self.frame_event.set()

            try:
                cfgs = [self.shared_data.get_config_by_cam(it["cam_key"]) for it in items]
                prepared = await loop.run_in_executor(
                    model_process.prep_executor,
                    model_process.prepare_batch_sync,
                    [it["frame"] for it in items],
                    cfgs,
                    model_process.INPUT_SIZE,
                )
                await self._prep_queue.put((items, cfgs, prepared))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Prepare error: {e!r}")
                self.shared_data.set_LIFE("error")
                await asyncio.sleep(0.1)

    async def _infer_loop(self):
        loop = self._loop
        while not self.stop_process:
            items, cfgs, prepared = await self._prep_queue.get()
            try:
                conf = self.shared_data.get_confidence_threshold()
                dets, timing = await loop.run_in_executor(
                    model_process.gpu_executor,
                    model_process.infer_batch_sync,
                    prepared,
                    conf,
                )
                merged = dict(prepared.get("timing", {}))
                merged.update(timing)
                await self._result_queue.put((items, cfgs, dets, merged))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Inference error: {e!r}")
                self.shared_data.set_LIFE("error")
                await asyncio.sleep(0.1)
            finally:
                self._prep_queue.task_done()

    async def _result_loop(self):
        while not self.stop_process:
            items, cfgs, batch_dets, timing = await self._result_queue.get()
            try:
                for it, cfg, cam_models_det in zip(items, cfgs, batch_dets):
                    # cam_models_det = {"model_1": dets_array_1, "model_2": dets_array_2, ...}
                    if isinstance(cam_models_det, dict):
                        models_payload = {
                            m_name: model_process.build_payload(
                                d, cfg, model_process.model_class_names(m_name)
                            )
                            for m_name, d in cam_models_det.items()
                        }
                    else:
                        # Fallback กรณีส่งกลับมาเป็น array เดี่ยว
                        models_payload = {"model_1": model_process.build_payload(cam_models_det, cfg)}

                    await self.logic(it["frame"], it["cam_key"], models_payload)
                    self.update_process_fps(it["cam_key"])
                self._profile(timing)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Result handling error: {e!r}")
            finally:
                self._result_queue.task_done()

    # --------------------------------- profiling------------------------------------
    @staticmethod
    def _new_prof():
        return {"t": time.time(), "batches": 0, "frames": 0, "pre": 0.0, "infer": 0.0,
                "shape": None, "path": None, "speed": None}

    def _profile(self, timing):
        if not model_process.PROFILE:
            return
        p = self._prof
        p["batches"] += 1
        p["frames"] += timing.get("n", 0)
        p["pre"] += timing.get("pre_ms", 0.0)
        p["infer"] += timing.get("infer_ms", 0.0)
        p["shape"] = timing.get("frame_shape", p["shape"])
        p["path"] = timing.get("path", p["path"])
        p["speed"] = timing.get("speed", p["speed"])
        now = time.time()
        if now - p["t"] >= 5.0 and p["batches"]:
            b = p["batches"]
            print(
                f"[Profile] batches/s={b/(now-p['t']):.1f} avg_batch={p['frames']/b:.2f} "
                f"pre={p['pre']/b:.1f}ms (stage A) infer={p['infer']/b:.1f}ms (stage B) "
                f"path={p['path']} frame_hw={p['shape']} ultralytics_speed={p['speed']}"
            )
            self._prof = self._new_prof()

    # ------------------------------ FPS ------------------------------------------
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

    # ------------------------------ run ---------------------------
    async def run(self):
        self.start_time = time.time()
        self._loop = asyncio.get_running_loop()
        self.frame_event = asyncio.Event()
        self._prep_queue = asyncio.Queue(maxsize=1)
        self._result_queue = asyncio.Queue(maxsize=2)

        print("Waiting for AI Models to load...")
        await model_process.ensure_loaded()
        print("All models loaded successfully!")

        print(
            f"CCTV_AI System Running (cameras={len(self.cameras)}, "
            f"models={len(model_process.MODEL_PATHS)})..."
        )
        await self.alert_service.start()

        for cam in self.cameras:
            cam.on_frame = self._on_camera_frame
            cam.start()
        try:
            self._tasks = [
                asyncio.create_task(self._prepare_loop(), name="prepare-loop"),
                asyncio.create_task(self._infer_loop(), name="infer-loop"),
                asyncio.create_task(self._result_loop(), name="result-loop"),
            ]
            await asyncio.gather(*self._tasks)
        finally:
            self.cleanup()
            await self.alert_service.stop()

    # --------------------------------- alarms ---------------------------------
    def _render_alarm_image(self, image, detections, render_config=None):
        """Render on a copy so the shared RTSP frame is never mutated."""
        return self._encode_image(render_detections(image, detections, render_config))

    async def alarm_detect(
        self, camid, detection_class, message, confidence, image, detections=None,
        *, model_detections=None, image_mode="combined", render_config=None,
        model_id=None, rule_id=None,
    ):
        image_base64_str = ""
        model_image_base64_str = ""
        if image is not None:
            loop = asyncio.get_running_loop()
            if image_mode in {"combined", "both"}:
                image_base64_str = await loop.run_in_executor(
                    model_process.cpu_executor,
                    self._render_alarm_image, image, detections, render_config,
                )
            if image_mode == "model":
                image_base64_str = await loop.run_in_executor(
                    model_process.cpu_executor,
                    self._render_alarm_image, image, model_detections or detections, render_config,
                )
            elif image_mode == "both":
                model_image_base64_str = await loop.run_in_executor(
                    model_process.cpu_executor,
                    self._render_alarm_image, image, model_detections or [], render_config,
                )

        payload = {
            "alarmType": detection_class,
            "message": message,
            "cameraId": f"{camid[:3].upper()}-0{camid[3:]}",
            "confidence": confidence * 100,
            "imageBase64": f"data:image/jpeg;base64,{image_base64_str}" if image_base64_str else None,
        }
        # Optional additive metadata; legacy fields above remain unchanged.
        if model_id is not None:
            payload["modelId"] = model_id
        if rule_id is not None:
            payload["ruleId"] = rule_id
        if image_mode == "both":
            payload["modelImageBase64"] = (
                f"data:image/jpeg;base64,{model_image_base64_str}" if model_image_base64_str else None
            )

        # Queue the alert and return immediately; delivery must not block detection.
        await self.alert_service.enqueue(payload)
        return payload

    async def logic(self, image, cam_id, models_payload):
        """Evaluate all configured model/rule pairs without hard-coded model slots."""
        try:
            if not models_payload:
                return
            # Accept the former single-model payload shape as well as the new model-id map.
            if isinstance(models_payload, dict) and "detection" in models_payload:
                models_payload = {"model_1": models_payload}
            self.rule_engine.legacy_miss_seconds = self.MISS_SECONDS
            self.rule_engine.legacy_cooldown_seconds = self.COOLDOWN_SEC
            events = self.rule_engine.evaluate(cam_id, models_payload)
            if not self.rule_engine.config.get("models"):
                self.miss_since[cam_id] = self.rule_engine._since.get(
                    (cam_id, "model_1", "legacy_machine_missing")
                )
            for event in events:
                await self.alarm_detect(
                    cam_id,
                    event["alarm_type"],
                    event["message"],
                    event["confidence"],
                    image,
                    event["detections"],
                    model_detections=event["model_detections"],
                    image_mode=event["image_mode"],
                    render_config=event.get("render"),
                    model_id=event["model_id"],
                    rule_id=event["rule_id"],
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

    # ------------------------------ cleanup ------------------------------------------
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

# Backward-compatible public name used by earlier callers/tests.
CCTV_AI_Ultralytics = CCTV_AI
