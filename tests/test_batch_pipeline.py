"""Pipeline tests with fake cameras and a fake model (no GPU, no RTSP)."""
import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src import camera_manager
from src.ee_cctv_ai.acquistion import Acquisition


class FakeShared:
    def __init__(self, count=0):
        self.life = "active"
        self.count = count

    def set_config(self, c): pass
    def set_confidence_threshold(self, c): pass
    def set_LIFE(self, v): self.life = v
    def get_confidence_threshold(self): return 0.5
    def get_config_by_cam(self, name): return {"name": name, "machineCount": self.count, "roi": None}


class FakeAlert:
    def __init__(self, *a, **k): self.sent = []
    async def start(self): pass
    async def stop(self): pass
    async def enqueue(self, payload): self.sent.append(payload)


class FakeCam:
    """Mimics Acquisition without threads: tests push frames by hand."""
    def __init__(self, url, key="cam1", **kw):
        self.name, self.on_frame = key, None
        self.frame_id, self._image, self.started = 0, None, False
    def start(self): self.started = True
    def stop(self): self.started = False
    def push(self, value=1):
        self._image = np.full((4, 4, 3), value, np.uint8)
        self.frame_id += 1
        if self.on_frame: self.on_frame(self.name)
    def take_new_frame(self, last_id):
        if self._image is None or self.frame_id == last_id:
            return None, self.frame_id
        return self._image, self.frame_id


def make_pipeline(n=4, shared=None):
    with patch.object(camera_manager, "Acquisition", FakeCam), \
         patch.object(camera_manager, "AlertService", FakeAlert), \
         patch.object(camera_manager, "load_config", side_effect=RuntimeError("no file")):
        return camera_manager.CCTV_AI_Ultralytics(
            ["x"] * n, shared or FakeShared(), "key", "http://srv")


class Recorder:
    """Fake predict_batch_sync that records each batch."""
    def __init__(self, delay=0.0, boxes=None):
        self.batches, self.delay, self.boxes = [], delay, boxes
    def __call__(self, frames, cfgs, size, conf):
        self.batches.append([int(f[0, 0, 0]) for f in frames])
        if self.delay: time.sleep(self.delay)
        d = [self.boxes if self.boxes is not None else np.zeros((0, 6), np.float32)] * len(frames)
        return d, {"n": len(frames), "pre_ms": 0, "infer_ms": 0, "post_ms": 0}


async def _noop(): return None


async def run_for(pipe, seconds, driver):
    task = asyncio.create_task(pipe.run())
    await asyncio.sleep(0.05)
    await driver()
    await asyncio.sleep(seconds)
    pipe.cleanup()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


class BatchPipelineTests(unittest.TestCase):
    def _patches(self, rec, max_batch=4):
        return (patch.object(camera_manager.model_process, "predict_batch_sync", rec),
                patch.object(camera_manager.model_process, "ensure_loaded", _noop),
                patch.object(camera_manager.model_process, "max_batch", lambda: max_batch))

    def test_simultaneous_frames_form_one_batch(self):
        pipe, rec = make_pipeline(), Recorder()
        async def go():
            async def drive():
                for i, cam in enumerate(pipe.cameras):
                    cam.push(i + 1)
            await run_for(pipe, 0.3, drive)
        p1, p2, p3 = self._patches(rec)
        with p1, p2, p3:
            asyncio.run(go())
        flat = sorted(v for b in rec.batches for v in b)
        self.assertEqual(flat, [1, 2, 3, 4])          # every camera processed exactly once
        self.assertLessEqual(len(rec.batches), 2)     # (first wake-up may only see some)

    def test_slow_gpu_drops_stale_frames_and_keeps_newest(self):
        pipe, rec = make_pipeline(1), Recorder(delay=0.1)
        async def go():
            async def drive():
                cam = pipe.cameras[0]
                for v in range(1, 11):    # 10 frames arrive while the first batch runs
                    cam.push(v); await asyncio.sleep(0.01)
            await run_for(pipe, 0.5, drive)
        p1, p2, p3 = self._patches(rec)
        with p1, p2, p3:
            asyncio.run(go())
        seen = [v for b in rec.batches for v in b]
        self.assertLess(len(seen), 10)                # some were dropped on purpose
        self.assertEqual(seen, sorted(seen))          # order preserved
        self.assertEqual(seen[-1], 10)                # newest frame is never lost

    def test_batch_limit_one_serves_cameras_round_robin(self):
        pipe, rec = make_pipeline(4), Recorder()
        async def go():
            async def drive():
                for i, cam in enumerate(pipe.cameras):
                    cam.push(i + 1)
            await run_for(pipe, 0.4, drive)
        p1, p2, p3 = self._patches(rec, max_batch=1)
        with p1, p2, p3:
            asyncio.run(go())
        self.assertTrue(all(len(b) == 1 for b in rec.batches))
        self.assertEqual(sorted(v for b in rec.batches for v in b), [1, 2, 3, 4])

    def test_capture_fps_counts_every_frame_even_if_inference_drops(self):
        pipe, rec = make_pipeline(1), Recorder(delay=0.05)
        async def go():
            async def drive():
                for v in range(1, 31):
                    pipe.cameras[0].push(v); await asyncio.sleep(0.005)
            await run_for(pipe, 0.3, drive)
        p1, p2, p3 = self._patches(rec)
        with p1, p2, p3:
            asyncio.run(go())
        stats = pipe.fps_stats["cam1"]
        self.assertGreater(stats["capture_count"], stats["process_count"])

    def test_missing_machine_alarm_is_time_based_and_draws_on_copy(self):
        shared = FakeShared(count=2)
        pipe = make_pipeline(1, shared)
        pipe.MISS_SECONDS = 0.2
        frame = np.zeros((20, 20, 3), np.uint8)
        det = {"detection": [{"class": 0, "conf": 0.9, "box": (1, 1, 10, 10)}], "machineCount": 2}
        async def go():
            await pipe.logic(frame, "cam1", det)       # starts the miss timer
            self.assertEqual(pipe.alert_service.sent, [])
            await asyncio.sleep(0.25)
            await pipe.logic(frame, "cam1", det)       # persisted long enough
            self.assertEqual(len(pipe.alert_service.sent), 1)
            self.assertEqual(pipe.alert_service.sent[0]["alarmType"], "machine missing")
            await pipe.logic(frame, "cam1", det)       # cooldown suppresses a second alarm
            self.assertEqual(len(pipe.alert_service.sent), 1)
        asyncio.run(go())
        self.assertEqual(int(frame.sum()), 0)          # shared frame never drawn on

    def test_open_alarm_and_reset_of_miss_timer(self):
        pipe = make_pipeline(1, FakeShared(count=1))
        frame = np.zeros((20, 20, 3), np.uint8)
        pipe.miss_since["cam1"] = time.time() - 100
        ok = {"detection": [{"class": 1, "conf": 0.9, "box": (1, 1, 5, 5)}], "machineCount": 1}
        asyncio.run(pipe.logic(frame, "cam1", ok))
        self.assertIsNone(pipe.miss_since["cam1"])
        self.assertEqual(pipe.alert_service.sent[0]["alarmType"], "alarm open")


class AcquisitionIntakeTests(unittest.TestCase):
    def test_take_new_frame_semantics_and_no_copy(self):
        cam = Acquisition("rtsp://x", key="cam1")
        self.assertEqual(cam.take_new_frame(-1), (None, 0))
        frame = np.zeros((2, 2, 3), np.uint8)
        cam._image, cam.frame_id = frame, 5
        got, fid = cam.take_new_frame(4)
        self.assertIs(got, frame)                      # no copy
        self.assertEqual(fid, 5)
        self.assertEqual(cam.take_new_frame(5), (None, 5))   # nothing new

    def test_on_frame_callback_fires_from_reader_loop(self):
        calls = []
        cam = Acquisition("rtsp://x", key="camX", on_frame=calls.append)

        class Cap:
            def __init__(self): self.n = 0
            def isOpened(self): return True
            def read(self):
                self.n += 1
                if self.n > 3:
                    cam.stop_process = True
                    return False, None
                return True, np.zeros((2, 2, 3), np.uint8)
            def release(self): pass
        cam.cap = Cap()
        cam._update()
        self.assertEqual(calls, ["camX"] * 3)
        self.assertEqual(cam.frame_id, 3)


if __name__ == "__main__":
    unittest.main()
