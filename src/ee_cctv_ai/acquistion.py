"""Per-camera frame acquisition with isolated reconnect handling."""
import os
import platform
import threading
import time
import cv2


class Acquisition:
    def __init__(self, url, key="cam1", on_stream_end=None):
        self.url = url
        self.name = key
        self.on_stream_end = on_stream_end  # compatibility only; no pipeline-wide shutdown
        self.cap = None
        self._image = None
        self._lock = threading.Lock()
        self.stop_process = False
        self.t = None
        self.frame_id = 0
        self.connected = False
        self._reconnect_count = 0

    @property
    def image(self):
        with self._lock:
            return None if self._image is None else self._image.copy()

    def initial(self):
        if self.cap is not None:
            return
        if isinstance(self.url, str) and self.url.startswith("rtsp://"):
            pipeline = (
                f"rtspsrc location={self.url} latency=0 ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif isinstance(self.url, str) and self.url.endswith(".mp4"):
            abs_path = os.path.abspath(self.url)
            cap = None
            if platform.system() == "Linux" and platform.machine() == "aarch64":
                pipeline = (
                    f'filesrc location="{abs_path}" ! '
                    "qtdemux name=d d.video_0 ! queue ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! "
                    "appsink drop=true sync=false max-buffers=1"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(abs_path)
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video stream for {self.name}")
        self.cap = cap
        self.connected = True

    def start(self):
        if self.t is not None and self.t.is_alive():
            return
        self.stop_process = False
        self.t = threading.Thread(target=self._update, name=f"{self.name}-reader", daemon=True)
        self.t.start()

    def stop(self):
        self.stop_process = True
        cap = self.cap
        self.cap = None
        self.connected = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self.t is not None and self.t.is_alive() and threading.current_thread() is not self.t:
            self.t.join(timeout=2.0)

    def _update(self):
        while not self.stop_process:
            if self.cap is None:
                try:
                    self.initial()
                    self._reconnect_count = 0
                    print(f"[{self.name}] Camera connected")
                except Exception as exc:
                    self.connected = False
                    self._reconnect_count += 1
                    delay = min(10.0, 0.5 * (2 ** min(self._reconnect_count - 1, 5)))
                    print(f"[{self.name}] Connect failed ({self._reconnect_count}); retrying in {delay:.1f}s: {exc}")
                    self._wait_or_stop(delay)
                    continue
            cap = self.cap
            if cap is None:
                continue
            try:
                ret, frame = cap.read()
            except Exception as exc:
                print(f"[{self.name}] Frame read failed: {exc}")
                ret, frame = False, None
            if not ret or frame is None:
                self.connected = False
                try:
                    cap.release()
                except Exception:
                    pass
                if self.cap is cap:
                    self.cap = None
                self._wait_or_stop(0.5)
                continue
            with self._lock:
                self._image = frame
                self.frame_id += 1
            self.connected = True
        cap = self.cap
        self.cap = None
        self.connected = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _wait_or_stop(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stop_process and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
