"""Per-camera frame acquisition with isolated reconnect handling."""
import os
import platform
import threading
import time
import cv2


class Acquisition:
    def __init__(self, url, key="cam1", on_stream_end=None, on_frame=None):
        self.url = url
        self.name = key
        self.on_stream_end = on_stream_end  # compatibility only; no pipeline-wide shutdown
        # Called from the reader thread with the camera name after every new frame.
        # Must be cheap and thread-safe; the pipeline uses it to wake the inference loop.
        self.on_frame = on_frame
        self.cap = None
        self._image = None
        self._lock = threading.Lock()
        self.stop_process = False
        self.t = None
        self.frame_id = 0
        self.last_frame_at = None
        self.connected = False
        self._reconnect_count = 0

    @property
    def image(self):
        with self._lock:
            return None if self._image is None else self._image.copy()

    def latest_frame(self):
        """Return an image copy and its matching frame id atomically."""
        with self._lock:
            image = None if self._image is None else self._image.copy()
            return image, self.frame_id

    def take_new_frame(self, last_id):
        """Return (frame, frame_id) only if a frame newer than last_id exists.

        No copy is made: cap.read() allocates a fresh array per frame and the reader
        only ever REPLACES self._image, never modifies it in place. The caller owns
        the returned array for reading; it must copy before drawing on it.
        Returns (None, frame_id) when nothing new is available.
        """
        with self._lock:
            if self._image is None or self.frame_id == last_id:
                return None, self.frame_id
            return self._image, self.frame_id

    def initial(self):
        if self.cap is not None:
            return
        is_gstreamer = False

        if isinstance(self.url, str) and self.url.startswith("rtsp://"):
            # กำหนด protocols=tcp latency=200 เพื่อความเสถียร
            pipeline = (
                f"rtspsrc location={self.url} protocols=tcp latency=200 ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            is_gstreamer = True

        elif isinstance(self.url, str) and self.url.endswith(".mp4"):
            abs_path = os.path.abspath(self.url)
            cap = None
            if platform.system() == "Linux" and platform.machine() == "aarch64":
                pipeline = (
                    f'filesrc location="{abs_path}" ! '
                    "qtdemux ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! "
                    "appsink drop=true sync=false max-buffers=1"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                is_gstreamer = True

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(abs_path)
                is_gstreamer = False
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            is_gstreamer = False

        # เรียก set buffer size เฉพาะตอนที่ไม่ใช่ GStreamer เท่านั้น
        if not is_gstreamer:
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

    # Platform camera-adapter contract. These methods reuse this Acquisition
    # instance; they do not create another RTSP reader.
    def connection(self, width=1920, height=1080, fps=15):
        self.start()
        deadline = time.monotonic() + 0.5
        while not self.connected and self.t is not None and self.t.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        return bool(self.connected)

    def is_connected(self):
        return bool(self.connected)

    def reconnect(self):
        # The reader already performs exponential-backoff reconnects while its
        # thread is alive. Do not reset that backoff from the health monitor.
        if self.t is not None and self.t.is_alive():
            return bool(self.connected)
        self.start()
        deadline = time.monotonic() + 0.5
        while not self.connected and self.t is not None and self.t.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        return bool(self.connected)

    def disconnect(self):
        self.stop()

    def get_latest_frame(self):
        frame, _frame_id = self.latest_frame()
        return frame

    def stop(self):
        # The reader thread must be the only thread that calls read()/release() on
        # this VideoCapture. Releasing it concurrently with cap.read() can trigger
        # FFmpeg/libavcodec assertions such as "fctx->async_lock failed".
        self.stop_process = True
        self.connected = False
        if self.t is not None and self.t.is_alive() and threading.current_thread() is not self.t:
            self.t.join(timeout=5.0)
            if self.t.is_alive():
                print(
                    f"[{self.name}] Reader has not stopped yet; leaving VideoCapture "
                    "release to the reader thread to avoid a decoder race."
                )

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
                self.last_frame_at = time.time()
            self.connected = True
            callback = self.on_frame
            if callback is not None:
                try:
                    callback(self.name)
                except Exception as exc:
                    print(f"[{self.name}] on_frame callback error: {exc}")
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
