import threading
import cv2
import time


class Acquisition:
    def __init__(self, url, key="cam1",on_stream_end=None):
        print(f"Initializing camera {key} with URL: {url}")
        self.cap = None
        self.url = url
        self._image = None
        self._lock = threading.Lock()
        self.width = -1
        self.height = -1
        self.stop_process = False
        self.scale_factor = -1
        self.name = key
        self.on_stream_end = on_stream_end
        self.initial()

    @property
    def image(self):
        with self._lock:
            return self._image

    def initial(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.cap = None
            raise Exception("Failed to open video stream from " + self.url)

    def start(self):
        self.stop_process = False
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def stop(self):
        self.stop_process = True
        if hasattr(self, 't') and self.t.is_alive():
            self.t.join(timeout=2.0)

    def _update(self):
        while not self.stop_process:
            if self.cap is None:
                self._reconnect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.name}] Stream lost, reconnecting...")
                self.cap.release()
                self.cap = None
                continue

            with self._lock:
                self._image = frame
            time.sleep(0.01)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _reconnect(self):
        # self.on_stream_end()
        RETRY_DELAY = 10
        MAX_RETRY = 50
        self._reconnect_count = getattr(self, '_reconnect_count', 0) + 1
        
        print(f"[{self.name}] Attempting reconnect to {self.url} ({self._reconnect_count}/{MAX_RETRY})")
        
        if self._reconnect_count > MAX_RETRY:
            print(f"[{self.name}] Max retry exceeded, triggering cleanup...")
            self.stop_process = True
            if self.on_stream_end:
                self.on_stream_end()
            return
        
        time.sleep(RETRY_DELAY)
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                self._reconnect_count = 0  
                print(f"[{self.name}] Reconnected successfully")
            else:
                cap.release()
                print(f"[{self.name}] Reconnect failed, will retry...")
        except Exception as e:
            print(f"[{self.name}] Reconnect error: {e}")  
