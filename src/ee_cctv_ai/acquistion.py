import threading
import cv2
import time
import os
import platform

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
        target_size = 640 

        if isinstance(self.url, str) and self.url.startswith('rtsp://'):
            pipeline = (
                f"rtspsrc location={self.url} latency=0 ! "
                f"rtph264depay ! h264parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "  
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false max-buffers=1"    
            )
            # print(f"[{self.name}] Connecting to RTSP via GStreamer NVDEC + VIC...")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
        elif isinstance(self.url, str) and self.url.endswith('.mp4'):
            abs_path = os.path.abspath(self.url)
    
            if not os.path.exists(abs_path):
                print(f"[{self.name}] ERROR: File NOT FOUND at: {abs_path}")
            else:
                print(f"[{self.name}] File exists at: {abs_path}")
                
            if platform.system() == 'Linux' and platform.machine() == 'aarch64':
                pipeline = (
                    f"filesrc location=\"{abs_path}\" ! "
                    f"qtdemux name=d d.video_0 ! queue ! h264parse ! nvv4l2decoder ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink drop=true sync=false max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if self.cap is None or not self.cap.isOpened():
                print(f"[{self.name}] Warning: GStreamer pipeline failed or not supported in this OpenCV build. Falling back to default VideoCapture...")
                self.cap = cv2.VideoCapture(abs_path)
        else:
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
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue

            with self._lock:
                # เก็บเฟรมสดใหม่ล่าสุดไว้เสมอ
                self._image = frame

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
            self.initial()  
            if self.cap and self.cap.isOpened():
                self._reconnect_count = 0
                print(f"[{self.name}] Reconnected successfully")
        except Exception as e:
            print(f"[{self.name}] Reconnect error: {e}") 
