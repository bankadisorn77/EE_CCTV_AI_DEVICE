import torch
import cv2
import asyncio
import json
import json
from tritonclient.grpc import aio as grpc
from src.ee_cctv_ai import model_process
from src.ee_cctv_ai.acquistion import Acquisition
import zmq
import atexit
import time
# from shere_data import data



class CCTV_AI:
    def __init__(self, rtsp_urls,shared_data):
        self.stop_process = False
        self.shared_data = shared_data 
        self.read_config()
        self.config_cam = self.shared_data.get_config()
        self.confidence = self.shared_data.get_confidence_threshold()
        #  ZMQ Publisher Setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.set(zmq.SNDHWM, 5)
        self.zmq_socket.bind("tcp://0.0.0.0:5555")
        self.start_time = None

        self.cameras = [
            Acquisition(rtsp_urls[i], key=f"cam{i+1}",on_stream_end=self.cleanup)
            for i in range(len(rtsp_urls))
        ]      
        atexit.register(self.cleanup)

    def read_config(self):
        try:
            config_path = "/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/camera_config.json"
            with open(config_path, 'r') as f:
                self.shared_data.set_config(json.load(f))
        except Exception as e:
            print(f"Config error can not load file : {e}")
            config_cam = [{"name": f"cam{i+1}", "roi": None} for i in range(4)]
            self.shared_data.set_config(config_cam)
        try:
            confidence_path = "/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/confident.json"
            with open(confidence_path, 'r') as f:
                self.shared_data.set_confidence_threshold(json.load(f)["confidence"])
        except Exception as e:
            print(f"Config error can not load confidence file : {e}")
            self.shared_data.set_confidence_threshold(0.5)

    async def run(self):
        self.shared_data.set_active_device(1)
        URL = 'localhost:8001' # gRPC port
        try:
            tritonurl = grpc.InferenceServerClient(URL)
        except Exception as e:
            print(f"Error connecting to Triton Inference Server: {e}")
            self.cleanup()
            return

        for cam in self.cameras:
            cam.start()
        print("Waiting for cameras to initialize...")

        while any(cam.image is None for cam in self.cameras):
            await asyncio.sleep(0.1)

        print("All cameras ready!")
        try:
            self.start_time = time.perf_counter()
            while not self.stop_process:
                if all(cam.image is None for cam in self.cameras):
                    self.cleanup()
                    return
                self.config_cam = self.shared_data.get_config() 
                self.confidence = self.shared_data.get_confidence_threshold()
                tasks = [model_process.predict_model(cam.image, tritonurl, next((item for item in self.config_cam if item["name"] == cam.name), None), size=640, conf=self.confidence)for cam in self.cameras if cam.image is not None]
                
                # inference       
                if tasks:
                    results = await asyncio.gather(*tasks)
                    # check results error
                    if results is None:
                        print("Error occurred while processing results.")
                        self.shared_data.set_LIFE('error')
                        return
                    for i, (res, detection_class,) in enumerate(results):
                        cam_id = f"cam{i+1}"
                        # print(f"form camera manager:: Detections: {detection_class}")
                        payload_json = json.dumps(detection_class)
                        success, buffer = cv2.imencode('.jpg', res, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        if success:
                            if self.zmq_socket:
                                self.zmq_socket.send_multipart([
                                    cam_id.encode('utf-8'),         
                                    payload_json.encode('utf-8'),   
                                    buffer.tobytes()                
                                ])
                            del buffer
                    await asyncio.sleep(1)
                            
                await asyncio.sleep(0.001)
        except Exception as e:
            self.shared_data.set_LIFE('error')
            print(f"Exception in run loop: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        end_time = time.perf_counter()
        duration = end_time - self.start_time if self.start_time else 0
        print(f"Total runtime duration: {duration:.2f} seconds")
        self.shared_data.set_active_device(0)
        print("Cleaning up CCTV_AI resources...")
        # self.shared_data.set_LIFE('inactive')
        # stop camera threads and release capture devices
        for cam in getattr(self, 'cameras', []):
            try:
                cam.stop()
            except Exception:
                cam.stop_process = True
        # close zmq handles
        if hasattr(self, 'zmq_socket') and self.zmq_socket:
            try:
                self.zmq_socket.close(linger=0)
            except Exception:
                pass
            self.zmq_socket = None
        if hasattr(self, 'zmq_context') and self.zmq_context:
            try:
                self.zmq_context.term()
            except Exception:
                pass
            self.zmq_context = None
        # destroy opencv windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # free gpu memory if torch is present
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("CCTV_AI cleanup complete.")

    


# async def start(cctv_ai):
#     await asyncio.gather(
#         cctv_ai.run(),
#     )


# if __name__ == "__main__":
#     sheredata = data.Data()
#     rtsp_urls = [
#         "images/video_test/D01.mp4",
#         "images/video_test/D02.mp4",
#         "images/video_test/D03.mp4",
#         "images/video_test/D04.mp4"
#     ]
#     cctv_ai = CCTV_AI(rtsp_urls,sheredata)
    
#     try:
#         asyncio.run(start(cctv_ai))
#     except KeyboardInterrupt:
#         print("Stopping system...")
#     finally:
#         cctv_ai.cleanup()