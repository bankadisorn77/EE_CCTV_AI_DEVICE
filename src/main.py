#!/usr/bin/python3
import asyncio
import base64
from time import time
from fastapi import FastAPI, Response
import socketio
import uvicorn
from contextlib import asynccontextmanager
import httpx
import json
import psutil
from src.camera_manager import CCTV_AI
from src.shere_data import data
from pydantic import BaseModel
from typing import List
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



# -----------------------------Global variables----------------------------------------------
sio = socketio.AsyncClient()
API_KEY = None
shared_data = data.Data()
LIFE = None
nodeJS_URL = 'http://192.168.1.2:5000'
cctv_task = None
cctv_ai = None
class AIcontoll(BaseModel):
    acction: str
    roi: List[dict] = None
    confidence_threshold: float = None
# rtsp_urls = [
#     "rtsp://192.168.1.1:8554/cam1",
#     "rtsp://192.168.1.1:8554/cam2",
#     "rtsp://192.168.1.1:8554/cam3",
#     "rtsp://192.168.1.1:8554/cam4"
# ]
rtsp_urls = [
    "images/video_test/cam1_h264.mp4",
    "images/video_test/cam2_h264.mp4",
    "images/video_test/cam3_h264.mp4",
    "images/video_test/cam4_h264.mp4"
]

#  -----------------------------FastAPI app and lifespan----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cctv_ai, cctv_task,shared_data
    task1 = asyncio.create_task(register_device())
    await task1  
    # await sio.connect(nodeJS_URL)
    asyncio.create_task(connect_socketio())
    task2 = asyncio.create_task(send_status('stanby'))
    # cctv_task = asyncio.create_task(run_cctv_ai())
    # asyncio.create_task(send_status())
    asyncio.create_task(send_status('now'))
    yield
    task1.cancel()
    task2.cancel()
    if cctv_ai:
        cctv_ai = None
    if cctv_task:
        cctv_task.cancel()
    await sio.disconnect()
   


app = FastAPI(lifespan=lifespan)

# -------------------------------------Endpoints-----------------------------------------------

@app.post("/control")
async def control_ai(control: AIcontoll):
    global cctv_ai, cctv_task
    roi = control.roi
    confidence_threshold = control.confidence_threshold
    with open ('./config/camera_config.json','w') as f:
        json.dump(roi, f, indent=4)
    shared_data.set_config(roi)
    with open ('./config/confident.json','w') as f:
        json.dump({"confidence": confidence_threshold}, f, indent=4)
    shared_data.set_confidence_threshold(confidence_threshold)
    print(f"Control AI: action={control.acction}, roi={roi}, confidence_threshold={confidence_threshold}")
    if control.acction == "start":
        if cctv_ai is not None:
            return {"message": "AI is already running"}
        else:
            cctv_task = asyncio.create_task(run_cctv_ai())
        return {"message": "AI started"}
    elif control.acction == "stop":
        if cctv_ai is not None:
            cctv_ai.cleanup()
            cctv_ai = None
        return {"message": "AI stopped"}
    await send_status('now')
    
@app.get("/initial-frames")
async def get_initial_frames():
    global rtsp_urls
    image_base64_frames = init_cctv_ai_frame(rtsp_urls)
    print(image_base64_frames['cam1'])
    return image_base64_frames

# ----------------------------------------------socketio events-----------------------------------------------------------
@sio.event
async def connect():
    print("SocketIO connected")
    await sio.emit('register-device-status', {'apiKey': API_KEY})
@sio.event
async def disconnect():
    print("SocketIO disconnected")
    
# ----------------------------------------------function-----------------------------------------------------------

async def connect_socketio():
    global API_KEY
    while True:
        try:
            if not sio.connected:
                print(f" Attempting to connect SocketIO to {nodeJS_URL}...")
                await sio.connect(nodeJS_URL)
                print("SocketIO connected successfully!")
                break 
        except Exception as e:
            print(f"SocketIO connection failed: {e}")
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)

# send status to nodeJS via SocketIO
async def send_status(time='stanby'):
    global API_KEY ,LIFE
    if time == 'now':
        LIFE = shared_data.get_LIFE()
        if API_KEY:
            try:
                await sio.emit('heartbeat', get_jetson_stats(LIFE))
            except Exception as e:
                print(f"Failed to send status via SocketIO: {e}")
    else:
        while True:
            LIFE = shared_data.get_LIFE()
            if API_KEY:
                try:
                    await sio.emit('heartbeat', get_jetson_stats(LIFE))
                except Exception as e:
                    print(f"Failed to send status via SocketIO: {e}")
            LIFE = 'inactive'
            await asyncio.sleep(5)
       

def get_jetson_stats(status):
        def read_temp(zone_path):
            try:
                with open(zone_path, "r") as f:
                    return float(f.read().strip()) / 1000.0
            except:
                return 0.0
        cpu_temp = read_temp("/sys/class/thermal/thermal_zone0/temp") 
        gpu_temp = read_temp("/sys/class/thermal/thermal_zone1/temp")
        mem = psutil.virtual_memory()
        mem_used = mem.used // (1024 * 1024)
        mem_free = mem.available // (1024 * 1024)

        return {
            "runningStatus": status,
            "cpuTemp": cpu_temp,
            "gpuTemp": gpu_temp,
            "memoryFree": mem_free,
            "memoryUsed": mem_used,
            "aiVersion": "v1.12.3", 
            "activeDevice": shared_data.get_active_device()
        }

async def register_device():
    global API_KEY, rtsp_urls
    image_base64_frames = init_cctv_ai_frame(rtsp_urls)
    async def reg():
            global API_KEY
            print("Device config missing mcId or apiKey")
            URL = f'{nodeJS_URL}/api/device/register'
            data ={
                'mac':device_config.get('mac'),
                'ip':device_config.get('ip'),
                'frame': image_base64_frames,
            }
            async with httpx.AsyncClient() as requests:
                try:
                    response = await requests.post(url=URL,json=data)
                    if response.status_code == 200:
                        res_data = response.json()
                        # device_config['mcId'] = res_data.get('mcId')
                        device_config['apiKey'] = res_data.get('apiKey')
                        API_KEY = res_data.get('apiKey')
                        try: 
                            with open("/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/device.json", 'w') as f:
                                json.dump(device_config, f, indent=4)
                        except Exception as e:
                            print(f'Error seve file : {e}')        
                    else:
                        print(response.status_code)
                        return
                except Exception as e:
                    print(f'Error regsiter:{e}')
    while True:
        try:
            try:
                with open("/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/device.json", 'r') as f:
                    device_config = json.load(f)
            except Exception as e:
                print(f"Device config error: {e}")
                device_config = {}
            saved_api_key = device_config.get('apiKey')
            if not saved_api_key:
                await reg()
            else:
                try:
                    async with httpx.AsyncClient() as requests:
                        headers = {
                            "Content-Type": "application/json",
                            "x-api-key": saved_api_key
                        }
                        res = await requests.get(f"{nodeJS_URL}/api/device/check-api-key", headers=headers)
                        
                        if res.status_code == httpx.codes.OK:
                            API_KEY = saved_api_key
                            print("✅ Device already registered with valid API key.")
                        else:
                            print("⚠️ API key invalid (Server rejected), re-registering device.")
                            await reg()

                except Exception as e:
               
                    print(f"Cannot reach NodeJS Server to verify API Key: {e}")
                    print("Network is down, using cached API Key to start AI locally.")
    
                    API_KEY = saved_api_key 

         
            print(f'Current API_KEY is {API_KEY}')
            if API_KEY:
                print("Registration sequence passed. Starting system...")
                break 
            
    
            print("Retrying registration in 5 seconds...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"Fatal Error in register_device: {e}")
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)

def init_cctv_ai_frame(rtsp_urls, max_retries=5):
    attempt = 0
    while attempt < max_retries:
        try:
            initial_frames = {}
            for i, url in enumerate(rtsp_urls):
                cam_id = f"cam{i+1}"
                print(f"Capturing initial frame for {cam_id}...")
                
            
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    success = False
                   
                    for _ in range(50):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            initial_frames[cam_id] = frame
                            print(f"✅ Success: {cam_id} frame captured.")
                            success = True
                            break 
                            
                    if not success:
                        print(f"❌ Error: Cannot read frame from {cam_id} (Timeout waiting for Keyframe).")
                        initial_frames[cam_id] = None
                        
                    cap.release()
                else:
                    print(f"❌ Error: Cannot open stream for {cam_id}.")
                    initial_frames[cam_id] = None

            image_base64_frames = {}
            for cam_id, frame in initial_frames.items():
                if frame is not None:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    image_base64_frames[cam_id] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                else:
                    image_base64_frames[cam_id] = None
                    
            return image_base64_frames 

        except Exception as e:
            attempt += 1
            print(f"Error in init_cctv_ai_frame: {e}")
            print(f"Retrying... ({attempt}/{max_retries}) in 5 seconds")
            import time
            time.sleep(5) 
            
    print("FATAL: Failed to initialize CCTV frames after multiple attempts.")
    return None


async def run_cctv_ai():
    global LIFE, cctv_ai
    try:
        shared_data.set_active_device(0)
        global rtsp_urls
        
        from tritonclient.grpc import aio as grpc
        triton_client = grpc.InferenceServerClient("localhost:8001")
        
        cctv_ai = CCTV_AI(rtsp_urls, shared_data,API_KEY,nodeJS_URL)
        await cctv_ai.run(triton_client) 
    except Exception as e:
        shared_data.set_LIFE('inactive')
        print(f"CCTV_AI Error: {e}")
    finally:
        cctv_ai = None

def runserverTriton():
        
        return {"message": "AI is already running"}
# check divice are Exit
def check_device():
    try:
        with open("/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/device.json", 'r') as f:
                print("Device config Exit")
    except Exception as e:
        with open("/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/device.json", 'w') as f:
            json.dump({
                "ip": "",
                "mac": "",
                "apiKey": "",
            }, f)
            print("Device config created with default values.")



# async def zmq_listener():
#     global latest_frames
#     context = zmq.asyncio.Context()
#     receiver = context.socket(zmq.SUB)
#     # สมมติว่ารับจาก port 5555 และรวมทุก topic
#     receiver.connect("tcp://localhost:5555") 
#     receiver.setsockopt_string(zmq.SUBSCRIBE, "")

#     print("ZMQ Listener started...")
#     while True:
#         try:
#             # รับข้อมูลทั้งหมดมาเก็บเป็น List ก่อน (ไม่บังคับว่าต้องมีแค่ 2)
#             message_parts = await receiver.recv_multipart()
            
#             # สมมติว่า Topic คือส่วนแรกสุด (Index 0)
#             topic = message_parts[0]
#             topic_str = topic.decode('utf-8')

#             # สมมติว่า Frame ภาพ คือส่วนสุดท้ายเสมอ (Index -1)
#             frame_bytes = message_parts[-1]

#             if topic_str in latest_frames:
#                 # เก็บ bytes ของภาพลงในตัวแปร global
#                 latest_frames[topic_str] = frame_bytes
                
#         except Exception as e:
#             print(f"ZMQ Error: {e}")
#             await asyncio.sleep(1)



# from fastapi.responses import StreamingResponse

# @app.get("/video/{cam_id}")
# async def video_feed(cam_id: str):
#     """
#     Endpoint สำหรับทดสอบดูวิดีโอผ่าน HTTP
#     เช่น http://localhost:5000/video/cam1
#     """
#     if cam_id not in latest_frames:
#         return Response(content="Camera not found", status_code=404)

#     async def generate():
#         while True:
#             frame = latest_frames.get(cam_id)
#             if frame:
#                 # ส่งข้อมูลในรูปแบบ Multipart (MJPEG)
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
#             # ควบคุม FPS ไม่ให้ดึงภาพเร็วเกินไป (เช่น 30 FPS)
#             await asyncio.sleep(0.033)

#     return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
