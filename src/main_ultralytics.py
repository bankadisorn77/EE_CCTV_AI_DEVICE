#!/usr/bin/python3
import asyncio
import base64
import os
import sys
from pathlib import Path
from time import time
from fastapi import FastAPI, Response
import socketio
import uvicorn
from contextlib import asynccontextmanager
import httpx
import json
import psutil
from pydantic import BaseModel
from typing import List
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE_CONFIG_PATH = PROJECT_ROOT / "config" / "device.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def load_device_config():
    defaults = {
        "rstp_url": ["", "", "", ""],
        "webserver_url": "localhost",
        "webserver_port": 8000,
        "ip": "127.0.0.1",
        "mac": "00:00:00:00:00:00",
        "apiKey": "your_api_key_here",
    }
    try:
        with DEVICE_CONFIG_PATH.open("r") as config_file:
            config = json.load(config_file)
        defaults.update(config)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Device config error: {error}. Using defaults.")
    return defaults


device_config = load_device_config()

from src.camera_manager_ultralytics import CCTV_AI_Ultralytics as CCTV_AI # USE ULTRALYTICS
from src.shere_data import data

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



# -----------------------------Global variables----------------------------------------------
sio = socketio.AsyncClient()

API_KEY = device_config.get("apiKey")
rtsp_urls = device_config.get("rstp_url", ["", "", "", ""])
webserver_host = device_config.get("webserver_url", "localhost")
webserver_port = int(device_config.get("webserver_port", 8000))

shared_data = data.Data()
LIFE = None
nodeJS_URL = os.getenv("NODEJS_URL", f"http://{webserver_host}:{webserver_port}")
cctv_task = None
cctv_ai = None
class AIcontoll(BaseModel):
    action: str
    roi: List[dict] = None
    confidence_threshold: float = None

# rtsp_urls = [
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101"
# ]

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
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!"}
@app.post("/control")
async def control_ai(control: AIcontoll):
    global cctv_ai, cctv_task
    roi = control.roi
    confidence_threshold = control.confidence_threshold

    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config_data = {}

    if roi is not None:
        config_data["cameras"] = roi
        shared_data.set_config(roi)

    if confidence_threshold is not None:
        config_data["confidence_threshold"] = confidence_threshold
        shared_data.set_confidence_threshold(confidence_threshold)

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"Control AI: action={control.action}, roi={roi}, confidence_threshold={confidence_threshold}")
    if control.action == "start":
        if cctv_ai is not None:
            return {"message": "AI is already running"}
        else:
            cctv_task = asyncio.create_task(run_cctv_ai())
        shared_data.set_active_device(1)
        return {"message": "AI started"}
    elif control.action == "stop":
        if cctv_ai is not None:
            cctv_ai.cleanup()
            cctv_ai = None
        shared_data.set_active_device(0)
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
        if API_KEY and sio.connected:
            try:
                await sio.emit('heartbeat', get_jetson_stats(LIFE))
            except Exception as e:
                print(f"Failed to send status via SocketIO: {e}")
        elif API_KEY:
            print("SocketIO is not connected; skipping heartbeat.")
    else:
        while True:
            LIFE = shared_data.get_LIFE()
            if API_KEY and sio.connected:
                try:
                    await sio.emit('heartbeat', get_jetson_stats(LIFE))
                except Exception as e:
                    print(f"Failed to send status via SocketIO: {e}")
            elif API_KEY:
                print("SocketIO is not connected; skipping heartbeat.")
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
                            with DEVICE_CONFIG_PATH.open('w') as f:
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
                with DEVICE_CONFIG_PATH.open('r') as f:
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
        shared_data.set_active_device(1)
        global rtsp_urls
        
        # REMOVED Triton client setup
        # from tritonclient.grpc import aio as grpc
        # triton_client = grpc.InferenceServerClient("localhost:8001")
        
        cctv_ai = CCTV_AI(rtsp_urls, shared_data,API_KEY,nodeJS_URL)
        await cctv_ai.run() # MODIFIED: run without client
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
        with DEVICE_CONFIG_PATH.open('r') as f:
                print("Device config Exit")
    except Exception as e:
        with DEVICE_CONFIG_PATH.open('w') as f:
            json.dump(load_device_config(), f, indent=4)
            print("Device config created with default values.")


if __name__ == "__main__":
    uvicorn.run(app, host=webserver_host, port=5000)
