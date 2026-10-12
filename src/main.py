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
import signal
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

from src.camera_manager import CCTV_AI_Ultralytics as CCTV_AI 
from src.shere_data import data
from src.ee_cctv_ai.getip import get_lan_network_info

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
register_task = None
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
    global cctv_ai, cctv_task, register_task
    background_tasks = []
    register_task = asyncio.create_task(register_device(), name="device-registration")
    background_tasks.append(register_task)
    background_tasks.append(asyncio.create_task(connect_socketio(), name="socketio-connection"))
    background_tasks.append(asyncio.create_task(send_status("stanby"), name="status-standby"))
    background_tasks.append(asyncio.create_task(send_status("now"), name="status-now"))
    try:
        yield
    finally:
        if cctv_ai is not None:
            cctv_ai.cleanup()
        if cctv_task is not None and not cctv_task.done():
            cctv_task.cancel()
            try:
                await cctv_task
            except asyncio.CancelledError:
                pass
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await sio.disconnect()
   


app = FastAPI(lifespan=lifespan)

# -------------------------------------Endpoints-----------------------------------------------
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!"}


@app.get("/status")
async def get_pipeline_status():
    running = cctv_task is not None and not cctv_task.done()
    cameras = []
    alert_status = None
    if cctv_ai is not None:
        for camera in cctv_ai.cameras:
            last_frame = camera.last_frame_at
            cameras.append({
                "id": camera.name,
                "connected": camera.connected,
                "frameId": camera.frame_id,
                "frameAgeSeconds": round(max(0.0, time() - last_frame), 2) if last_frame else None,
            })
        alert_status = cctv_ai.alert_service.status()
    return {
        "running": running,
        "cameraCount": len(cameras),
        "cameras": cameras,
        "alerts": alert_status,
    }
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

    # print(f"Control AI: action={control.action}, roi={roi}, confidence_threshold={confidence_threshold}")
    if control.action == "start":
        if cctv_task is not None and not cctv_task.done():
            return {"message": "AI is already running"}
        cctv_task = asyncio.create_task(run_cctv_ai(), name="cctv-pipeline")
        shared_data.set_active_device(1)
        await send_status("now")
        return {"message": "AI started"}
    if control.action == "stop":
        if cctv_ai is not None:
            cctv_ai.cleanup()
        if cctv_task is not None and not cctv_task.done():
            cctv_task.cancel()
            try:
                await cctv_task
            except asyncio.CancelledError:
                pass
        cctv_task = None
        cctv_ai = None
        shared_data.set_active_device(0)
        await send_status("now")
        return {"message": "AI stopped"}
    return {"message": "Unsupported action", "action": control.action}
    
@app.get("/initial-frames")
async def get_initial_frames():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, init_cctv_ai_frame, rtsp_urls)
@app.post('/reset')
async def reset():
    try:
        # 1. Reset config.json to its default state
        default_config = {
            "cameras": [
                {
                    "name": f"cam{i+1}",
                    "machineCount": 0,
                    "roi": {"row1": None, "col1": None, "row2": None, "col2": None}
                }
                for i in range(len(rtsp_urls))
            ],
            "confidence_threshold": 0.5
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
        print("config.json has been reset to default values.")

        # 2. Clear sensitive/unique fields in device.json
        try:
            with DEVICE_CONFIG_PATH.open('r') as f:
                device_conf = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            device_conf = {}

        device_conf['ip'] = ""
        device_conf['mac'] = ""
        device_conf['apiKey'] = ""
        
        with DEVICE_CONFIG_PATH.open('w') as f:
            json.dump(device_conf, f, indent=4)
        print("ip, mac, and apiKey in device.json have been cleared.")

    except Exception as e:
        print(f"Error during reset file operations: {e}")
        return Response(status_code=500, content=f"Error during reset: {e}")

    # 3. Schedule a graceful shutdown
    async def shutdown():
        await asyncio.sleep(1)  # Allow time for the HTTP response to be sent
        print("Shutting down application...")
        os.kill(os.getpid(), signal.SIGINT)

    asyncio.create_task(shutdown())
    
    return {"message": "Device reset successfully. Shutting down."}

@app.post('/restart')
async def restart():
    return {"success": True}
    
# ----------------------------------------------socketio events-----------------------------------------------------------
@sio.event
async def connect():
    print("SocketIO connected")
    await sio.emit('register-device-status', {'apiKey': API_KEY})
@sio.event
async def disconnect():
    print("SocketIO disconnected")
@sio.event
async def connect_error(data):
    print("SocketIO connect_error:", data)
    
# ----------------------------------------------function-----------------------------------------------------------

async def connect_socketio():
    while True:
        if sio.connected:
            break
        try:
            print(f" Attempting to connect SocketIO to {nodeJS_URL}...")
            await sio.connect(nodeJS_URL, wait_timeout=10)
            print("SocketIO connected successfully!")
            break
        except Exception as e:
            print(f"SocketIO connection failed: {e!r}")
            try:
                await sio.disconnect()   
            except Exception:
                pass
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
    loop = asyncio.get_running_loop()
    image_base64_frames = await loop.run_in_executor(None, init_cctv_ai_frame, rtsp_urls)
    async def reg():
        print("Device config missing mcId or apiKey")
        URL = f'{nodeJS_URL}/api/device/register'

        lan_ip, lan_mac = get_lan_network_info()

        device_ip = lan_ip if not device_config.get('ip') else device_config.get('ip')
        device_mac = lan_mac if not device_config.get('mac') else device_config.get('mac')

        device_config['ip'] = device_ip
        device_config['mac'] = device_mac

        data = {
            'mac': device_mac,
            'ip': device_ip,
            'frame': image_base64_frames,
        }
        
        async with httpx.AsyncClient() as requests:
            try:
                response = await requests.post(url=URL, json=data)
                if response.status_code == 200:
                    res_data = response.json()
                    device_config['apiKey'] = res_data.get('apiKey')
                    API_KEY = res_data.get('apiKey')
                    try: 
                        # เซฟ config พร้อม IP และ MAC ลง device.json
                        with DEVICE_CONFIG_PATH.open('w') as f:
                            json.dump(device_config, f, indent=4)
                        print(f"✅ Device registered: IP={device_ip}, MAC={device_mac}")
                    except Exception as e:
                        print(f'Error save file: {e}')        
                else:
                    print(f"Register response status: {response.status_code}")
                    return
            except Exception as e:
                print(f'Error register: {e}')
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
                            try:
                                async with httpx.AsyncClient() as requests:
                                    headers = {
                                        "Content-Type": "application/json",
                                        "x-api-key": API_KEY
                                    }
                                    payload = {'frame':image_base64_frames}
                                    res = await requests.post(f"{nodeJS_URL}/api/device/re-captureImage", headers=headers,json=payload)
                                    if res.status_code == httpx.codes.OK:
                                        print('✅ Re-capture Image success')
                                    else:
                                        print('Re-capture Image failed')
                            except Exception as e :
                                print(f'Re-capture Image failed:{e}')
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
            await asyncio.sleep(5) 
            
    print("FATAL: Failed to initialize CCTV frames after multiple attempts.")
    return None


async def run_cctv_ai():
    global LIFE, cctv_ai
    pipeline = None
    try:
        shared_data.set_active_device(1)
        pipeline = CCTV_AI(rtsp_urls, shared_data, API_KEY, nodeJS_URL)
        cctv_ai = pipeline
        await pipeline.run()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        shared_data.set_LIFE("inactive")
        print(f"CCTV_AI Error: {e}")
    finally:
        if pipeline is not None:
            pipeline.cleanup()
        cctv_ai = None
        shared_data.set_active_device(0)

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
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
