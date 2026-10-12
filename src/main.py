#!/usr/bin/python3
import asyncio
import os
import sys
from pathlib import Path
from time import time
from fastapi import FastAPI, Response
import socketio
import uvicorn
import signal
from contextlib import asynccontextmanager
import json

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
        "ip": "",
        "mac": "",
        "apiKey": "",
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
from src.ee_cctv_ai.frame_capture import init_cctv_ai_frame
from src.ee_cctv_ai.schemas import AIControlRequest
from src.ee_cctv_ai.config_store import load_config, save_config
from src.ee_cctv_ai.pipeline_manager import PipelineManager
from src.ee_cctv_ai.device_service import register_device
from src.ee_cctv_ai.status_service import get_jetson_stats
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
register_task = None
pipeline_manager = PipelineManager(CCTV_AI, rtsp_urls, shared_data, lambda: API_KEY, nodeJS_URL)
# rtsp_urls = [
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101",
#     "rtsp://admin:Utac12345@192.168.1.64/Streaming/Channels/101"
# ]

#  -----------------------------FastAPI app and lifespan----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global register_task
    background_tasks = []
    register_task = asyncio.create_task(
        register_device(
            DEVICE_CONFIG_PATH,
            init_cctv_ai_frame,
            rtsp_urls,
            nodeJS_URL,
            get_network_info=get_lan_network_info,
            get_api_key=lambda: API_KEY,
            set_api_key=set_api_key,
            socket_client=sio,
        ),
        name="device-registration",
    )
    background_tasks.append(register_task)
    background_tasks.append(asyncio.create_task(connect_socketio(), name="socketio-connection"))
    background_tasks.append(asyncio.create_task(send_status("stanby"), name="status-standby"))
    background_tasks.append(asyncio.create_task(send_status("now"), name="status-now"))
    try:
        yield
    finally:
        await pipeline_manager.stop()
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
    running = pipeline_manager.running
    cameras = []
    alert_status = None
    pipeline = pipeline_manager.pipeline
    if pipeline is not None:
        for camera in pipeline.cameras:
            last_frame = camera.last_frame_at
            cameras.append({
                "id": camera.name,
                "connected": camera.connected,
                "frameId": camera.frame_id,
                "frameAgeSeconds": round(max(0.0, time() - last_frame), 2) if last_frame else None,
            })
        alert_status = pipeline.alert_service.status()
    return {
        "running": running,
        "cameraCount": len(cameras),
        "cameras": cameras,
        "alerts": alert_status,
    }
@app.post("/control")
async def control_ai(control: AIControlRequest):
    roi = control.roi
    confidence_threshold = control.confidence_threshold

    config_data = load_config(CONFIG_PATH)

    if roi is not None:
        config_data["cameras"] = roi
        shared_data.set_config(roi)

    if confidence_threshold is not None:
        config_data["confidence_threshold"] = confidence_threshold
        shared_data.set_confidence_threshold(confidence_threshold)

    save_config(config_data, CONFIG_PATH)

    # print(f"Control AI: action={control.action}, roi={roi}, confidence_threshold={confidence_threshold}")
    if control.action == "start":
        started = await pipeline_manager.start()
        if not started:
            return {"message": "AI is already running"}
        await send_status("now")
        return {"message": "AI started"}
    if control.action == "stop":
        await pipeline_manager.stop()
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
        save_config(default_config, CONFIG_PATH)
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
    if API_KEY:
        await sio.emit("register-device-status", {"apiKey": API_KEY})
    else:
        print("SocketIO connected before device registration; registration will be emitted when the API key is ready.")
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
def set_api_key(value):
    global API_KEY
    API_KEY = value


async def send_status(time='stanby'):
    global API_KEY ,LIFE
    if time == 'now':
        LIFE = shared_data.get_LIFE()
        if API_KEY and sio.connected:
            try:
                await sio.emit('heartbeat', get_jetson_stats(LIFE, shared_data))
            except Exception as e:
                print(f"Failed to send status via SocketIO: {e}")
        elif API_KEY:
            print("SocketIO is not connected; skipping heartbeat.")
    else:
        while True:
            LIFE = shared_data.get_LIFE()
            if API_KEY and sio.connected:
                try:
                    await sio.emit('heartbeat', get_jetson_stats(LIFE, shared_data))
                except Exception as e:
                    print(f"Failed to send status via SocketIO: {e}")
            elif API_KEY:
                print("SocketIO is not connected; skipping heartbeat.")
            LIFE = 'inactive'
            await asyncio.sleep(5)
       

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
