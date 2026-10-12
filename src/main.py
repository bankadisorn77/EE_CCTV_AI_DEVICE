#!/usr/bin/python3
import asyncio
import os
import sys
from pathlib import Path
from fastapi import FastAPI
import socketio
import uvicorn
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

from src.api.routes import create_router

app = FastAPI(lifespan=lifespan)
app.include_router(
    create_router(
        pipeline_manager=pipeline_manager,
        shared_data=shared_data,
        config_path=CONFIG_PATH,
        device_config_path=DEVICE_CONFIG_PATH,
        rtsp_urls=rtsp_urls,
        initial_frame_capture=init_cctv_ai_frame,
        send_status=send_status,
    )
)

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
