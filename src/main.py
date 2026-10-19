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

DEFAULT_CAMERA_COUNT = int(os.getenv("EE_CCTV_NUM_CAMERAS", "8"))


def load_device_config():
    defaults = {
        "rstp_url": ["" for _ in range(DEFAULT_CAMERA_COUNT)],
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

from src.camera_manager import CCTV_AI
from src.shere_data import data
from src.ee_cctv_ai.frame_capture import init_cctv_ai_frame
from src.ee_cctv_ai.pipeline_manager import PipelineManager
from src.ee_cctv_ai.device_service import register_device
from src.ee_cctv_ai.getip import get_lan_network_info
from src.ee_cctv_ai.socket_service import SocketStatusService

# set rtsp
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# -----------------------------Global variables----------------------------------------------
sio = socketio.AsyncClient()

API_KEY = device_config.get("apiKey")
rtsp_urls = device_config.get("rstp_url", ["" for _ in range(DEFAULT_CAMERA_COUNT)])
webserver_host = device_config.get("webserver_url", "localhost")
webserver_port = int(device_config.get("webserver_port", 8000))

shared_data = data.Data()
LIFE = None
nodeJS_URL = os.getenv("NODEJS_URL", f"http://{webserver_host}:{webserver_port}")
register_task = None
pipeline_manager = PipelineManager(CCTV_AI, rtsp_urls, shared_data, lambda: API_KEY, nodeJS_URL)
socket_service = SocketStatusService(sio, nodeJS_URL, shared_data, lambda: API_KEY)


def set_api_key(value):
    global API_KEY
    API_KEY = value


# -----------------------------FastAPI app and lifespan----------------------------------------------
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
    background_tasks.append(asyncio.create_task(socket_service.connect_forever(), name="socketio-connection"))
    background_tasks.append(asyncio.create_task(socket_service.heartbeat_loop(), name="status-heartbeat"))
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
        send_status=socket_service.send_status_now,
    )
)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)