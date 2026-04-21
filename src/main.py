#!/usr/bin/python3
import asyncio
import time
import zmq.asyncio
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
import base64



# -----------------------------Global variables----------------------------------------------
latest_frames = {"cam1": b"", "cam2": b"", "cam3": b"", "cam4": b""}
latest_payloads = {"cam1": [], "cam2": [], "cam3": [], "cam4": []}
miss_counter = {"cam1": 0, "cam2": 0, "cam3": 0, "cam4": 0}
alarm_cooldown = {"cam1": 0, "cam2": 0, "cam3": 0, "cam4": 0}
MISS_THRESHOLD = 5 
COOLDOWN_SEC = 5 
sio = socketio.AsyncClient()
API_KEY = None
shared_data = data.Data()
LIFE = None
nodeJS_URL = 'http://10.153.89.128:5000'
cctv_task = None
cctv_ai = None
class AIcontoll(BaseModel):
    acction: str
    roi: List[dict] = None
    confidence_threshold: float = None

#  -----------------------------FastAPI app and lifespan----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cctv_ai, cctv_task,shared_data
    task = asyncio.create_task(zmq_listener())
    task2 = asyncio.create_task(register_device())
    await task2  
    await sio.connect(nodeJS_URL)
    task3 = asyncio.create_task(send_status('5s'))
    # cctv_task = asyncio.create_task(run_cctv_ai())
    # asyncio.create_task(send_status())
    asyncio.create_task(send_status('now'))
    yield
    shared_data.set_active_device(0)
    task.cancel()
    task2.cancel()
    task3.cancel()
    if cctv_ai:
        cctv_ai = None
    if cctv_task:
        cctv_task.cancel()
    await sio.disconnect()
   


app = FastAPI(lifespan=lifespan)

# -------------------------------------Endpoints-----------------------------------------------

@app.get("/image/{cam_id}")
async def get_image(cam_id: str):
    if cam_id not in latest_frames:
        return {"error": "Camera not found"}
    
    frame_bytes = latest_frames.get(cam_id)
    if not frame_bytes:
        print(f"No frame available for {cam_id}")
        return Response(status_code=204,)
        
    return Response(content=frame_bytes, media_type="image/jpeg")

# Alam delete this endpoint
@app.get("/alam_detect/{camid}")
async def alam_detect(camid: str):
    print("Alam detect endpoint called")
    try:
        payload= await alarm_detect(camid,detaction_class="NO_CAT",message="Alarm triggered",confidence=0.8)
    except Exception as e:
        print(f"Error in alarm_detect: {e}")
    return payload



@app.post("/control")
async def control_ai(control: AIcontoll):
    global cctv_ai, cctv_task,latest_frames
    roi = control.roi
    confidence_threshold = control.confidence_threshold
    with open ('./config/camera_config.json','w') as f:
        json.dump(roi, f, indent=4)
    shared_data.set_config(roi)
    with open ('./config/confident.json','w') as f:
        json.dump({"confidence": confidence_threshold}, f, indent=4)
    shared_data.set_confidence_threshold(confidence_threshold)
    # print(f"Control AI: action={control.acction}, roi={roi}, confidence_threshold={confidence_threshold}")
    if control.acction == "start":
        if cctv_ai is not None:
            return {"message": "AI is already running"}
        else:
            cctv_task = asyncio.create_task(run_cctv_ai())
        return {"message": "AI started"}
    elif control.acction == "stop":
        # latest_frames = {"cam1": b"", "cam2": b"", "cam3": b"", "cam4": b""}
        if cctv_ai is not None:
            cctv_ai.cleanup()
            cctv_ai = None
        return {"message": "AI stopped"}
    send_status('now')
    # get 
# ----------------------------------------------socketio events-----------------------------------------------------------
@sio.event
async def connect():
    print("SocketIO connected")
    await sio.emit('register-device-status', {'apiKey': API_KEY})
@sio.event
async def disconnect():
    print("SocketIO disconnected")
    
# ----------------------------------------------function-----------------------------------------------------------

# send status to nodeJS via SocketIO
async def send_status(time='5s'):
    global API_KEY ,LIFE
    if time == '5s':
        while True:
            LIFE = shared_data.get_LIFE()
            if API_KEY:
                try:
                    await sio.emit('heartbeat', get_jetson_stats(LIFE))
                except Exception as e:
                    print(f"Failed to send status via SocketIO: {e}")
            LIFE = 'inactive'
            await asyncio.sleep(5)
    else:
        LIFE = shared_data.get_LIFE()
        if API_KEY:
            try:
                await sio.emit('heartbeat', get_jetson_stats(LIFE))
            except Exception as e:
                print(f"Failed to send status via SocketIO: {e}")

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
    global API_KEY
    async def reg():
            global API_KEY
            print("Device config missing mcId or apiKey")
            URL = f'{nodeJS_URL}/api/device/register'
            data ={
                'mac':device_config.get('mac'),
                'ip':device_config.get('ip')
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
    try:
        with open("/home/orin_nano/@OtherProject/EE_CCTV_AI-V2/EE_CCTV_AI/config/device.json", 'r') as f:
            device_config = json.load(f)
    except Exception as e:
        print(f"Device config error: {e}")


    # if not device_config.get('mcId') or not device_config.get('apiKey'):
    if not device_config.get('apiKey'):
       await reg()
    else:
        try:
            async with httpx.AsyncClient() as requests:
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": device_config.get('apiKey')
                }
                res = await requests.get(f"{nodeJS_URL}/api/device/check-api-key", headers=headers)
                if res.status_code == httpx.codes.OK:
                    API_KEY = device_config.get('apiKey')
                    print("Device already registered with valid API key.")
                else:
                    print("API key invalid, re-registering device.")
                    await reg()
        except Exception as e:
            print(f"Error occurred while verifying device info: {e}")
    print(f'API_KEY is {API_KEY}')
    return 



async def zmq_listener():
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://127.0.0.1:5555")
    sock.subscribe("") 
    while True:
        try:
            topic_bytes, payload_bytes, frame_bytes = await sock.recv_multipart()
            cam_id = topic_bytes.decode('utf-8')
            latest_frames[cam_id] = frame_bytes
            latest_payloads[cam_id] = json.loads(payload_bytes.decode('utf-8'))

            if not latest_payloads[cam_id]:
                continue
            if shared_data.get_active_device() != 1:
                continue

            cam_data = latest_payloads[cam_id]

            if len(cam_data["detection"]) < cam_data["machineCount"]:
                miss_counter[cam_id] += 1

                if miss_counter[cam_id] >= MISS_THRESHOLD:
                    # เช็ค cooldown ก่อน alarm
                    now = time.time()
                    if now - alarm_cooldown[cam_id] >= COOLDOWN_SEC:
                        detection_class = 'machine missing'
                        confidence = 0.0
                        message = 'can not detect machine'
                        print(f"Detection on {cam_id}: Class={detection_class}, miss={miss_counter[cam_id]} frames")
                        alarm_cooldown[cam_id] = now
                        await alarm_detect(cam_id, detection_class, message, confidence)
            else:
                miss_counter[cam_id] = 0  

                class1_det = next(
                    (det for det in cam_data["detection"] if det["class"] == 1), None
                )
                if class1_det:
                    now = time.time()
                    if now - alarm_cooldown[cam_id] >= COOLDOWN_SEC:
                        detection_class = 'alarm open'
                        confidence = class1_det["conf"]
                        message = 'open detected'
                        print(f"Detection on {cam_id}: Class={detection_class}, Confidence={confidence * 100:.1f}%")
                        alarm_cooldown[cam_id] = now
                        await alarm_detect(cam_id, detection_class, message, confidence)

        except Exception as e:
            print(f"Error in zmq_listener: {e}")




async def run_cctv_ai():
    global LIFE ,cctv_ai
    try:
        shared_data.set_LIFE('active')
        rtsp_urls = [
            "images/video_test/D01_20260326154700.mp4",
            "images/video_test/D02_20260326154700.mp4",
            "images/video_test/D03_20260326154702.mp4",
            "images/video_test/D04_20260326154710.mp4"
        ]
        cctv_ai = CCTV_AI(rtsp_urls, shared_data)
        await cctv_ai.run()
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

async def alarm_detect(camid, detaction_class,message,confidence):
    header = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    image = latest_frames.get(camid)
    image_base64 = base64.b64encode(image) if image else None
    payload = {
        "alarmType": detaction_class,
        "message": message,
        "cameraId": f"{camid[:3].upper()}-0{camid[3:]}",
        "confidence": confidence * 100,
        "imageBase64": f"data:image/jpeg;base64,{image_base64.decode('utf-8')}"
    }
    async with httpx.AsyncClient() as requests:
        try:
            res = await requests.post(f'{nodeJS_URL}/api/device/alarm', headers=header, json=payload, timeout=5.0)
            if res.status_code == 200:
                # print("Alarm data sent successfully")
                pass
            else:
                print(f"Failed to send alarm data: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Error occurred while sending alarm data: {e}")

    return payload




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
