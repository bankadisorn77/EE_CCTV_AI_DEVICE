#!/usr/bin/env python3
"""Standalone runner to benchmark CCTV_AI_Ultralytics worker loops directly."""

import asyncio
import os
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# บังคับเปิด TCP สำหรับ RTSP และเปิดตัววัด Profile
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["EE_CCTV_PROFILE"] = "1"

from src.camera_manager import CCTV_AI_Ultralytics
from src.main import load_device_config
from src.shere_data import data


async def main():
    print("=== Starting Standalone CCTV_AI Worker Test ===")

    # 1. โหลดการตั้งค่าอุปกรณ์ (URL กล้อง)
    device_config = load_device_config()
    rtsp_urls = device_config.get("rstp_url", ["", "", "", ""])
    api_key = device_config.get("apiKey", "")
    webserver_host = device_config.get("webserver_url", "localhost")
    webserver_port = int(device_config.get("webserver_port", 8000))
    nodejs_url = os.getenv("NODEJS_URL", f"http://{webserver_host}:{webserver_port}")

    print(f"RTSP URLs: {rtsp_urls}")

    # 2. จำลอง Shared Data Store
    shared_data = data.Data()

    # 3. สร้าง Instance ของ CCTV_AI_Ultralytics
    cctv_app = CCTV_AI_Ultralytics(
        rtsp_urls=rtsp_urls,
        shared_data=shared_data,
        API_KEY=api_key,
        nodeJS_URL=nodejs_url,
    )

    # 4. จัดการสัญญาณยกเลิก (CTRL+C)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_signal():
        print("\nTermination signal received. Stopping worker...")
        stop_event.set()
        cctv_app.stop_process = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    # 5. รันคอร์หลักของระบบกล้องและการ Inference
    runner_task = asyncio.create_task(cctv_app.run())
    stop_waiter = asyncio.create_task(stop_event.wait())

    done, pending = await asyncio.wait(
        [runner_task, stop_waiter],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    # 6. คืนทรัพยากรกล้องและหน่วยความจำ
    cctv_app.cleanup()
    print("Standalone Worker Test Finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass