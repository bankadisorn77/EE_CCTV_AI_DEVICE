"""HTTP routes for pipeline control, status, and device maintenance."""
import asyncio
import json
import os
import signal
from pathlib import Path
from time import time
from typing import Optional

from fastapi import APIRouter, Response, Query

from src.ee_cctv_ai.schemas import AIControlRequest
from src.ee_cctv_ai.config_store import load_config, save_config


def create_router(
    project_adapter,
    shared_data,
    config_path,
    device_config_path,
    rtsp_urls,
    initial_frame_capture,
    send_status,
):
    router = APIRouter()

    @router.get("/test")
    async def test_endpoint():
        return {"message": "API is working!"}

    @router.get("/fps")
    async def get_fps(cam: Optional[str] = Query(None, description="Camera key เช่น cam1, cam2")):
        """Endpoint สำหรับดึงค่า FPS สรุปของระบบ หรือระบุเฉพาะกล้อง"""
        pipeline = project_adapter.pipeline
        if pipeline is None or not project_adapter.running:
            return {
                "running": False,
                "message": "Pipeline is not running",
                "stats": {}
            }

        fps_data = pipeline.get_camera_fps(cam)
        return {
            "running": True,
            "target": cam if cam else "all",
            "stats": fps_data
        }

    @router.get("/status")
    async def get_pipeline_status():
        running = pipeline_manager.running
        cameras = []
        alert_status = None
        pipeline = pipeline_manager.pipeline
        
        # ดึง FPS ทั้งหมดเตรียมไว้
        all_fps_stats = pipeline.get_camera_fps() if (pipeline and running) else {}

        if pipeline is not None:
            for camera in pipeline.cameras:
                last_frame = camera.last_frame_at
                cam_fps = all_fps_stats.get(camera.name, {})

                cameras.append({
                    "id": camera.name,
                    "connected": camera.connected,
                    "frameId": camera.frame_id,
                    "frameAgeSeconds": round(max(0.0, time() - last_frame), 2) if last_frame else None,
                    # พ่วงสถิติ FPS แสดงผลใน status ทันที
                    "captureFps": round(cam_fps.get("capture", 0.0), 1),
                    "inferenceFps": round(cam_fps.get("process", 0.0), 1),
                })
            alert_status = pipeline.alert_service.status()
            
        return {
            "running": running,
            "cameraCount": len(cameras),
            "cameras": cameras,
            "alerts": alert_status,
        }

    @router.post("/control")
    async def control_ai(control: AIControlRequest):
        config_data = load_config(config_path)
        if control.roi is not None:
            config_data["cameras"] = control.roi
            shared_data.set_config(control.roi)
        if control.confidence_threshold is not None:
            config_data["confidence_threshold"] = control.confidence_threshold
            shared_data.set_confidence_threshold(control.confidence_threshold)
        save_config(config_data, config_path)

        if control.action == "start":
            started = await project_adapter.start()
            if not started:
                return {"message": "AI is already running"}
            await send_status()
            return {"message": "AI started"}
        if control.action == "stop":
            await project_adapter.stop()
            await send_status()
            return {"message": "AI stopped"}
        return {"message": "Unsupported action", "action": control.action}

    @router.get("/initial-frames")
    async def get_initial_frames():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, initial_frame_capture, rtsp_urls)

    @router.post("/reset")
    async def reset():
        try:
            default_config = {
                "cameras": [
                    {
                        "name": f"cam{i + 1}",
                        "machineCount": 0,
                        "roi": {"row1": None, "col1": None, "row2": None, "col2": None},
                    }
                    for i in range(len(rtsp_urls))
                ],
                "confidence_threshold": 0.5,
            }
            save_config(default_config, config_path)

            try:
                with Path(device_config_path).open("r", encoding="utf-8") as handle:
                    device_config = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                device_config = {}

            device_config["ip"] = ""
            device_config["mac"] = ""
            device_config["apiKey"] = ""
            with Path(device_config_path).open("w", encoding="utf-8") as handle:
                json.dump(device_config, handle, indent=4)
        except Exception as exc:
            print(f"Error during reset file operations: {exc}")
            return Response(status_code=500, content=f"Error during reset: {exc}")

        async def shutdown():
            await asyncio.sleep(1)
            print("Shutting down application...")
            os.kill(os.getpid(), signal.SIGINT)

        asyncio.create_task(shutdown())
        return {"message": "Device reset successfully. Shutting down."}

    @router.post("/restart")
    async def restart():
        return {"success": True}

    return router