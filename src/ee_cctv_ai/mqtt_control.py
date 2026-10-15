"""MQTT command/status bridge for the existing CCTV PipelineManager."""
import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict

logger = logging.getLogger("CCTV.MQTT")


class MqttControlService:
    """Translate MQTT commands into async PipelineManager calls.

    Commands: start, stop, get_status.
    Topics default to <prefix>/command, <prefix>/status, and <prefix>/ack.
    MQTT is optional at runtime: without EE_CCTV_MQTT_BROKER the service stays
    disabled and the existing FastAPI controls continue to work.
    """

    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager
        self.broker = os.getenv("EE_CCTV_MQTT_BROKER", "").strip()
        self.port = int(os.getenv("EE_CCTV_MQTT_PORT", "1883"))
        self.username = os.getenv("EE_CCTV_MQTT_USERNAME", "")
        self.password = os.getenv("EE_CCTV_MQTT_PASSWORD", "")
        self.client_id = os.getenv("EE_CCTV_MQTT_CLIENT_ID", "ee-cctv-ai-device")
        self.prefix = os.getenv("EE_CCTV_MQTT_TOPIC_PREFIX", "devices/cctv").strip("/")
        self.command_topic = f"{self.prefix}/command"
        self.status_topic = f"{self.prefix}/status"
        self.ack_topic = f"{self.prefix}/ack"
        self.client = None
        self.loop = None
        self._dispatch_lock = None
        self._recent = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(self.broker)

    def start(self, loop):
        self.loop = loop
        self._dispatch_lock = asyncio.Lock()
        if not self.enabled:
            logger.info("MQTT disabled; EE_CCTV_MQTT_BROKER is not configured.")
            return False
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.exception("MQTT broker configured but paho-mqtt is not installed.")
            return False

        client = mqtt.Client(client_id=self.client_id)
        if self.username:
            client.username_pw_set(self.username, self.password or None)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        self.client = client
        client.connect_async(self.broker, self.port, keepalive=30)
        client.loop_start()
        logger.info("MQTT client starting for broker %s:%s", self.broker, self.port)
        return True

    def stop(self):
        client, self.client = self.client, None
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                logger.exception("MQTT disconnect failed")
            try:
                client.loop_stop()
            except Exception:
                logger.exception("MQTT network loop stop failed")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        code = getattr(reason_code, "value", reason_code)
        try:
            success = int(code) == 0
        except (TypeError, ValueError):
            success = str(code).lower() in ("success", "0")
        if success:
            client.subscribe(self.command_topic, qos=1)
            logger.info("MQTT subscribed to %s", self.command_topic)
            self.publish_status()
        else:
            logger.error("MQTT connect failed: %s", reason_code)

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        logger.warning("MQTT disconnected: %s", reason_code)

    def _on_message(self, client, userdata, message):
        if self.loop is None or self.loop.is_closed():
            return
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._publish(self.ack_topic, {
                "ok": False, "error": "invalid_json", "timestamp": time.time()
            })
            return
        if not isinstance(payload, dict):
            self._publish(self.ack_topic, {
                "ok": False, "error": "command_must_be_object", "timestamp": time.time()
            })
            return
        asyncio.run_coroutine_threadsafe(self._dispatch(payload), self.loop)

    async def _dispatch(self, payload):
        if self._dispatch_lock is None:
            return
        async with self._dispatch_lock:
            command_id = str(payload.get("command_id", "")).strip()
            if command_id and command_id in self._recent:
                self._publish(self.ack_topic, self._recent[command_id])
                return

            command = str(payload.get("command", "")).strip().lower()
            if command == "start":
                if self.pipeline_manager.running:
                    ack = {"ok": True, "command": command, "message": "already_running"}
                else:
                    started = await self.pipeline_manager.start()
                    ack = {"ok": bool(started or self.pipeline_manager.running),
                           "command": command,
                           "message": "started" if started else "already_running"}
            elif command == "stop":
                was_running = self.pipeline_manager.running
                if was_running:
                    await self.pipeline_manager.stop()
                ack = {"ok": True, "command": command,
                       "message": "stopped" if was_running else "already_stopped"}
            elif command in ("status", "get_status"):
                ack = {"ok": True, "command": "get_status", "status": self.status_snapshot()}
            else:
                ack = {"ok": False, "command": command, "error": "unsupported_command"}

            ack["command_id"] = command_id or None
            ack["timestamp"] = time.time()
            if command_id:
                self._recent[command_id] = ack
                self._recent.move_to_end(command_id)
                while len(self._recent) > 256:
                    self._recent.popitem(last=False)
            self._publish(self.ack_topic, ack)
            self.publish_status()

    def status_snapshot(self):
        pipeline = self.pipeline_manager.pipeline
        cameras = []
        fps = pipeline.get_camera_fps() if pipeline is not None else {}
        if pipeline is not None:
            now = time.time()
            for camera in pipeline.cameras:
                last_frame = getattr(camera, "last_frame_at", None)
                cameras.append({
                    "id": camera.name,
                    "connected": bool(getattr(camera, "connected", False)),
                    "frame_id": getattr(camera, "frame_id", None),
                    "frame_age_seconds": round(max(0.0, now - last_frame), 2) if last_frame else None,
                    "capture_fps": fps.get(camera.name, {}).get("capture", 0.0),
                    "inference_fps": fps.get(camera.name, {}).get("process", 0.0),
                })
        return {
            "service": "EE_CCTV_AI_DEVICE",
            "pipeline_running": bool(self.pipeline_manager.running),
            "cameras": cameras,
            "timestamp": time.time(),
        }

    def publish_status(self):
        self._publish(self.status_topic, self.status_snapshot(), retain=True)

    async def status_loop(self, interval=5.0):
        while True:
            self.publish_status()
            await asyncio.sleep(interval)

    def _publish(self, topic, payload, retain=False):
        client = self.client
        if client is None:
            return
        try:
            client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1, retain=retain)
        except Exception:
            logger.exception("MQTT publish failed for topic %s", topic)
