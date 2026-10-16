import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("CameraManager")


class CameraManager:
    """Lifecycle and frame access for camera adapters.

    The manager intentionally knows only the adapter contract; vendor SDKs and
    transport-specific camera implementations belong to a project or driver
    package. A camera adapter should expose connection(), disconnect(),
    is_connected(), and get_latest_frame().
    """

    def __init__(self, cameras: Dict[str, Any]):
        self.cameras = dict(cameras)
        self._active_alignment_cam: Optional[str] = None
        self.active = False

    def get(self, camera_id: str) -> Optional[Any]:
        return self.cameras.get(camera_id)

    def ids(self) -> List[str]:
        return list(self.cameras.keys())

    @classmethod
    def from_definitions(
        cls,
        camera_definitions: list,
        factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> "CameraManager":
        cameras = {}
        for definition in camera_definitions:
            if not isinstance(definition, dict) or not definition.get("enabled", True):
                continue
            if factory is None:
                raise ValueError(
                    "A camera factory is required; provide project.create_camera(definition)."
                )
            camera_id = str(definition.get("id", "cam_1"))
            cameras[camera_id] = factory(definition)
        return cls(cameras)

    def set_active(self, active: bool):
        """Mark whether the owning project currently expects live camera streams."""
        self.active = bool(active)

    def connect_all(self, width: int, height: int, fps: int = 15, activate: bool = True) -> bool:
        self.active = False
        ok = True
        for camera_id, camera in self.cameras.items():
            try:
                connected = camera.connection(width, height, fps)
            except TypeError:
                try:
                    connected = camera.connection(width, height)
                except Exception as exc:
                    logger.error("Error connecting camera %s: %s", camera_id, exc)
                    connected = False
            except Exception as exc:
                logger.error("Error connecting camera %s: %s", camera_id, exc)
                connected = False
            if not connected:
                ok = False
        if activate:
            self.active = True
        return ok

    def snap(self, camera_id: str, timeout_ms: int = 1000):
        camera = self.get(camera_id)
        if camera is None:
            logger.error("Camera %s not found", camera_id)
            return None
        if hasattr(camera, "snap_frame"):
            return camera.snap_frame(timeout_ms=timeout_ms)
        return self.get_frame(camera_id, copy=True)

    def start_alignment_live(self, camera_id: str) -> bool:
        target = self.get(camera_id)
        if target is None or not hasattr(target, "start_live"):
            return False
        for cid, camera in self.cameras.items():
            if cid != camera_id and getattr(camera, "is_streaming", False):
                stop_live = getattr(camera, "stop_live", None)
                if callable(stop_live):
                    stop_live()
        success = bool(target.start_live())
        if success:
            self._active_alignment_cam = camera_id
        return success

    def stop_alignment_live(self, camera_id: Optional[str] = None):
        targets = (
            {camera_id: self.get(camera_id)} if camera_id else self.cameras
        )
        for cid, camera in targets.items():
            stop_live = getattr(camera, "stop_live", None) if camera else None
            if callable(stop_live):
                stop_live()
            if self._active_alignment_cam == cid:
                self._active_alignment_cam = None

    def get_frame(self, camera_id: str, copy: bool = True):
        camera = self.get(camera_id)
        if camera is None:
            return None
        try:
            frame = camera.get_latest_frame()
            if frame is None:
                return None
            return frame.copy() if copy and hasattr(frame, "copy") else frame
        except Exception as exc:
            logger.error("Error getting frame from camera %s: %s", camera_id, exc)
            return None

    def status(self) -> Dict[str, str]:
        result = {}
        for camera_id, camera in self.cameras.items():
            try:
                result[camera_id] = "ONLINE" if camera and camera.is_connected() else "ERROR"
            except Exception:
                result[camera_id] = "ERROR"
        return result

    def all_connected(self) -> bool:
        statuses = self.status()
        return bool(statuses) and all(value == "ONLINE" for value in statuses.values())

    def reconnect_disconnected(self, width: int = 1920, height: int = 1080, fps: int = 15) -> bool:
        all_ok = True
        for camera_id, camera in self.cameras.items():
            try:
                if camera and camera.is_connected():
                    continue
                reconnect = getattr(camera, "reconnect", None)
                if callable(reconnect):
                    connected = reconnect()
                else:
                    disconnect = getattr(camera, "disconnect", None)
                    if callable(disconnect):
                        disconnect()
                    try:
                        connected = camera.connection(width, height, fps)
                    except TypeError:
                        connected = camera.connection(width, height)
                if not connected:
                    all_ok = False
            except Exception as exc:
                logger.error("Error reconnecting camera %s: %s", camera_id, exc)
                all_ok = False
        return all_ok

    def disconnect_all(self):
        self.active = False
        self.stop_alignment_live()
        for camera_id, camera in self.cameras.items():
            try:
                if camera is not None and callable(getattr(camera, "disconnect", None)):
                    camera.disconnect()
            except Exception as exc:
                logger.error("Error disconnecting camera %s: %s", camera_id, exc)
