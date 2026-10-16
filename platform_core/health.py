from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, Any, Callable, Optional

if TYPE_CHECKING:
    from .camera_manager import CameraManager
    from .io_manager import BaseIOManager

logger = logging.getLogger("HealthMonitor")


class HealthMonitor:
    """Background health checks for the capabilities installed in a runtime.

    Missing optional capabilities are omitted from status rather than reported
    as failed hardware. Camera and IO adapters remain responsible for their
    own low-level connection checks.
    """

    def __init__(
        self,
        camera_manager: Optional["CameraManager"] = None,
        io_manager: Optional["BaseIOManager"] = None,
        check_interval: float = 3.0,
        on_status_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.camera_manager = camera_manager
        self.io_manager = io_manager
        self.check_interval = max(0.1, float(check_interval))
        self.on_status_change = on_status_change
        self._monitors: Dict[str, Callable[[], bool]] = {}
        self._current_status: Dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_check(self, name: str, check_func: Callable[[], bool]):
        self._monitors[name] = check_func
        self._current_status[name] = True

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="HealthMonitorThread"
        )
        self._thread.start()
        logger.info("HealthMonitor background service started.")

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("HealthMonitor background service stopped.")

    def check_now(self) -> Dict[str, bool]:
        results = {}
        for name, func in self._monitors.items():
            try:
                results[name] = bool(func())
            except Exception as exc:
                logger.error("Error checking health for '%s': %s", name, exc)
                results[name] = False
        return results

    def is_healthy(self) -> bool:
        def ok(value):
            return all(value.values()) if isinstance(value, dict) else bool(value)
        return all(ok(value) for value in self._current_status.values()) if self._current_status else True

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "healthy": self.is_healthy(),
            "details": dict(self._current_status),
            "timestamp": time.time(),
        }

    def _publish(self, new_status):
        if new_status == self._current_status:
            return
        self._current_status = new_status
        if self.on_status_change:
            try:
                self.on_status_change(self.status)
            except Exception:
                logger.exception("Error executing on_status_change callback")

    @staticmethod
    def _connected(camera):
        try:
            return bool(camera and camera.is_connected())
        except Exception:
            return False

    def _run_loop(self):
        while not self._stop_event.is_set():
            status: Dict[str, Any] = {}
            camera_statuses = {}
            camera_active = bool(self.camera_manager is not None and getattr(self.camera_manager, "active", True))
            if camera_active:
                for camera_id, camera in self.camera_manager.cameras.items():
                    camera_statuses[camera_id] = self._connected(camera)
                status["camera"] = camera_statuses

            io_alive = None
            if self.io_manager is not None:
                try:
                    io_alive = bool(self.io_manager.is_connected())
                except Exception:
                    io_alive = False
                status["gpio"] = io_alive

            # Run project-defined checks as well as the standard hardware checks.
            # Standard camera/IO checks above publish richer per-device details.
            for name, check in self._monitors.items():
                if name in ("camera", "gpio"):
                    continue
                try:
                    status[name] = bool(check())
                except Exception:
                    logger.exception("Health check failed for %s", name)
                    status[name] = False

            self._publish(status)

            if self.io_manager is not None and not io_alive:
                logger.warning("[HealthMonitor] IO disconnected; attempting reconnect.")
                try:
                    try:
                        self.io_manager.connect(max_retries=1)
                    except TypeError:
                        self.io_manager.connect()
                except Exception:
                    logger.exception("[HealthMonitor] IO reconnect failed")

            if camera_active and self.camera_manager is not None and any(not ok for ok in camera_statuses.values()):
                try:
                    self.camera_manager.reconnect_disconnected()
                except Exception:
                    logger.exception("[HealthMonitor] Camera reconnect failed")

            if self._stop_event.wait(self.check_interval):
                break
