import logging
import threading
import uuid
from typing import List, Optional

from .camera_manager import CameraManager
from .context import InspectionContext
from .health import HealthMonitor
from .pipeline import PipelineEngine
from .project import ProjectAdapter

logger = logging.getLogger("PlatformRuntime")


class PlatformRuntime:
    """Owns lifecycle and shared services for exactly one Project instance."""

    def __init__(
        self,
        project: ProjectAdapter,
        camera_manager: Optional[CameraManager] = None,
        io_manager=None,
        health_monitor: Optional[HealthMonitor] = None,
        status_reporter=None,
        health_check_interval: float = 3.0,
    ):
        self.project = project
        self.pipeline_engine = PipelineEngine()
        self.reporter = status_reporter or project.create_status_reporter()
        self.camera_manager = camera_manager or CameraManager.from_definitions(
            project.camera_definition(), factory=project.create_camera
        )
        self.io_manager = io_manager if io_manager is not None else project.create_io_manager()
        self.io_required = bool(project.requires_io())
        if self.io_required and self.io_manager is None:
            raise ValueError("This project requires IO but create_io_manager() returned None.")
        self.health_monitor = health_monitor or HealthMonitor(
            on_status_change=self._on_health_change,
            io_manager=self.io_manager,
            camera_manager=self.camera_manager,
            check_interval=health_check_interval,
        )
        storage = project.storage_definition().get("base_dir") if hasattr(project, "storage_definition") else None
        self.storage = None
        if storage:
            # Keep optional image dependencies out of projects that do not store frames.
            from .storage import ImageStorage
            self.storage = ImageStorage(storage)
        self._running = False
        self._last_reported = None
        self._status_lock = threading.Lock()
        self._register_handlers()
        self._register_health_checks()

    def _register_handlers(self):
        for step_type, handler in self.project.get_step_handlers().items():
            self.pipeline_engine.register(step_type, handler)

    def _register_health_checks(self):
        if self.camera_manager.cameras:
            self.health_monitor.register_check("camera", self.camera_manager.all_connected)
        if self.io_manager is not None:
            self.health_monitor.register_check("gpio", self.io_manager.is_connected)

    def _on_health_change(self, status):
        self.update_status()

    def start(self, width=1920, height=1080, fps=15):
        camera_ok = True
        io_ok = not self.io_required
        try:
            if self.camera_manager.cameras and self.project.connect_cameras_on_start():
                camera_ok = self.camera_manager.connect_all(width, height, fps, activate=True)
            if self.io_manager is not None:
                try:
                    io_ok = bool(self.io_manager.connect())
                except Exception:
                    logger.exception("Failed to connect project IO")
                    io_ok = False
            self.health_monitor.start()
            self._running = True
            self.update_status(program_status="RUNNING")
        except Exception:
            logger.exception("Failed to start platform runtime")
        return bool(camera_ok and io_ok)

    def hardware_ready(self) -> bool:
        if not self._running:
            return False
        details = self.health_monitor.status.get("details", {})
        if self.camera_manager.cameras:
            if not self.camera_manager.active:
                return False
            cameras = details.get("camera")
            if not isinstance(cameras, dict) or not cameras or not all(cameras.values()):
                return False
        if self.io_required and not bool(details.get("gpio")):
            return False
        return True

    def run(self):
        if not self._running:
            raise RuntimeError("PlatformRuntime.start() must be called before run().")
        return self.project.run(self)

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            self.project.shutdown(self)
        except Exception:
            logger.exception("Project shutdown failed")
        self.health_monitor.stop()
        self.camera_manager.disconnect_all()
        if self.io_manager is not None:
            close = getattr(self.io_manager, "close", None)
            if not callable(close):
                close = getattr(self.io_manager, "closeIO", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close project IO")
        self.update_status(program_status="OFF")
        close_reporter = getattr(self.reporter, "close", None)
        if callable(close_reporter):
            close_reporter()

    def update_status(self, **kwargs):
        details = self.health_monitor.status.get("details", {})
        camera_statuses = {}
        if isinstance(details.get("camera"), dict):
            camera_statuses = dict(details["camera"])
        elif self.camera_manager.cameras:
            camera_statuses = {
                camera_id: bool(cam and getattr(cam, "is_connected", lambda: False)())
                for camera_id, cam in self.camera_manager.cameras.items()
            }
        status = {"program_status": "RUNNING" if self._running else "OFF", "camera": camera_statuses}
        if self.io_manager is not None:
            status["gpio"] = bool(details.get("gpio", self.io_manager.is_connected()))
        status.update(self.project.status_payload())
        status.update(kwargs)
        with self._status_lock:
            if status == self._last_reported:
                return
            self._last_reported = dict(status)
        self.reporter.report(status)

    def run_cycle(self, pipeline_id: str, cycle_id: Optional[str] = None,
                  cam_active: Optional[List[str]] = None) -> InspectionContext:
        context = InspectionContext(
            project_id=self.project.project_id,
            pipeline_id=pipeline_id,
            device_id=getattr(self.project, "device_id", "edge_device"),
            cycle_id=cycle_id or uuid.uuid4().hex,
        )
        definition = self.project.pipeline_definition(pipeline_id)
        if not definition:
            return context
        target_cameras = cam_active if cam_active is not None else self.camera_manager.ids()
        for camera_id in target_cameras:
            if camera_id not in self.camera_manager.ids():
                continue
            frame = self.camera_manager.snap(camera_id)
            if frame is not None:
                context.frames[camera_id] = frame
                if self.storage:
                    try:
                        self.storage.save_cycle_image(frame, context.cycle_id, camera_id)
                    except Exception as exc:
                        logger.warning("Failed to save cycle image: %s", exc)
            else:
                logger.warning("Failed to snap frame from %s", camera_id)
        return self.pipeline_engine.run(definition, context)
