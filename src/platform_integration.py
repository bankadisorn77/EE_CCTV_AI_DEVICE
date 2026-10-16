"""Bind the EE CCTV project to the reusable Platform Core contracts.

PlatformRuntime owns camera adapters and health lifecycle. This adapter exposes
project-level pipeline controls so MQTT and other transports do not own runtime
or hardware lifecycle directly.
"""
from platform_core.project import ProjectAdapter
from src.ee_cctv_ai.acquistion import Acquisition
import asyncio


class CCTVProjectAdapter(ProjectAdapter):
    project_id = "EE_CCTV_AI_DEVICE"

    def __init__(self, rtsp_urls):
        self.rtsp_urls = list(rtsp_urls)
        self._pipeline_manager = None
        self._camera_manager = None
        self._lifecycle_lock = None

    def configure_pipeline(self, pipeline_factory, shared_data, api_key_provider, nodejs_url, camera_manager):
        """Create the project pipeline using the adapters owned by PlatformRuntime."""
        from src.ee_cctv_ai.pipeline_manager import PipelineManager

        self._camera_manager = camera_manager
        self._pipeline_manager = PipelineManager(
            pipeline_factory, self.rtsp_urls, shared_data, api_key_provider,
            nodejs_url, camera_manager=camera_manager,
        )
        return self._pipeline_manager

    def _manager(self):
        if self._pipeline_manager is None:
            raise RuntimeError("CCTV PipelineManager has not been bound.")
        return self._pipeline_manager

    @property
    def running(self):
        return bool(self._pipeline_manager and self._pipeline_manager.running)

    @property
    def pipeline(self):
        return self._pipeline_manager.pipeline if self._pipeline_manager else None

    async def start(self):
        """Activate platform-managed cameras, then start the existing AI pipeline."""
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        async with self._lifecycle_lock:
            manager = self._manager()
            if manager.running:
                return False
            camera_manager = self._camera_manager
            if camera_manager is not None:
                loop = asyncio.get_running_loop()
                # RTSP connection attempts may block briefly; keep them off the event loop.
                await loop.run_in_executor(
                    None, camera_manager.connect_all, 1920, 1080, 15, False
                )
                camera_manager.set_active(True)
            started = await manager.start()
            if not started and camera_manager is not None and not manager.running:
                camera_manager.set_active(False)
            return started

    async def stop(self):
        """Stop inference first, then deactivate platform-managed cameras."""
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        async with self._lifecycle_lock:
            manager = self._manager()
            await manager.stop()
            if self._camera_manager is not None:
                self._camera_manager.set_active(False)
                self._camera_manager.disconnect_all()
            return None

    def camera_definition(self):
        return [
            {"id": "cam{}".format(index + 1), "url": url, "enabled": True}
            for index, url in enumerate(self.rtsp_urls)
        ]

    def create_camera(self, definition):
        return Acquisition(definition.get("url", ""), key=definition["id"])

    def connect_cameras_on_start(self):
        # CCTV live streams are activated on project pipeline start, not API startup.
        return False

    def pipeline_definition(self, pipeline_id):
        # CCTV uses the existing optimized inference implementation rather than
        # the generic step engine; the platform core remains the lifecycle owner.
        return {}

    def get_step_handlers(self):
        return {}

    def run(self, runtime):
        # FastAPI owns the event loop; pipeline execution is on-demand through
        # start()/stop() and is never started implicitly during app startup.
        return None

    def status_payload(self):
        return {"project": self.project_id}

    def shutdown(self, runtime):
        # FastAPI lifespan awaits stop() before calling PlatformRuntime.stop().
        return None
