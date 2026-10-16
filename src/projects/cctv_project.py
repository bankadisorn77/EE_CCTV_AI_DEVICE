"""EE CCTV project implementation for the reusable Platform Core.

The adapter is the project boundary: PlatformRuntime owns generic lifecycle,
camera health and reconnect; this class owns the CCTV inference lifecycle and
retains the optimized batched TensorRT/Ultralytics pipeline.
"""
import asyncio
import logging

from platform_core.project import ProjectAdapter
from src.ee_cctv_ai.acquistion import Acquisition

logger = logging.getLogger("EE_CCTV.Project")


class CCTVProjectAdapter(ProjectAdapter):
    project_id = "EE_CCTV_AI_DEVICE"

    def __init__(self, rtsp_urls):
        self.rtsp_urls = list(rtsp_urls)
        self._pipeline_factory = None
        self._shared_data = None
        self._api_key_provider = None
        self._nodejs_url = None
        self._camera_manager = None
        self._pipeline = None
        self._pipeline_task = None
        # Created lazily inside the active event loop (compatible with Python 3.8).\n        self._lifecycle_lock = None

    def bind_services(self, pipeline_factory, shared_data, api_key_provider, nodejs_url,
                      camera_manager):
        """Bind project-specific services after PlatformRuntime creates adapters."""
        if self._pipeline_task is not None and not self._pipeline_task.done():
            raise RuntimeError("Cannot reconfigure CCTV services while running.")
        self._pipeline_factory = pipeline_factory
        self._shared_data = shared_data
        self._api_key_provider = api_key_provider
        self._nodejs_url = nodejs_url
        self._camera_manager = camera_manager

    @property
    def running(self):
        return self._pipeline_task is not None and not self._pipeline_task.done()

    @property
    def pipeline(self):
        return self._pipeline

    def _require_bound(self):
        if any(value is None for value in (
            self._pipeline_factory, self._shared_data, self._api_key_provider,
            self._nodejs_url, self._camera_manager,
        )):
            raise RuntimeError("CCTVProjectAdapter services have not been bound.")

    def _get_lifecycle_lock(self):
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    async def start(self):
        """Start shared camera adapters and the CCTV inference pipeline once."""
        async with self._get_lifecycle_lock():
            self._require_bound()
            if self.running:
                return False

            loop = asyncio.get_running_loop()
            # Acquisition.connection() starts its single reader thread and waits
            # briefly for a connection. Keep RTSP work off FastAPI's event loop.
            await loop.run_in_executor(
                None, self._camera_manager.connect_all, 1920, 1080, 15, False
            )
            self._camera_manager.set_active(True)
            self._shared_data.set_active_device(1)
            self._pipeline_task = asyncio.create_task(
                self._run_inference(), name="cctv-platform-project"
            )
            return True

    async def stop(self):
        """Stop inference and release the same camera adapters owned by Runtime."""
        async with self._lifecycle_lock:
            task = self._pipeline_task
            if task is not None and not task.done():
                task.cancel()

        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lifecycle_lock:
            if self._pipeline_task is task:
                self._pipeline_task = None
            self._pipeline = None
            if self._shared_data is not None:
                self._shared_data.set_active_device(0)
            if self._camera_manager is not None:
                self._camera_manager.set_active(False)
                self._camera_manager.disconnect_all()

    async def _run_inference(self):
        pipeline = None
        try:
            pipeline = self._pipeline_factory(
                self.rtsp_urls,
                self._shared_data,
                self._api_key_provider(),
                self._nodejs_url,
                camera_adapters=list(self._camera_manager.cameras.values()),
            )
            self._pipeline = pipeline
            await pipeline.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("CCTV inference pipeline failed")
            if self._shared_data is not None:
                self._shared_data.set_LIFE("inactive")
        finally:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception:
                    logger.exception("CCTV pipeline cleanup failed")
            if self._pipeline is pipeline:
                self._pipeline = None
            if self._shared_data is not None:
                self._shared_data.set_active_device(0)
            if self._camera_manager is not None:
                self._camera_manager.set_active(False)

    # ---- Platform Core project contract ---------------------------------
    def camera_definition(self):
        # Preserve each configured slot, including blank URLs, for stable UI IDs.
        return [
            {"id": "cam{}".format(index + 1), "url": url, "enabled": True}
            for index, url in enumerate(self.rtsp_urls)
        ]

    def create_camera(self, definition):
        return Acquisition(definition.get("url", ""), key=definition["id"])

    def connect_cameras_on_start(self):
        # Live cameras are activated on project start, not web-server startup.
        return False

    def pipeline_definition(self, pipeline_id):
        # CCTV uses its optimized batch inference engine rather than the generic
        # inspection-step engine. PlatformRuntime still owns lifecycle and health.
        return {}

    def get_step_handlers(self):
        return {}

    def run(self, runtime):
        # FastAPI drives the event loop; commands start/stop this project on demand.
        return None

    def status_payload(self):
        return {"project": self.project_id, "pipeline_running": self.running}

    def shutdown(self, runtime):
        # Runtime invokes this synchronously; FastAPI lifespan awaits stop() first.
        return None
