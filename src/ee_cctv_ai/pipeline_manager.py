"""Own the async lifecycle of the camera/inference pipeline."""
import asyncio


class PipelineManager:
    def __init__(self, pipeline_factory, camera_urls, shared_data, api_key_provider, nodejs_url, camera_manager=None):
        self.pipeline_factory = pipeline_factory
        self.camera_urls = camera_urls
        self.shared_data = shared_data
        self.api_key_provider = api_key_provider
        self.nodejs_url = nodejs_url
        self.camera_manager = camera_manager
        self.pipeline = None
        self._task = None
        self._lock = asyncio.Lock()

    @property
    def running(self):
        return self._task is not None and not self._task.done()

    async def start(self):
        """Start once; return False if a pipeline task is already active."""
        async with self._lock:
            if self.running:
                return False
            self._task = asyncio.create_task(self._run(), name="cctv-pipeline")
            self.shared_data.set_active_device(1)
            return True

    async def stop(self):
        """Cancel and await the pipeline, allowing its finally block to clean up."""
        async with self._lock:
            task = self._task
            if task is not None and not task.done():
                task.cancel()

        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            if self._task is task:
                self._task = None
            self.pipeline = None
            self.shared_data.set_active_device(0)
            if self.camera_manager is not None:
                self.camera_manager.set_active(False)

    async def _run(self):
        pipeline = None
        try:
            if self.camera_manager is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, self.camera_manager.connect_all, 1920, 1080, 15, False
                )
                self.camera_manager.set_active(True)
                pipeline = self.pipeline_factory(
                    self.camera_urls,
                    self.shared_data,
                    self.api_key_provider(),
                    self.nodejs_url,
                    camera_adapters=list(self.camera_manager.cameras.values()),
                )
            else:
                pipeline = self.pipeline_factory(
                    self.camera_urls,
                    self.shared_data,
                    self.api_key_provider(),
                    self.nodejs_url,
                )
            self.pipeline = pipeline
            await pipeline.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.shared_data.set_LIFE("inactive")
            print(f"CCTV_AI Error: {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception as exc:
                    print(f"Pipeline cleanup warning: {exc}")
            if self.pipeline is pipeline:
                self.pipeline = None
            self.shared_data.set_active_device(0)
            if self.camera_manager is not None:
                self.camera_manager.set_active(False)
