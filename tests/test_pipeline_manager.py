"""Unit tests for pipeline lifecycle ownership without camera hardware."""
import asyncio
import unittest

from src.ee_cctv_ai.pipeline_manager import PipelineManager


class FakeSharedData:
    def __init__(self):
        self.active = 0
        self.life = "active"

    def set_active_device(self, value):
        self.active = value

    def set_LIFE(self, value):
        self.life = value


class FakePipeline:
    def __init__(self, *args):
        self.started = asyncio.Event()
        self.cleaned = 0

    async def run(self):
        self.started.set()
        await asyncio.Event().wait()

    def cleanup(self):
        self.cleaned += 1


class PipelineManagerTests(unittest.TestCase):
    def test_start_is_idempotent_and_stop_cleans_pipeline(self):
        async def run_test():
            shared = FakeSharedData()
            instances = []

            def factory(*args):
                pipeline = FakePipeline(*args)
                instances.append(pipeline)
                return pipeline

            manager = PipelineManager(factory, ["rtsp://cam"], shared, lambda: "key", "http://server")
            self.assertTrue(await manager.start())
            self.assertFalse(await manager.start())
            for _ in range(20):
                if instances:
                    break
                await asyncio.sleep(0)
            self.assertEqual(len(instances), 1)
            await instances[0].started.wait()
            self.assertTrue(manager.running)
            self.assertEqual(shared.active, 1)

            await manager.stop()
            self.assertFalse(manager.running)
            self.assertIsNone(manager.pipeline)
            self.assertEqual(instances[0].cleaned, 1)
            self.assertEqual(shared.active, 0)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
