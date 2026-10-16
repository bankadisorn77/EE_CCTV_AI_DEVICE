"""Hardware-independent integration tests for the CCTV Platform Core adapter."""
import asyncio
import unittest

from src.platform_integration import CCTVProjectAdapter


class FakeSharedData:
    def __init__(self):
        self.active = 0
        self.life = "active"

    def set_active_device(self, value):
        self.active = value

    def set_LIFE(self, value):
        self.life = value


class FakeCameraManager:
    def __init__(self):
        self.cameras = {"cam1": object()}
        self.active = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect_all(self, width, height, fps, activate=False):
        self.connect_calls += 1
        return True

    def set_active(self, active):
        self.active = bool(active)

    def disconnect_all(self):
        self.disconnect_calls += 1
        self.active = False


class FakePipeline:
    def __init__(self, *args, **kwargs):
        self.started = asyncio.Event()
        self.cleaned = 0
        self.cameras = []

    async def run(self):
        self.started.set()
        await asyncio.Event().wait()

    def cleanup(self):
        self.cleaned += 1


class PlatformIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_project_owns_camera_and_pipeline_lifecycle(self):
        project = CCTVProjectAdapter(["rtsp://camera"])
        camera_manager = FakeCameraManager()
        shared = FakeSharedData()
        instances = []

        def factory(*args, **kwargs):
            pipeline = FakePipeline(*args, **kwargs)
            instances.append(pipeline)
            return pipeline

        manager = project.configure_pipeline(
            factory, shared, lambda: "", "http://localhost", camera_manager
        )

        self.assertTrue(await project.start())
        self.assertFalse(await project.start())
        for _ in range(20):
            if instances:
                break
            await asyncio.sleep(0)
        await instances[0].started.wait()

        self.assertEqual(camera_manager.connect_calls, 1)
        self.assertTrue(camera_manager.active)
        self.assertTrue(project.running)
        self.assertIs(project.pipeline, manager.pipeline)

        await project.stop()
        self.assertFalse(project.running)
        self.assertFalse(camera_manager.active)
        self.assertEqual(camera_manager.disconnect_calls, 1)
        self.assertEqual(instances[0].cleaned, 1)

    def test_camera_definitions_preserve_configured_camera_slots(self):
        project = CCTVProjectAdapter(["rtsp://camera-1", "", "rtsp://camera-3"])
        definitions = project.camera_definition()
        self.assertEqual([item["id"] for item in definitions], ["cam1", "cam2", "cam3"])
        self.assertTrue(all(item["enabled"] for item in definitions))


if __name__ == "__main__":
    unittest.main()
