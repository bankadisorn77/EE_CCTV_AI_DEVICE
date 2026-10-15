import asyncio
import unittest

from src.ee_cctv_ai.mqtt_control import MqttControlService


class FakePipelineManager:
    def __init__(self):
        self.running = False
        self.starts = 0
        self.stops = 0
        self.pipeline = None

    async def start(self):
        if self.running:
            return False
        self.running = True
        self.starts += 1
        return True

    async def stop(self):
        self.running = False
        self.stops += 1


class FakeMqttClient:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.messages.append((topic, payload, qos, retain))


class MqttControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_command_id_does_not_start_twice(self):
        manager = FakePipelineManager()
        service = MqttControlService(manager)
        service._dispatch_lock = asyncio.Lock()
        service.client = FakeMqttClient()

        command = {"command": "start", "command_id": "cmd-1"}
        await service._dispatch(command)
        await service._dispatch(command)

        self.assertEqual(manager.starts, 1)
        acknowledgements = [
            item for item in service.client.messages
            if item[0] == service.ack_topic
        ]
        self.assertEqual(len(acknowledgements), 2)

    async def test_start_and_stop_are_idempotent(self):
        manager = FakePipelineManager()
        service = MqttControlService(manager)
        service._dispatch_lock = asyncio.Lock()
        service.client = FakeMqttClient()

        await service._dispatch({"command": "start", "command_id": "start-1"})
        await service._dispatch({"command": "start", "command_id": "start-2"})
        await service._dispatch({"command": "stop", "command_id": "stop-1"})
        await service._dispatch({"command": "stop", "command_id": "stop-2"})

        self.assertEqual(manager.starts, 1)
        self.assertEqual(manager.stops, 1)
        self.assertFalse(manager.running)


if __name__ == "__main__":
    unittest.main()
