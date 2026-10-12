"""Tests for bounded, nonblocking alert queue behavior."""
import asyncio
import unittest

from src.ee_cctv_ai.alert_service import AlertService


class AlertServiceTests(unittest.TestCase):
    def test_queue_overflow_drops_oldest_payload(self):
        async def run_test():
            service = AlertService("http://localhost:8000", "test-key", max_queue_size=2)

            async def no_start():
                return None

            # Isolate queue behavior from HTTP/network activity.
            service.start = no_start
            service.queue.put_nowait({"id": 1})
            service.queue.put_nowait({"id": 2})
            await service.enqueue({"id": 3})

            self.assertEqual(service.dropped, 1)
            self.assertEqual(service.queue.qsize(), 2)
            self.assertEqual((await service.queue.get())["id"], 2)
            service.queue.task_done()
            self.assertEqual((await service.queue.get())["id"], 3)
            service.queue.task_done()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
