import asyncio
import httpx


class AlertService:
    def __init__(self, base_url, api_key, max_queue_size=32):
        self.url = f"{base_url.rstrip('/')}/api/device/alarm"
        self.api_key = api_key
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.client = None
        self._task = None
        self.dropped = 0
        self.sent = 0
        self.failed = 0

    async def start(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=2.0)
            )
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="alarm-delivery")

    async def enqueue(self, payload):
        await self.start()
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                pass
            self.dropped += 1
            self.queue.put_nowait(payload)

    async def _run(self):
        while True:
            payload = await self.queue.get()
            try:
                response = await self.client.post(
                    self.url,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    json=payload,
                )
                if response.is_success:
                    self.sent += 1
                else:
                    self.failed += 1
                    print(f"Alarm delivery failed: HTTP {response.status_code}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failed += 1
                print(f"Alarm delivery error: {exc}")
            finally:
                self.queue.task_done()

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def status(self):
        return {
            "queued": self.queue.qsize(),
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
        }
