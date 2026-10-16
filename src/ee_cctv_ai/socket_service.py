"""Socket.IO connection, registration events, and periodic health reporting."""
import asyncio

from src.ee_cctv_ai.status_service import get_jetson_stats


class SocketStatusService:
    def __init__(self, socket_client, server_url, shared_data, api_key_provider):
        self.client = socket_client
        self.server_url = server_url
        self.shared_data = shared_data
        self.api_key_provider = api_key_provider

        @self.client.event
        async def connect():
            print("Socket.IO connected")
            api_key = self.api_key_provider()
            if api_key:
                await self.client.emit("register-device-status", {"apiKey": api_key})
            else:
                print("Socket.IO connected before device registration.")

        @self.client.event
        async def disconnect():
            print("Socket.IO disconnected")

        @self.client.event
        async def connect_error(data):
            print("Socket.IO connect_error:", data)

    async def connect_forever(self):
        while not self.client.connected:
            try:
                print(f"Attempting Socket.IO connection to {self.server_url}...")
                await self.client.connect(self.server_url, wait_timeout=10)
                print("Socket.IO connected successfully.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Socket.IO connection failed: {exc!r}")
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                await asyncio.sleep(5)

    async def send_status_now(self):
        api_key = self.api_key_provider()
        if not api_key:
            return
        if not self.client.connected:
            print("Socket.IO is not connected; skipping heartbeat.")
            return
        try:
            await self.client.emit(
                "heartbeat",
                get_jetson_stats(self.shared_data.get_LIFE(), self.shared_data),
            )
        except Exception as exc:
            print(f"Failed to send status via Socket.IO: {exc}")

    async def heartbeat_loop(self):
        while True:
            await self.send_status_now()
            await asyncio.sleep(5)
