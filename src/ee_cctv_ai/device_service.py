"""Device registration and API-key verification workflow."""
import asyncio
import json
from pathlib import Path

import httpx


async def register_device(
    device_config_path,
    initial_frame_capture,
    rtsp_urls,
    nodejs_url,
    get_network_info,
    get_api_key,
    set_api_key,
    socket_client,
):
    device_config_path = Path(device_config_path)
    loop = asyncio.get_running_loop()
    image_base64_frames = await loop.run_in_executor(
        None, initial_frame_capture, rtsp_urls
    )

    async def register(device_config):
        print("Device config missing API key")
        lan_ip, lan_mac = get_network_info()
        device_ip = device_config.get("ip") or lan_ip
        device_mac = device_config.get("mac") or lan_mac
        device_config["ip"] = device_ip
        device_config["mac"] = device_mac
        payload = {"mac": device_mac, "ip": device_ip, "frame": image_base64_frames}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{nodejs_url}/api/device/register", json=payload
                )
            if response.status_code != 200:
                print(f"Register response status: {response.status_code}")
                return
            api_key = response.json().get("apiKey")
            if not api_key:
                print("Registration response did not include an API key.")
                return
            device_config["apiKey"] = api_key
            set_api_key(api_key)
            try:
                with device_config_path.open("w", encoding="utf-8") as handle:
                    json.dump(device_config, handle, indent=4)
                print(f"Device registered: IP={device_ip}, MAC={device_mac}")
            except OSError as exc:
                print(f"Error saving device config: {exc}")
        except Exception as exc:
            print(f"Error registering device: {exc}")

    while True:
        try:
            try:
                with device_config_path.open("r", encoding="utf-8") as handle:
                    device_config = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Device config error: {exc}")
                device_config = {}

            saved_api_key = device_config.get("apiKey")
            if not saved_api_key:
                await register(device_config)
            else:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(
                            f"{nodejs_url}/api/device/check-api-key",
                            headers={"Content-Type": "application/json", "x-api-key": saved_api_key},
                        )
                    if response.status_code == httpx.codes.OK:
                        set_api_key(saved_api_key)
                        try:
                            async with httpx.AsyncClient(timeout=10.0) as client:
                                image_response = await client.post(
                                    f"{nodejs_url}/api/device/re-captureImage",
                                    headers={"Content-Type": "application/json", "x-api-key": saved_api_key},
                                    json={"frame": image_base64_frames},
                                )
                            if image_response.status_code == httpx.codes.OK:
                                print("Re-capture image success")
                            else:
                                print("Re-capture image failed")
                        except Exception as exc:
                            print(f"Re-capture image failed: {exc}")
                        print("Device already registered with valid API key.")
                    else:
                        print("API key invalid; registering device again.")
                        await register(device_config)
                except Exception as exc:
                    print(f"Cannot verify API key; using cached key if available: {exc}")
                    set_api_key(saved_api_key)

            api_key = get_api_key()
            print(f"API key configured: {bool(api_key)}")
            if api_key:
                if socket_client.connected:
                    try:
                        await socket_client.emit(
                            "register-device-status", {"apiKey": api_key}
                        )
                    except Exception as exc:
                        print(f"Socket.IO registration emit failed: {exc}")
                print("Registration sequence passed.")
                return

            print("Retrying registration in 5 seconds...")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Fatal error in device registration: {exc}")
            await asyncio.sleep(5)
