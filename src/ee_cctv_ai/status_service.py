"""Device health/statistics helpers sent through Socket.IO."""
import psutil


def get_jetson_stats(status, shared_data):
    def read_temp(zone_path):
        try:
            with open(zone_path, "r", encoding="utf-8") as handle:
                return float(handle.read().strip()) / 1000.0
        except (OSError, ValueError):
            return 0.0

    memory = psutil.virtual_memory()
    return {
        "runningStatus": status,
        "cpuTemp": read_temp("/sys/class/thermal/thermal_zone0/temp"),
        "gpuTemp": read_temp("/sys/class/thermal/thermal_zone1/temp"),
        "memoryFree": memory.available // (1024 * 1024),
        "memoryUsed": memory.used // (1024 * 1024),
        "aiVersion": "v1.12.3",
        "activeDevice": shared_data.get_active_device(),
    }
