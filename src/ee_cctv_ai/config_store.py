"""Centralized, atomic JSON configuration I/O for the edge device."""
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
DEFAULT_CONFIG = {"cameras": [], "confidence_threshold": 0.5}


def load_config(path=DEFAULT_CONFIG_PATH):
    config = dict(DEFAULT_CONFIG)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            config.update(loaded)
        else:
            print(f"Config at {path} is not a JSON object; using defaults.")
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Config read error at {path}: {exc}; using defaults.")
    return config


def save_config(config, path=DEFAULT_CONFIG_PATH):
    """Atomically replace the config file to avoid partial JSON on power loss."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(config, handle, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise
