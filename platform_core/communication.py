import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Communication")


class PlatformStatusReporter:
    """Optional status-reporting adapter.

    The platform core does not select a transport. A project may inject a
    sender (shared memory, MQTT, HTTP, etc.); without one, status reporting is
    safely disabled and lifecycle behavior remains independent of transport.
    """

    def __init__(self, sender=None):
        self.sender = sender
        if self.sender is not None:
            connect = getattr(self.sender, "connect", None)
            if callable(connect):
                try:
                    connect()
                except Exception:
                    logger.exception("Status sender initialization failed")
                    self.sender = None

    def report(self, payload: Optional[Dict[str, Any]] = None):
        if self.sender is None or payload is None:
            return
        try:
            data = dict(payload)
            data["last_update"] = time.time()
            self.sender.send_event(data)
        except Exception:
            logger.exception("Status report failed")

    def close(self):
        if self.sender is None:
            return
        close = getattr(self.sender, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Status sender close failed")
        self.sender = None
