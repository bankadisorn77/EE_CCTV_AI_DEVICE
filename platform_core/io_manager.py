# platform_core/io_manager.py
from abc import ABC, abstractmethod
import logging
import traceback
logger = logging.getLogger("IOManager")


class BaseIOManager(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def read_input(self, channel: int) -> int:
        pass

    @abstractmethod
    def write_output(self, channel: int, value: bool) -> bool:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass


class AdvantechIOManager(BaseIOManager):

    def __init__(self, device: str = "USB-4761,BID#0", auto_connect: bool = False):
        self.device = device
        self.gpio = None
        self._is_connected = False
        
        if auto_connect:
            self.connect()

    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        try:
            from .module.gpio.advantech import GPIO
            if self.gpio:
                self.close()
            self._is_connected = False
            attempts = 0
            while not self._is_connected and attempts < max_retries:
                attempts += 1
                logger.info(f"[IOManager] Attempting to connect device {self.device} ({attempts}/{max_retries})...")
                self.gpio = GPIO(device=self.device)
                self._is_connected = getattr(self.gpio, "is_connected", False)

                if self._is_connected:
                    logger.info(f"[IOManager] Connected successfully to device {self.device}.")
                    return True
                else:
                    logger.warning(f"[IOManager] Connection attempt {attempts} failed. Retrying in {retry_delay}s...")
                    import time
                    if attempts < max_retries:
                        time.sleep(retry_delay)

            logger.error(f"[IOManager] Failed to connect to {self.device} after {max_retries} attempts.")
            self._is_connected = False
            return False

        except Exception as e:
            logger.error(f"[IOManager] Connect failed: {e}")
            self._is_connected = False
            return False

    def read_input(self, channel: int) -> int:
        if not self._is_connected or not self.gpio:
            return 0
        try:
            val = self.gpio.readInput(channel)
            if val is None:
                self._is_connected = False
                return 0
            return val
        except Exception as e:
            logger.error(f"[IOManager] read_input error: {e}")
            self._is_connected = False
            return 0

    def write_output(self, channel: int, value: bool) -> bool:
        # if channel in (4, 5):
        #     print(f"[IO] ch{channel}={value} <- {traceback.extract_stack(limit=3)[0].name}")
        if not self._is_connected or not self.gpio:
            return False
        try:
            success = bool(self.gpio.outputWrite(channel, on=value))
            if not success:
                self._is_connected = False
            return success
        except Exception as e:
            logger.error(f"[IOManager] write_output error: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        if not self._is_connected or not self.gpio:
            return False
        alive = self.gpio.ping()
        self._is_connected = alive
        return alive

    def close(self):
        if self.gpio:
            try:
                self.gpio.closeIO()
            except Exception:
                pass
        self._is_connected = False
        self.gpio = None


class DummyIOManager(BaseIOManager):
    def connect(self) -> bool:
        return True

    def read_input(self, channel: int) -> int:
        return 0

    def write_output(self, channel: int, value: bool) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def close(self):
        pass
