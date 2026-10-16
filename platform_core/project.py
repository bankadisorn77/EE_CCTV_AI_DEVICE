from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable

from .context import InspectionContext
from .communication import PlatformStatusReporter


class ProjectAdapter(ABC):
    """Project contract consumed by PlatformRuntime.

    Hardware and communication transports are supplied by the project through
    factories. The core never imports a concrete camera, IO driver, or project
    implementation.
    """

    @property
    @abstractmethod
    def project_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def pipeline_definition(self, pipeline_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_step_handlers(self) -> Dict[str, Callable[[InspectionContext, Dict[str, Any]], Any]]:
        raise NotImplementedError

    def camera_definition(self) -> List[Dict[str, Any]]:
        return []

    def create_camera(self, definition: Dict[str, Any]):
        """Create a camera adapter for one camera definition."""
        raise NotImplementedError(
            "Projects with cameras must implement create_camera(definition)."
        )

    def connect_cameras_on_start(self) -> bool:
        """Whether Runtime should connect camera adapters during startup."""
        return True

    def requires_io(self) -> bool:
        """Whether IO hardware is mandatory for this project."""
        return False

    def create_io_manager(self):
        """Return an IO adapter, or None when the project has no IO capability."""
        return None

    def create_status_reporter(self):
        """Return a transport adapter; no transport is required by default."""
        return PlatformStatusReporter()

    def io_definition(self) -> Dict[str, Any]:
        return {}

    def storage_definition(self) -> Dict[str, Any]:
        return {}

    def status_payload(self) -> Dict[str, Any]:
        return {}

    def run(self, runtime):
        raise NotImplementedError

    def shutdown(self, runtime):
        """Optional project cleanup hook."""
        return None

    def run_pipeline(self, pipeline_id: str, context: InspectionContext):
        return None
