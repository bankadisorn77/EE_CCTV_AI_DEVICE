"""Bind the existing CCTV project and its RTSP adapters to PlatformRuntime.

The same Acquisition instances are shared with PipelineManager; no duplicate
RTSP readers are created.
"""
from platform_core.project import ProjectAdapter\nfrom src.ee_cctv_ai.acquistion import Acquisition\n

class CCTVProjectAdapter(ProjectAdapter):
    project_id = "EE_CCTV_AI_DEVICE"

    def __init__(self, rtsp_urls):
        self.rtsp_urls = list(rtsp_urls)

    def camera_definition(self):
        return [
            {"id": "cam{}".format(index + 1), "url": url, "enabled": True}
            for index, url in enumerate(self.rtsp_urls)
        ]

    def create_camera(self, definition):
        return Acquisition(definition.get("url", ""), key=definition["id"])

    def connect_cameras_on_start(self):
        # RTSP streams are activated only when PipelineManager starts AI.
        return False

    def pipeline_definition(self, pipeline_id):
        return {}

    def get_step_handlers(self):
        return {}

    def run(self, runtime):
        # FastAPI owns the application event loop. PipelineManager starts and
        # stops the inference coroutine on demand within that loop.
        return None

    def status_payload(self):
        return {"project": self.project_id}
