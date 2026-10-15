"""Bind the existing CCTV service to the reusable PlatformRuntime.

The high-throughput inference pipeline remains owned by PipelineManager. This
adapter deliberately does not create a second set of RTSP readers.
"""
from platform_core.project import ProjectAdapter


class CCTVProjectAdapter(ProjectAdapter):
    project_id = "EE_CCTV_AI_DEVICE"

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
