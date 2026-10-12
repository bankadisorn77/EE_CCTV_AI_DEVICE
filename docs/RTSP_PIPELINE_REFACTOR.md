# RTSP pipeline refactor notes

## Included in this increment
- FastAPI startup no longer waits indefinitely for device registration.
- Initial frame capture is isolated in `frame_capture.py` and runs in an executor, not on the FastAPI event loop.
- Start/stop tracks the pipeline task, awaits cancellation, and cleans up from `finally`.
- Camera acquisition opens lazily and reconnects independently; one failed camera does not abort other cameras.
- Per-camera queues retain only the latest pending frame.
- Shared YOLO inference is serialized while camera capture stays independent.
- Debug image writes are disabled by default; set `EE_CCTV_SAVE_DEBUG_IMAGES=1` to enable.
- Alarm delivery is moved to a bounded queue with a reusable HTTP client; slow webserver calls no longer stall detection workers.
- `GET /status` reports pipeline state, per-camera connectivity/frame freshness and alert queue counters.
- The committed device API key is cleared in this branch. Rotate/revoke the old key; this does not erase it from Git history.

## Must validate on Jetson
1. Confirm `models/best_V6.pt` exists under repository-root `models/`.
2. Confirm OpenCV/GStreamer supports `nvv4l2decoder` on the installed JetPack image.
3. Test four real RTSP streams, not only MP4 files.
4. Disconnect one camera; confirm other cameras continue and the failed camera reconnects.
5. Repeat start/stop and monitor threads, RAM, GPU memory, FPS and latency over a multi-hour run.
6. Verify the server's registration API issues a new key when `apiKey` is blank.
7. Run `python -m compileall src` and hardware-in-the-loop tests.
8. Confirm the webserver alarm endpoint accepts the existing payload and monitor `GET /status` while the webserver is unavailable.

## Next slices
- Split routes, configuration, pipeline orchestration, camera readers, inference scheduling, detection rules and alert delivery into dedicated modules.
- Add mocked tests for alert queue overflow, task cancellation and pipeline lifecycle.
- Add a four-stream RTSP soak test on the target Jetson.
