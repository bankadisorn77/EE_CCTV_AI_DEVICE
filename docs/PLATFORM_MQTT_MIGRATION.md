# CCTV Platform and MQTT Integration

## Scope of this branch

This branch integrates the universal Platform Core with the existing CCTV
service and adds an MQTT command/status bridge. The existing FastAPI routes,
Socket.IO client, Web Server registration, and alert delivery remain unchanged.
Web Server migration is Phase 4 and is intentionally out of scope.

The PlatformRuntime owns process-level lifecycle and HealthMonitor. The existing
PipelineManager continues to own the async inference pipeline and RTSP readers.
This avoids opening duplicate streams or changing latest-frame, round-robin,
batching, and GPU-executor behavior.

## Enable MQTT

Set these environment variables on the CCTV service:

- `EE_CCTV_MQTT_BROKER` — required to enable MQTT.
- `EE_CCTV_MQTT_PORT` — default `1883`.
- `EE_CCTV_MQTT_USERNAME` and `EE_CCTV_MQTT_PASSWORD` — optional credentials.
- `EE_CCTV_MQTT_CLIENT_ID` — default `ee-cctv-ai-device`.
- `EE_CCTV_MQTT_TOPIC_PREFIX` — default `devices/cctv`.

Topics:

- `<prefix>/command`: commands published by the controller.
- `<prefix>/ack`: command acknowledgements.
- `<prefix>/status`: retained status snapshots and periodic heartbeat.

Example command:

```json
{"command_id":"cmd-001","command":"start"}
```

Supported commands are `start`, `stop`, and `get_status` (also `status`).
Commands with the same non-empty `command_id` are de-duplicated in a bounded
in-memory cache. Start and stop are idempotent even without a command ID.
Acknowledgements include `ok`, `command`, `message` or `error`,
`command_id`, and `timestamp`.

Status includes service identity, pipeline-running state, camera connection,
frame ID, frame age, capture FPS, inference FPS, and timestamp.

## Phase boundaries

- Phase 0 (SFS branch): universalize Platform Core hardware and transport contracts.
- Phase 1 (this branch): bind CCTV service lifecycle to PlatformRuntime without
  creating a second RTSP reader or replacing the inference pipeline.
- Phase 2 (this branch): add MQTT command handling and status publication.
- Phase 3 (this branch): idempotent commands, MQTT reconnect via the client
  network loop, bounded command de-duplication, periodic status, and tests.
- Phase 4 (later): change the Web Server to use MQTT instead of the existing
  integration. Do not remove FastAPI or Socket.IO before that migration is tested.

## Camera lifecycle and performance boundary

PlatformRuntime's CameraManager creates the RTSP Acquisition adapters once.
PipelineManager receives those same instances and passes them into the existing
inference pipeline. There is no second set of RTSP readers.

The CCTV project opts out of connecting cameras during process startup. When
AI starts, PipelineManager asks CameraManager to start the shared adapters;
HealthMonitor then checks connection state and delegates recovery to the
adapter's reconnect method. The Acquisition reader keeps its existing
per-camera exponential-backoff reconnect loop. On AI stop, the pipeline cleans
up the adapters and the manager marks the camera capability inactive.

The inference scheduling path (latest-frame selection, round-robin batching,
GPU executor, result handling) is intentionally unchanged. Jetson tests are
still required to validate connection/reconnect behavior and confirm no FPS
regression.
