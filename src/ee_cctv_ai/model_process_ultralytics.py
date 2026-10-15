import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _env_bool(name, default):
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


MODEL_PATH = Path(
    os.getenv(
        "EE_CCTV_MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "best_V6.engine"),
    )
)
INPUT_SIZE = int(os.getenv("EE_CCTV_IMGSZ", "640"))
MAX_BATCH = max(1, int(os.getenv("EE_CCTV_MAX_BATCH", "4")))
USE_GPU_PREPROCESS = _env_bool("EE_CCTV_GPU_PREPROCESS", True)
PROFILE = _env_bool("EE_CCTV_PROFILE", False)

PAD_VALUE = 114  # same grey value Ultralytics uses for letterbox padding

# Every CUDA call (load, warm-up, predict) is submitted to this single thread.
gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-infer")
# Light CPU work that must not block the event loop (JPEG encode for alarms).
cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-work")

_model = None
_load_lock = threading.Lock()
_state = {
    "max_batch": MAX_BATCH,   # how many frames the pipeline may put in one batch
    "pad_to": 0,              # >0: engine has a fixed batch size, pad up to it
    "gpu_preprocess": USE_GPU_PREPROCESS,
    "mode": "unloaded",
}

_EMPTY = np.zeros((0, 6), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pure helpers (no torch / ultralytics needed -> unit-testable anywhere)
# --------------------------------------------------------------------------- #
def max_batch():
    return _state["max_batch"]


def letterbox_params(h, w, size):
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (size - new_w) / 2.0, (size - new_h) / 2.0
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    return r, new_w, new_h, left, top


def map_boxes(dets, lb, crop_hw, offset, frame_hw):
    d = np.array(dets, dtype=np.float32, copy=True)
    if d.size == 0:
        return _EMPTY
    if lb is not None:
        r, left, top = lb
        d[:, [0, 2]] = (d[:, [0, 2]] - left) / r
        d[:, [1, 3]] = (d[:, [1, 3]] - top) / r
    ch, cw = crop_hw
    d[:, [0, 2]] = np.clip(d[:, [0, 2]], 0, cw)
    d[:, [1, 3]] = np.clip(d[:, [1, 3]], 0, ch)
    col1, row1 = offset
    d[:, [0, 2]] += col1
    d[:, [1, 3]] += row1
    fh, fw = frame_hw
    d[:, [0, 2]] = np.clip(d[:, [0, 2]], 0, fw)
    d[:, [1, 3]] = np.clip(d[:, [1, 3]], 0, fh)
    return d

def cv2_preprocess_fast(img, config_cam):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    h_orig, w_orig = img.shape[:2]
    row1, col1 = 0, 0
    cropped_img = img

    roi = config_cam.get("roi") if config_cam is not None else None
    if roi and isinstance(roi, dict):
        try:
            r1 = max(0, int(roi.get("row1") or 0))
            r2 = min(h_orig, int(roi.get("row2") or h_orig))
            c1 = max(0, int(roi.get("col1") or 0))
            c2 = min(w_orig, int(roi.get("col2") or w_orig))
            if (r2 > r1) and (c2 > c1):
                cropped_img = img[r1:r2, c1:c2]
                row1, col1 = r1, c1
        except Exception as e:
            print(f"Error in fast crop: {e}")

    return cropped_img, (col1, row1)


def build_payload(dets, config_cam):
    count = config_cam.get("machineCount", 0) if config_cam else 0
    detection = []
    for x1, y1, x2, y2, score, cls_id in dets:
        detection.append({
            "class": int(cls_id),
            "conf": round(float(score), 2),
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })
    return {"detection": detection, "machineCount": count if count is not None else 0}


def draw_detections(img, detections):
    out = img.copy()
    for det in detections or []:
        x1, y1, x2, y2 = det["box"]
        color = (0, 0, 255) if det["class"] == 1 else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out



def _device():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _gpu_letterbox_batch(crops, size, batch_n):
    import torch
    import torch.nn.functional as F

    dev = _device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    out = torch.full((batch_n, 3, size, size), PAD_VALUE / 255.0, dtype=dtype, device=dev)
    lbs = []
    for i, crop in enumerate(crops):
        h, w = crop.shape[:2]
        r, nw, nh, left, top = letterbox_params(h, w, size)
        t = torch.from_numpy(np.ascontiguousarray(crop)).to(dev)
        t = t.permute(2, 0, 1).flip(0).unsqueeze(0).to(dtype).div_(255.0)  # BGR->RGB
        if (nh, nw) != (h, w):
            # no antialias + align_corners=False == cv2.INTER_LINEAR used by Ultralytics
            t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        out[i, :, top:top + nh, left:left + nw] = t[0]
        lbs.append((r, left, top))
    return out, lbs

def _try_batch(model, n, gpu, size):
    try:
        crops = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(n)]
        if gpu:
            inp, _ = _gpu_letterbox_batch(crops, size, n)
        else:
            inp = crops
        res = model.predict(inp, imgsz=size, conf=0.25, verbose=False)
        return len(res) == n
    except Exception as exc:
        print(f"[model] warm-up batch={n} gpu_preprocess={gpu} failed: {exc!r}")
        return False


def _detect_mode(model, gpu, size):
    n = MAX_BATCH
    ok1 = _try_batch(model, 1, gpu, size)
    if n == 1:
        return ("single", 1) if ok1 else None
    okn = _try_batch(model, n, gpu, size)
    if ok1 and okn:
        mids = sorted({2, n - 1} - {1, n})
        if all(_try_batch(model, m, gpu, size) for m in mids):
            return ("dynamic", n)
        return ("pad", n)          # only some sizes work -> always run full batch
    if okn:
        return ("pad", n)
    if ok1:
        return ("single", 1)
    return None


def load_model():
    global _model
    with _load_lock:
        if _model is not None:
            return _model
        from ultralytics import YOLO

        print(f"[model] loading {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH), task="detect")
        size = INPUT_SIZE

        candidates = [True, False] if USE_GPU_PREPROCESS else [False]
        for gpu in candidates:
            found = _detect_mode(model, gpu, size)
            if found is not None:
                mode, n = found
                _state["gpu_preprocess"] = gpu
                _state["mode"] = mode
                _state["max_batch"] = n if mode != "single" else 1
                _state["pad_to"] = n if mode == "pad" else 0
                print(
                    f"[model] ready: mode={mode} max_batch={_state['max_batch']} "
                    f"pad_to={_state['pad_to']} gpu_preprocess={gpu}"
                )
                _model = model
                return _model
            print(f"[model] gpu_preprocess={gpu} unusable, trying next option")
        raise RuntimeError("Engine failed warm-up with every preprocessing option")


async def ensure_loaded():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_executor, load_model)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_batch_sync(frames, cfgs, size=INPUT_SIZE, conf=0.5):
    model = load_model()
    t0 = time.perf_counter()

    crops, offsets, frame_hws = [], [], []
    for frame, cfg in zip(frames, cfgs):
        crop, off = cv2_preprocess_fast(frame, cfg)
        crops.append(crop)
        offsets.append(off)
        frame_hws.append(frame.shape[:2])
    n = len(crops)
    batch_n = max(n, _state["pad_to"])

    if _state["gpu_preprocess"]:
        inp, lbs = _gpu_letterbox_batch(crops, size, batch_n)
    else:
        inp = list(crops)
        if batch_n > n:
            inp += [np.zeros((size, size, 3), dtype=np.uint8)] * (batch_n - n)
        lbs = [None] * n
    if PROFILE and _state["gpu_preprocess"]:
        import torch
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    results = model.predict(inp, imgsz=size, conf=conf, verbose=False)
    t2 = time.perf_counter()

    dets = []
    for i in range(n):
        boxes = results[i].boxes
        if boxes is None or len(boxes) == 0:
            dets.append(_EMPTY)
            continue
        d = boxes.data.cpu().numpy()
        d = d[d[:, 4] >= conf]
        dets.append(map_boxes(d, lbs[i], crops[i].shape[:2], offsets[i], frame_hws[i]))
    t3 = time.perf_counter()

    timing = {
        "n": n,
        "pre_ms": (t1 - t0) * 1000.0,
        "infer_ms": (t2 - t1) * 1000.0,
        "post_ms": (t3 - t2) * 1000.0,
        "frame_shape": tuple(frames[0].shape[:2]) if frames else None,
    }
    if PROFILE and results:
        timing["speed"] = getattr(results[0], "speed", None)
    return dets, timing
