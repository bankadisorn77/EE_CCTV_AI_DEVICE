import asyncio
import contextlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _env_bool(name, default):
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


#DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "best_V6.engine")
DEFAULT_MODEL_PATHS = (
    f"{Path(__file__).resolve().parents[2] / 'models' / 'best_V6.engine'},"
    f"{Path(__file__).resolve().parents[2] / 'models' / 'yolo26n.engine'}"
)

# รองรับ 1-4 โมเดลผ่าน EE_CCTV_MODEL_PATHS (คั่นด้วยลูกน้ำ เช่น "models/m1.engine,models/m2.engine")

_raw_paths = os.getenv("EE_CCTV_MODEL_PATHS") or os.getenv("EE_CCTV_MODEL_PATH", DEFAULT_MODEL_PATHS)
MODEL_PATHS = [Path(p.strip()) for p in _raw_paths.split(",") if p.strip()]
MODEL_PATH = MODEL_PATHS[0]  

INPUT_SIZE = int(os.getenv("EE_CCTV_IMGSZ", "640"))
MAX_BATCH = max(1, int(os.getenv("EE_CCTV_MAX_BATCH", "4")))
USE_GPU_PREPROCESS = _env_bool("EE_CCTV_GPU_PREPROCESS", True)
USE_FAST_POST = _env_bool("EE_CCTV_FAST_POST", True)

USE_ROI_CROP = _env_bool("EE_CCTV_USE_ROI", False)
NMS_IOU = float(os.getenv("EE_CCTV_NMS_IOU", "0.7"))
MAX_DET = 300
PROFILE = _env_bool("EE_CCTV_PROFILE", False)

PAD_VALUE = 114  #


gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-infer")

prep_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-prep")

cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-work")

_models = None
_load_lock = threading.Lock()
_state = {
    "max_batch": MAX_BATCH,  
    "pad_to": 0,             
    "gpu_preprocess": USE_GPU_PREPROCESS,
    "fast_post": USE_FAST_POST,
    "mode": "unloaded",
}
_stream = None
_stream_lock = threading.Lock()

_EMPTY = np.zeros((0, 6), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pure helpers (no torch / ultralytics needed -> unit-testable anywhere)
# --------------------------------------------------------------------------- #
def max_batch():
    return _state["max_batch"]


def model_names():
    if _models is not None:
        return [m["name"] for m in _models]
    return [f"model_{i + 1}" for i in range(len(MODEL_PATHS))]


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


def _iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.maximum(a + b - inter, 1e-9)


def detections_match(fast, ref, conf, iou_thr=0.8, margin=0.03):
    def covered(a, b):
        for box in a:
            if box[4] < conf + margin:
                continue
            if len(b) == 0:
                return False
            same = b[b[:, 5] == box[5]]
            if len(same) == 0 or _iou_one_to_many(box, same).max() < iou_thr:
                return False
        return True

    return covered(ref, fast) and covered(fast, ref)


def cv2_preprocess_fast(img, config_cam):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    if not USE_ROI_CROP:
        return img, (0, 0)

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


# --------------------------------------------------------------------------- #
# GPU letterbox
# --------------------------------------------------------------------------- #
def _device():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _prep_stream():
    global _stream
    import torch
    if _stream is None and torch.cuda.is_available():
        with _stream_lock:
            if _stream is None:
                _stream = torch.cuda.Stream()
    return _stream


def _gpu_letterbox_batch(crops, size, batch_n):
    import torch
    import torch.nn.functional as F

    dev = _device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    stream = _prep_stream() if dev.type == "cuda" else None
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    lbs = []
    with ctx:
        out = torch.full((batch_n, 3, size, size), PAD_VALUE / 255.0, dtype=dtype, device=dev)
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            r, nw, nh, left, top = letterbox_params(h, w, size)
            t = torch.from_numpy(np.ascontiguousarray(crop)).to(dev)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dtype)         # HWC uint8 -> 1x3xHxW float
            if (nh, nw) != (h, w):
                t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
            t = t[:, [2, 1, 0]].div_(255.0)                       # BGR->RGB on the SMALL image
            out[i, :, top:top + nh, left:left + nw] = t[0]
            lbs.append((r, left, top))
    if stream is not None:
        stream.synchronize()
    return out, lbs


# --------------------------------------------------------------------------- #
# Model loading / probing (supports 1 to 4 models)
# --------------------------------------------------------------------------- #
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
        return ("pad", n)          
    if okn:
        return ("pad", n)
    if ok1:
        return ("single", 1)
    return None


def _probe_fast(model, size, gpu_enabled, pad_to):
    if not (USE_FAST_POST and gpu_enabled):
        return False
    try:
        n = max(1, pad_to)
        inp, _ = _gpu_letterbox_batch([np.zeros((720, 1280, 3), np.uint8)], size, n)
        out = _raw_forward(model, inp)
        _fast_postprocess(out[:1], 0.25)
        return True
    except Exception as exc:
        print(f"[model] fast postprocess unavailable, using Ultralytics path: {exc!r}")
        return False


def load_model():
    global _models
    with _load_lock:
        if _models is not None:
            return _models
        from ultralytics import YOLO

        size = INPUT_SIZE
        loaded_list = []

        for idx, path in enumerate(MODEL_PATHS):
            m_name = f"model_{idx + 1}"
            print(f"[{m_name}] loading {path}")
            model = YOLO(str(path), task="detect")

            candidates = [True, False] if USE_GPU_PREPROCESS else [False]
            m_info = None
            for gpu in candidates:
                found = _detect_mode(model, gpu, size)
                if found is not None:
                    mode, n = found
                    pad_to = n if mode == "pad" else 0
                    fast_ok = _probe_fast(model, size, gpu, pad_to)
                    m_info = {
                        "name": m_name,
                        "path": str(path),
                        "model": model,
                        "mode": mode,
                        "max_batch": n if mode != "single" else 1,
                        "pad_to": pad_to,
                        "gpu_preprocess": gpu,
                        "fast_post": fast_ok,
                        "batches": 0,
                        "verified_with_dets": 0,
                    }
                    print(
                        f"[{m_name}] ready: mode={mode} max_batch={m_info['max_batch']} "
                        f"pad_to={pad_to} gpu_preprocess={gpu} fast_post={fast_ok}"
                    )
                    break
                print(f"[{m_name}] gpu_preprocess={gpu} unusable, trying next option")

            if m_info is None:
                raise RuntimeError(f"Engine {path} failed warm-up with every preprocessing option")
            loaded_list.append(m_info)

        all_gpu = all(m["gpu_preprocess"] for m in loaded_list)
        if not all_gpu:
            for m in loaded_list:
                m["gpu_preprocess"] = False
                m["fast_post"] = False

        _state["gpu_preprocess"] = all_gpu
        _state["max_batch"] = min(m["max_batch"] for m in loaded_list)
        _state["pad_to"] = max(m["pad_to"] for m in loaded_list)
        _state["fast_post"] = any(m["fast_post"] for m in loaded_list)
        _state["mode"] = loaded_list[0]["mode"] if len(loaded_list) == 1 else "multi"

        print(
            f"[pipeline] {len(loaded_list)} model(s) configured: "
            f"shared max_batch={_state['max_batch']} shared pad_to={_state['pad_to']} "
            f"gpu_preprocess={_state['gpu_preprocess']} use_roi_crop={USE_ROI_CROP}"
        )
        _models = loaded_list
        return _models


async def ensure_loaded():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_executor, load_model)


# --------------------------------------------------------------------------- #
# Fast path: direct engine call + batched GPU NMS
# --------------------------------------------------------------------------- #
def _raw_forward(model, inp):
    import torch
    backend = model.predictor.model          # exists after the first predict() (warm-up)
    inp = inp.half() if getattr(backend, "fp16", False) else inp.float()
    with torch.no_grad():
        out = backend(inp)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.ndim != 3 or out.shape[1] < 5 or out.shape[2] <= out.shape[1]:
        raise RuntimeError(f"unexpected engine output shape {tuple(out.shape)} (need B x 4+nc x anchors)")
    return out


def _fast_postprocess(out, conf, iou=NMS_IOU):
    import torch
    import torchvision

    out = out.float()
    bsz, ch, _ = out.shape
    nc = ch - 4
    scores, cls = out[:, 4:, :].max(dim=1)                  # (B, A)
    idx = (scores > conf).nonzero()                         # (K, 2)  -> first GPU sync
    if idx.shape[0] == 0:
        return [_EMPTY] * bsz
    b, a = idx[:, 0], idx[:, 1]
    box = out[b, :4, a]                                     # (K, 4) cx,cy,w,h
    xy, wh = box[:, :2], box[:, 2:]
    xyxy = torch.cat((xy - wh / 2, xy + wh / 2), 1)
    sc, cl = scores[b, a], cls[b, a]
    offset = (b * nc + cl).to(xyxy.dtype).unsqueeze(1) * 7680.0
    keep = torchvision.ops.nms(xyxy + offset, sc, iou)
    res = torch.cat(
        (xyxy[keep], sc[keep].unsqueeze(1), cl[keep].unsqueeze(1).to(xyxy.dtype),
         b[keep].unsqueeze(1).to(xyxy.dtype)), 1
    ).cpu().numpy()                                         # second (final) sync
    img_idx = res[:, 6].astype(np.int64)
    dets = []
    for i in range(bsz):
        r = res[img_idx == i][:MAX_DET, :6]
        dets.append(r.astype(np.float32) if len(r) else _EMPTY)
    return dets


# --------------------------------------------------------------------------- #
# Inference (two stages)
# --------------------------------------------------------------------------- #
def prepare_batch_sync(frames, cfgs, size=INPUT_SIZE):
    load_model()  # no-op once loaded; makes sure _state is final
    t0 = time.perf_counter()
    crops, offsets, frame_hws = [], [], []
    for frame, cfg in zip(frames, cfgs):
        crop, off = cv2_preprocess_fast(frame, cfg)
        crops.append(crop)
        offsets.append(off)
        frame_hws.append(frame.shape[:2])
    n = len(crops)
    batch_n = max(n, _state["pad_to"])

    gpu = _state["gpu_preprocess"]
    if gpu:
        inp, lbs = _gpu_letterbox_batch(crops, size, batch_n)
    else:
        inp = list(crops)
        if batch_n > n:
            inp += [np.zeros((size, size, 3), dtype=np.uint8)] * (batch_n - n)
        lbs = [None] * n
    return {
        "inp": inp, "lbs": lbs, "gpu": gpu, "n": n, "size": size,
        "crop_hws": [c.shape[:2] for c in crops], "offsets": offsets, "frame_hws": frame_hws,
        "timing": {"pre_ms": (time.perf_counter() - t0) * 1000.0,
                   "frame_shape": tuple(frame_hws[0]) if frame_hws else None},
    }


def _map_all(raw_list, prepared):
    return [
        map_boxes(raw_list[i], prepared["lbs"][i], prepared["crop_hws"][i],
                  prepared["offsets"][i], prepared["frame_hws"][i])
        for i in range(prepared["n"])
    ]


def _slice_input_for_model(m_info, prepared):
    inp = prepared["inp"]
    n = prepared["n"]
    target_len = max(n, m_info["pad_to"])
    if len(inp) > target_len:
        return inp[:target_len]
    return inp


def _ultralytics_infer(m_info, prepared, conf):
    model = m_info["model"]
    inp = _slice_input_for_model(m_info, prepared)
    results = model.predict(inp, imgsz=prepared["size"], conf=conf, iou=NMS_IOU, verbose=False)
    raw = []
    for i in range(prepared["n"]):
        boxes = results[i].boxes
        if boxes is None or len(boxes) == 0:
            raw.append(_EMPTY)
            continue
        d = boxes.data.cpu().numpy()
        raw.append(d[d[:, 4] >= conf])
    return _map_all(raw, prepared), (results[0].speed if PROFILE and results else None)


def _fast_infer(m_info, prepared, conf):
    model = m_info["model"]
    inp = _slice_input_for_model(m_info, prepared)
    import torch
    inp.record_stream(torch.cuda.current_stream())   # tensor was allocated on the prep stream
    out = _raw_forward(model, inp)
    engine_ms = None
    if PROFILE:
        t = time.perf_counter()
        torch.cuda.synchronize()
        engine_ms = (time.perf_counter() - t) * 1000.0
    raw = _fast_postprocess(out[:prepared["n"]], conf)
    return _map_all(raw, prepared), engine_ms


def infer_batch_sync(prepared, conf=0.5):
    models = load_model()
    t0 = time.perf_counter()

    all_models_dets = {}
    paths_used = []
    last_extra = None

    for m_info in models:
        m_name = m_info["name"]
        dets, path, extra = None, "ultralytics", None

        if m_info["fast_post"] and prepared["gpu"]:
            try:
                dets, extra = _fast_infer(m_info, prepared, conf)
                path = "fast"
            except Exception as exc:
                print(f"[{m_name}] fast path failed ({exc!r}); switching to Ultralytics path")
                m_info["fast_post"] = False
                dets = None

        if dets is None:
            dets, extra = _ultralytics_infer(m_info, prepared, conf)
        else:
            m_info["batches"] += 1
            b = m_info["batches"]
            if b <= 20 or b % 100 == 0:           # shadow-check against Ultralytics per model
                try:
                    ref, _ = _ultralytics_infer(m_info, prepared, conf)
                    bad = [i for i in range(prepared["n"]) if not detections_match(dets[i], ref[i], conf)]
                    if bad:
                        print(f"[{m_name}] fast NMS MISMATCH on batch {b} (images {bad}); "
                              f"fast={[len(d) for d in dets]} ref={[len(d) for d in ref]} -> disabling fast path")
                        m_info["fast_post"] = False
                        dets, path = ref, "ultralytics(verify-fallback)"
                    elif any(len(d) for d in ref):
                        m_info["verified_with_dets"] += 1
                    if b == 20 and m_info["fast_post"]:
                        print(f"[{m_name}] fast NMS agreed with Ultralytics on the first 20 batches "
                              f"({m_info['verified_with_dets']} with detections)")
                except Exception as exc:
                    print(f"[{m_name}] verification failed ({exc!r}); disabling fast path")
                    m_info["fast_post"] = False

        all_models_dets[m_name] = dets
        paths_used.append(f"{m_name}:{path}")
        last_extra = extra

    batch_results = []
    for i in range(prepared["n"]):
        cam_dets = {m_info["name"]: all_models_dets[m_info["name"]][i] for m_info in models}
        batch_results.append(cam_dets)

    timing = {
        "n": prepared["n"],
        "infer_ms": (time.perf_counter() - t0) * 1000.0,
        "path": ",".join(paths_used),
    }
    if PROFILE:
        timing["speed"] = last_extra
    return batch_results, timing


def predict_batch_sync(frames, cfgs, size=INPUT_SIZE, conf=0.5):
    prepared = prepare_batch_sync(frames, cfgs, size)
    dets, timing = infer_batch_sync(prepared, conf)
    timing = dict(prepared["timing"], **timing)
    return dets, timing