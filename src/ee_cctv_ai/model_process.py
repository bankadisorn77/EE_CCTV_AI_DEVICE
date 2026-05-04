import numpy as np
import cv2
from tritonclient.grpc import aio as grpc
import tritonclient.utils.shared_memory as shm
import asyncio
from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime

MODEL_NAME = 'model' 
cpu_executor = ThreadPoolExecutor(max_workers=4)
shm_regions = {}

async def predict_model(img, triton_client, config_cam, size=640, conf=0.5):
    numpy_image, croped_shape = await preprocess(img, config_cam, size)
    
    cam_name = config_cam['name'] if config_cam and 'name' in config_cam else "default_cam"
    
    results = await prediction(numpy_image, triton_client, cam_name)
    
    if results is None:
        return img, {"detection": [], "machineCount": config_cam.get("machineCount", 0) if config_cam else 0}
        
    loop = asyncio.get_event_loop()
    img, payload = await loop.run_in_executor(
        cpu_executor, 
        process_image, 
        results, img, croped_shape, config_cam, size, conf
    )   
    return img, payload


def process_image(results, img, croped_shape, config_cam, size=640, conf=0.5):
    col1, row1 = 0, 0
    count = 0
    if config_cam is not None and config_cam.get("roi") is not None:
        roi = config_cam["roi"]
        row1, col1 = roi["row1"] if roi["row1"] is not None else 0, roi["col1"] if roi["col1"] is not None else 0
        count = config_cam.get("machineCount")

    valid_detections = results[results[:, 4] >= conf]
    detection_class = []
    for detection in valid_detections:
        x1, y1, x2, y2 = detection[0:4]
        conf_score = float(detection[4])
        cls_id = int(detection[5])
        h_orig, w_orig = img.shape[:2]
            
        if croped_shape is not None:
            h_crop, w_crop = croped_shape
            x1 = int(x1 * w_crop / size) + col1
            y1 = int(y1 * h_crop / size) + row1
            x2 = int(x2 * w_crop / size) + col1
            y2 = int(y2 * h_crop / size) + row1
        else:
            x1 = int(x1 * w_orig / size)
            y1 = int(y1 * h_orig / size)
            x2 = int(x2 * w_orig / size)
            y2 = int(y2 * h_orig / size)

        if cls_id == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        detection_class.append({
            "class": cls_id,
            "conf": round(conf_score, 2),
        })
       
    # cv2.imwrite(f"debug_box_{config_cam['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg", img)  
    result = {
        "detection": detection_class,
        "machineCount": count if count is not None else 0,
    }
    return img, result

def cv2_preprocess(img, config_cam, size):
    croped_shape = None
    if config_cam is not None and config_cam["roi"] is not None:
        try:
            roi = config_cam["roi"]
            img = img[roi["row1"]:roi["row2"], roi["col1"]:roi["col2"]].copy()
            croped_shape = img.shape[:2]
        except Exception as e:
            print(f"Error in cropping image: {e}")
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    input_tensor = np.array(img, dtype=np.float32, order='C')
    
    return input_tensor, croped_shape

async def preprocess(image, config_cam, size):
    loop = asyncio.get_event_loop()
    img, croped_shape = await loop.run_in_executor(
        cpu_executor, 
        cv2_preprocess, 
        image, config_cam, size
    )
    return img, croped_shape

async def prediction(numpy_img, triton_client, cam_name):
    input_byte_size = numpy_img.nbytes
    
    task_id = id(asyncio.current_task())
    shm_name = f"shm_{task_id}"
    
    if shm_name not in shm_regions or shm_regions[shm_name]["size"] != input_byte_size:
        try:
            await triton_client.unregister_system_shared_memory(shm_name)
        except Exception:
            pass
        
        import os
        shm_file_path = f"/dev/shm/{shm_name}"
        if os.path.exists(shm_file_path):
            try:
                os.remove(shm_file_path)
            except Exception:
                pass

        shm_handle = shm.create_shared_memory_region(shm_name, "/" + shm_name, input_byte_size)
        await triton_client.register_system_shared_memory(shm_name, "/" + shm_name, input_byte_size)
        shm_regions[shm_name] = {"handle": shm_handle, "name": shm_name, "size": input_byte_size}

    shm.set_shared_memory_region(shm_regions[shm_name]["handle"], [numpy_img])

    inputs = [grpc.InferInput("images", numpy_img.shape, "FP32")]
    inputs[0].set_shared_memory(shm_name, input_byte_size)
    outputs = [grpc.InferRequestedOutput("output0")]

    try:
        results = await triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        return results.as_numpy("output0")[0]
    except Exception as e:
        print(f"Triton Inference Error on {cam_name}: {e}")
        return None
