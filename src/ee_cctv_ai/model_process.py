from matplotlib import image
import numpy as np
import cv2
from tritonclient.grpc import aio as grpc

MODEL_NAME = 'model' 


 
async def predict_model(img, triton_client,config_cam, size=640, conf=0.5):
    numpy_image,croped_shape =  preprocess(img,config_cam)
    results = await prediction(numpy_image,triton_client)
    if results is None:
        return img, []
    img,detection_class =   process_image(results, img,croped_shape,config_cam ,size, conf) 
    return img,detection_class

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
    
    result = {
        "detection": detection_class,
        "machineCount": count if count is not None else 0,
    }


    return img, result




async def prediction(numpy_img,triton_client):
    triton = triton_client
    inputs = [
        grpc.InferInput("images", numpy_img.shape,"FP16")
    ]
        # Set the data for the input tensor
    inputs[0].set_data_from_numpy(numpy_img)

    outputs=[
        grpc.InferRequestedOutput("output0")
    ]

    results =  await triton.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )
    result = results.as_numpy("output0")[0]
    # print(f"Shape: {result.shape}")
    # print(f"Max conf: {result[:, 4].max():.4f}")
    # print(f"Min conf: {result[:, 4].min():.4f}")
    # print(f"Sample row: {result[0]}")
    return result


def preprocess(image,config_cam):
    img = image
    croped_shape = None
    if config_cam is not None:
        if config_cam["roi"] is not None:
            img = crop_function(img, config_cam)
            croped_shape = img.shape[:2]
            
    else:
        img = image
    # img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    img = img.astype(np.float16)
    return img,croped_shape

def crop_function(img,cam_data):
    try:
        roi = cam_data["roi"]
        cropped_img = img[roi["row1"]:roi["row2"], roi["col1"]:roi["col2"]]
        return cropped_img
    except Exception as e:
        print(f"Error in cropping image: {e}")
        return img









 # if len(valid_detections) == 0:
    #     return img, detection_class

    # if 1 in valid_detections[:, 5]:
    #     for detection in valid_detections:
    #         if int(detection[5]) != 1:
    #             continue
    #         x1, y1, x2, y2 = detection[0:4]
    #         conf_score = float(detection[4])
    #         cls_id = int(detection[5])
    #         h_orig, w_orig = img.shape[:2]
    #         print(f"Detection: Class={cls_id}, Confidence={conf_score:.2f}, Box=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    #         if croped_shape is not None:
    #             h_crop, w_crop = croped_shape
    #             x1 = int(x1 * w_crop / size) + col1
    #             y1 = int(y1 * h_crop / size) + row1
    #             x2 = int(x2 * w_crop / size) + col1
    #             y2 = int(y2 * h_crop / size) + row1
    #         else:
    #             x1 = int(x1 * w_orig / size)
    #             y1 = int(y1 * h_orig / size)
    #             x2 = int(x2 * w_orig / size)
    #             y2 = int(y2 * h_orig / size)

    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         detection_class.append({
    #             "class": cls_id,
    #             "conf": round(conf_score, 2)
    #         })