import cv2
import zmq
import numpy as np
import json
import time

def start_recorder():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://127.0.0.1:5555")
    sock.subscribe("") 
    print("Recorder started, waiting for frames...")
    recorders = {} 

    while True:
        topic, payload, metadata = sock.recv_multipart()
        cam_id = topic.decode('utf-8')
        meta = json.loads(metadata.decode())

        if meta["has_object"]:
            if cam_id not in recorders:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = f"Video/record_{cam_id}_{int(time.time())}.mp4"
                recorders[cam_id] = {
                    "writer": cv2.VideoWriter(filename, fourcc, 20.0, (1920, 1080)),#(640, 640) size Video
                    "last_seen": time.time()
                }
                print(f"Start recording: {filename}")

   
            nparr = np.frombuffer(payload, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            recorders[cam_id]["writer"].write(frame)
            recorders[cam_id]["last_seen"] = time.time()

        
        for cid in list(recorders.keys()):
            if time.time() - recorders[cid]["last_seen"] > 5:
                recorders[cid]["writer"].release()
                del recorders[cid]
                print(f"Stopped recording: {cid}")

if __name__ == "__main__":
    start_recorder()