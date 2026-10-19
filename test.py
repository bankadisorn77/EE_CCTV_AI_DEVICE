import time
import threading
import traceback
import numpy as np
import torch
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"  # หรือ .engine
NUM_INSTANCES = 4  
IMG_SIZE = 640
TARGET_FPS = 25.0  

summary_stats = {}
stats_lock = threading.Lock()

def worker_stress_test(instance_id, model, stop_event, target_fps=25.0):
    print(f"[Instance-{instance_id}] Worker started (Target: {target_fps} FPS)")
    
    # สร้างภาพ Dummy uint8 (640, 640, 3)
    dummy_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    frame_interval = 1.0 / target_fps
    frame_count = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            loop_start = time.time()
            try:
                # บังคับรันบน device=0 และปิด verbose
                _ = model.predict(dummy_frame, imgsz=IMG_SIZE, conf=0.5, device=0, verbose=False)
                frame_count += 1
                
                elapsed_work = time.time() - loop_start
                sleep_time = frame_interval - elapsed_work
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_count % 50 == 0:
                    cur_elapsed = time.time() - start_time
                    cur_fps = frame_count / cur_elapsed
                    print(f"[Instance-{instance_id}] Frames: {frame_count} | FPS: {cur_fps:.2f}")

            except Exception as e:
                print(f"[Instance-{instance_id}] Error in prediction loop: {e}")
                traceback.print_exc()
                break
    finally:
        total_duration = time.time() - start_time
        avg_fps = frame_count / total_duration if total_duration > 0 else 0.0
        with stats_lock:
            summary_stats[instance_id] = {
                "total_frames": frame_count,
                "duration": total_duration,
                "avg_fps": avg_fps
            }
        print(f"[Instance-{instance_id}] Worker finished. Processed {frame_count} frames.")

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Stress Test: {NUM_INSTANCES} Instances | Target: {TARGET_FPS} FPS/Instance | Device: {device} ===")
    
    models = []
    dummy_init = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    for i in range(NUM_INSTANCES):
        print(f"Loading & Warming up Model Instance {i+1}/{NUM_INSTANCES} ({MODEL_NAME})...")
        m = YOLO(MODEL_NAME, task="detect")
        # Warm-up 1 ครั้ง เพื่อบังคับโหลด Weights และ CUDA Context ขึ้น VRAM
        m.predict(dummy_init, imgsz=IMG_SIZE, device=device, verbose=False)
        models.append(m)
        
    if torch.cuda.is_available():
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"\nGPU VRAM Allocated: {allocated_mem:.2f} MB\n")

    stop_event = threading.Event()
    threads = []
    
    test_start_time = time.time()
    for i in range(NUM_INSTANCES):
        t = threading.Thread(
            target=worker_stress_test, 
            args=(i + 1, models[i], stop_event, TARGET_FPS),
            daemon=False  # เปลี่ยนเป็น False เพื่อให้เก็บ stats จนเสร็จ
        )
        threads.append(t)
        t.start()

    print("All workers running. Press CTRL+C to stop and view final summary.\n")
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping workers and calculating final statistics...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

    total_test_time = time.time() - test_start_time
    total_all_frames = sum(stat["total_frames"] for stat in summary_stats.values())
    overall_throughput_fps = total_all_frames / total_test_time if total_test_time > 0 else 0.0

    print("\n" + "=" * 50)
    print("           STRESS TEST PERFORMANCE SUMMARY        ")
    print("=" * 50)
    print(f"Total Test Duration : {total_test_time:.2f} seconds")
    print(f"Target Setting      : {TARGET_FPS:.1f} FPS per Instance")
    print("-" * 50)
    
    for inst_id, stat in sorted(summary_stats.items()):
        print(f"Instance {inst_id}:")
        print(f"  - Processed Frames : {stat['total_frames']} frames")
        print(f"  - Average FPS      : {stat['avg_fps']:.2f} FPS")

    print("-" * 50)
    print(f"Total Frames Processed (All Instances) : {total_all_frames} frames")
    print(f"Combined System Throughput             : {overall_throughput_fps:.2f} FPS")
    print("=" * 50)

if __name__ == "__main__":
    main()