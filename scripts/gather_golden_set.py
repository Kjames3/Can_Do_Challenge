import asyncio
import os
import json
import time
import math
import base64
import numpy as np
import cv2
import websockets

# ==============================================================================
# GOLDEN TEST SET GATHERER (CS 228) - NATIVE WEBSOCKET CLIENT
# ==============================================================================
# The Paradigm Shift (research_innovator):
# Kinematic-Aware Active Sampling. Instead of uniformly saving frames, we 
# calculate the Variance of the Laplacian in real-time, coupled with odometry 
# movement checks, to selectively save frames that truly exhibit motion blur.
#
# The Academic Theory Bridge (academic_theory_bridge):
# Blur I_{blur}(x, y) is an integral of the sharp image over T_{exp} relative 
# to the robot's velocity vector. Therefore, ground-truth motion parameters 
# (v_linear, v_angular) must be recorded synchronously with each captured frame.
#
# Integration: Bypasses Viam SDK and couples directly with server_native.py
# via WebSockets to harvest frames and telemetry without hardware conflicts.
# Ensure `pip install opencv-python-headless websockets` is in the environment.
# ==============================================================================

# ----------------- CONFIGURATION -----------------
import sys
# Default to the robot's IP, but allow passing a custom one
ROBOT_IP = "192.168.137.91"
if len(sys.argv) > 1:
    ROBOT_IP = sys.argv[1]
    
SERVER_URI = f"ws://{ROBOT_IP}:8081"

# Dataset configuration
TARGET_IMAGES = 600
OUTPUT_DIR = "images"  # Save directly to the images folder in project directory

# Active Sampling Thresholds
# Note: Native Server sends encoder counts/sec for velocity. Adjust if units differ.
MIN_LINEAR_VELOCITY = 0.5     # minimum threshold to be considered "moving" forward
MIN_ANGULAR_VELOCITY = 0.1    # minimum rad/s to be considered "turning"
MAX_LAPLACIAN_VARIANCE = 200.0  # (Lower = more blurry)

def calculate_laplacian_variance(image: np.ndarray) -> float:
    """ Computes the variance of the Laplacian as a focus/blur metric. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

async def gather_golden_set():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "metadata"), exist_ok=True)
    
    collected_count = 0
    print(f"Connecting to Native Server at {SERVER_URI}...")

    try:
        async with websockets.connect(SERVER_URI, ping_interval=None) as websocket:
            print(f"Connected! Waiting for 'Golden Set: Start' to be pressed in GUI...")
            
            was_active = False
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "readout":
                    is_active = data.get("golden_collection_active", False)
                    
                    if is_active and not was_active:
                        print("\n[+] GUI Triggered START! Collecting frames...")
                        was_active = True
                        collected_count = 0
                        
                    elif not is_active and was_active:
                        print(f"\n[+] GUI Triggered STOP! Finished. Saved {collected_count} images to '{OUTPUT_DIR}'")
                        was_active = False
                        # We keep running so the user can start another session if needed.
                        print(f"Waiting for 'Golden Set: Start' to be pressed in GUI...")
                        
                    if is_active and data.get("image"):
                        # 1. Extract kinematic state
                        motor_vel = data.get("motor_velocity", {})
                        vel_l = motor_vel.get("left", 0.0)
                        vel_r = motor_vel.get("right", 0.0)
                        linear_velocity_mag = abs((vel_l + vel_r) / 2.0)
    
                        imu_data = data.get("imu", {})
                        # IMU yaw rate is typically in deg/s or rad/s
                        angular_velocity_mag = abs(imu_data.get("yaw_rate", 0.0))
                        
                        # 2. Check movement threshold
                        is_moving = linear_velocity_mag > MIN_LINEAR_VELOCITY or angular_velocity_mag > MIN_ANGULAR_VELOCITY
                        
                        if is_moving:
                            # 3. Decode the base64 image sent by native server
                            img_bytes = base64.b64decode(data["image"])
                            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                            
                            if img_bgr is not None:
                                # 4. Check blur severity
                                blur_metric = calculate_laplacian_variance(img_bgr)
                                
                                if blur_metric < MAX_LAPLACIAN_VARIANCE:
                                    timestamp = time.time()
                                    base_filename = f"blur_frame_{int(timestamp * 1000)}"
                                    img_path = os.path.join(OUTPUT_DIR, "images", f"{base_filename}.jpg")
                                    meta_path = os.path.join(OUTPUT_DIR, "metadata", f"{base_filename}.json")
    
                                    # Save the image
                                    cv2.imwrite(img_path, img_bgr)
    
                                    # Save the metadata
                                    metadata_record = {
                                        "timestamp": timestamp,
                                        "linear_velocity": linear_velocity_mag,
                                        "angular_velocity": angular_velocity_mag,
                                        "laplacian_variance": blur_metric,
                                        "robot_pose": data.get("robot_pose", {})
                                    }
                                    with open(meta_path, 'w') as f:
                                        json.dump(metadata_record, f, indent=4)
    
                                    collected_count += 1
                                    print(f"[{collected_count}] Saved blurred frame to {OUTPUT_DIR}/images/ (Blur: {blur_metric:.1f})")
                                    
                                    # If we hit the limit, tell the server to stop!
                                    if collected_count >= TARGET_IMAGES:
                                        print(f"Reached target of {TARGET_IMAGES} images. Stopping...")
                                        await websocket.send(json.dumps({"type": "stop_golden_collection"}))
                                        was_active = False

    except ConnectionRefusedError:
        print(f"Error: Could not connect to {SERVER_URI}. Is server_native.py running?")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"Data collection process ended. Gathered {collected_count}/{TARGET_IMAGES} blurry frames.")

if __name__ == '__main__':
    asyncio.run(gather_golden_set())
