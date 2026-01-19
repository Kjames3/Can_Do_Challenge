"""
Viam Rover Control Server - YAHBOOM X3 (Jetson Orin)

This WebSocket server specifically controls the Yahboom X3 robot.
Features:
- 4-Wheel Mecanum Drive (Holonomic)
- Serial Communication with ROS Controller Board
- YDLidar 4ROS Support
- Orbbec Astra Pro (RGB + Depth) Support
- YOLOv11 Detection
"""

import asyncio
import time
import json
import logging
import argparse
import base64
import numpy as np
import cv2
import websockets
import sys
import os
from ultralytics import YOLO

# Add root directory to sys.path to allow importing 'robot_state'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import X3 specific drivers
# Note: Since this script is in 'yahboom/', and drivers_x3 is also there, direct import works.
from drivers_x3 import (
    Rosmaster, MecanumDrive, YDLidarDriver, 
    # Reuse Camera drivers
    NativeCamera
)
# Reuse generic RobotState (Requires Odometry updates)
from robot_state import RobotState

# =============================================================================
# CONFIGURATION
# =============================================================================
parser = argparse.ArgumentParser(description='Yahboom X3 Control Server')
parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
args = parser.parse_args()
SIM_MODE = args.sim

# Hardware Ports
SERIAL_PORT = "/dev/ttyUSB0" # Motor Board
LIDAR_PORT = "/dev/ttyUSB1" # YDLidar
CAMERA_INDEX = 0 # Orbbec RGB usually shows as standard video device

# Detection Config
# Construct absolute path to model in ../models/
YOLO_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'yolo11n_cans.pt')
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 640

# =============================================================================
# GLOBAL STATE
# =============================================================================
ros_board = None
drive = None
lidar = None
camera = None
robot_state = None

detection_enabled = False
is_auto_driving = False # Placeholder for future autonomous logic
last_detections = []
model = None

connected_clients = set()

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_hardware():
    global ros_board, drive, lidar, camera, robot_state, model
    
    logger.info("="*50)
    logger.info("Initializing Yahboom X3 Hardware")
    logger.info("="*50)

    # 1. Motor Controller (Serial)
    logger.info("Connecting to Rosmaster...")
    ros_board = Rosmaster(port=SERIAL_PORT, sim_mode=SIM_MODE)
    
    # 2. Mecanum Drive Wrapper
    drive = MecanumDrive(ros_board)
    logger.info("Mecanum Drive initialized")

    # 3. Camera (Orbbec RGB)
    logger.info("Initializing Camera...")
    # Using NativeCamera (OpenCV) for now. 
    # Orbbec driver should be integrated here later for Depth.
    camera = NativeCamera(device=CAMERA_INDEX, width=640, height=480, sim_mode=SIM_MODE)

    # 4. YOLO Model
    try:
        logger.info(f"Loading YOLO: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        logger.error(f"YOLO Load Failed: {e}")

    # 5. YDLidar
    logger.info("Initializing Lidar...")
    lidar = YDLidarDriver(port=LIDAR_PORT, sim_mode=SIM_MODE)

    # 6. Robot State (EKF)
    robot_state = RobotState()
    
    logger.info("="*50)
    logger.info("Initialization Complete")
    logger.info("="*50)

def cleanup():
    logger.info("Cleaning up...")
    if ros_board: ros_board.cleanup()
    if camera: camera.cleanup()
    if lidar: lidar.cleanup()

# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

async def handle_client(websocket):
    logger.info("Client connected")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "set_move":
                    # Holonomic Move Command: vx, vy, omega
                    vx = float(data.get("vx", 0.0))
                    vy = float(data.get("vy", 0.0))
                    omega = float(data.get("omega", 0.0))
                    
                    if drive:
                        drive.move(vx, vy, omega)
                        
                elif msg_type == "move": 
                    # Legacy D-Pad Support (Forward, Back, Left, Right)
                    direction = data.get("direction")
                    if drive:
                        if direction == "forward": drive.move(0.5, 0, 0)
                        elif direction == "backward": drive.move(-0.5, 0, 0)
                        elif direction == "left": drive.move(0, 0, 0.5) # Rotate Left
                        elif direction == "right": drive.move(0, 0, -0.5) # Rotate Right
                        elif direction == "strafe_left": drive.move(0, -0.5, 0)
                        elif direction == "strafe_right": drive.move(0, 0.5, 0)
                        elif direction == "stop": drive.move(0, 0, 0)

                elif msg_type == "stop":
                    if drive: drive.move(0, 0, 0)

                elif msg_type == "toggle_detection":
                    global detection_enabled
                    detection_enabled = data.get("enabled", False)
                    logger.info(f"Detection: {detection_enabled}")

            except Exception as e:
                logger.error(f"Msg Error: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        logger.info("Client disconnected")

# =============================================================================
# MAIN LOOPS
# =============================================================================

async def broadcast_loop():
    """Broadcast sensor data to clients."""
    global last_detections
    
    while True:
        if connected_clients:
            # 1. Get Frame
            frame = camera.get_frame() if camera else None
            
            # 2. YOLO
            if detection_enabled and frame is not None and model:
                results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_SIZE)
                last_detections = []
                for r in results:
                    # Draw boxes (Basic)
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        last_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(box.conf[0])
                        })

            # 3. Encode Image
            img_str = ""
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                img_str = base64.b64encode(buffer).decode('utf-8')
            
            # 4. Get Lidar (Simulated or Real)
            scan_points = lidar.get_points_xy() if lidar else []
            
            # 5. Send Update
            msg = {
                "type": "telemetry",
                "image": img_str,
                "lidar": scan_points,
                "voltage": 12.0 # Placeholder
            }
            
            # Broadcast
            websockets.broadcast(connected_clients, json.dumps(msg))
            
        await asyncio.sleep(0.05) # 20 FPS Cap

async def main():
    initialize_hardware()
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        logger.info("Server started on ws://0.0.0.0:8765")
        await broadcast_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
