"""
Viam Rover Control Server - NATIVE RASPBERRY PI (No Viam SDK)

This WebSocket server directly controls hardware on the Raspberry Pi,
eliminating Viam's 100 API call/sec limit and network latency.

Hardware Control:
- Motors: gpiozero (In1/In2 + PWM)
- Encoders: GPIO interrupt counting
- Camera: OpenCV VideoCapture
- LIDAR: rplidar-roboticia
- IMU: mpu6050 via smbus2
- Power: INA219 via pi_ina219

Usage:
    python server_native.py          # Normal mode
    python server_native.py --sim    # Simulation mode (no hardware)

Dependencies (install on Pi):
    pip install gpiozero RPi.GPIO opencv-python ultralytics websockets numpy
    pip install rplidar-roboticia smbus2 pi-ina219
"""

import asyncio
import time
import json
import signal
import argparse
import threading
import base64
import numpy as np
import cv2
import websockets
from ultralytics import YOLO

# Import local modules
from navigation_fsm import NavigationFSM, NavigationConfig
from drivers import (
    NativeMotor, NativeEncoder, NativeIMU, NativeLidar, Picamera2Driver, NativePowerSensor,
    configure_pin_factory,
    IMU_I2C_BUS, IMU_I2C_ADDRESS, IMU_SAMPLE_RATE,
    TILT_SAFETY_ENABLED, STUCK_DETECTION_ENABLED,
    STUCK_MOTOR_THRESHOLD, STUCK_TIME_THRESHOLD, STUCK_ENCODER_THRESHOLD
)
from training.capture_blur_dataset import capture_sweep, DEFAULT_SAVE_DIR
from robot_state import RobotState

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Native Viam Rover Control Server')
parser.add_argument('--sim', action='store_true', help='Run in simulation mode (no hardware)')
args = parser.parse_args()
SIM_MODE = args.sim

if not SIM_MODE:
    # Try to set up gpiozero with a working pin factory for Pi 5
    if not configure_pin_factory():
        print("⚠ Falling back to SIMULATION MODE due to GPIO setup failure.")
        SIM_MODE = True

# =============================================================================
# GPIO PIN CONFIGURATION
# =============================================================================
# Left Motor (Physically connected to Right Pins)
LEFT_MOTOR_PIN_A = 31
LEFT_MOTOR_PIN_B = 29
LEFT_MOTOR_PWM = 15

# Right Motor (Physically connected to Left Pins)
RIGHT_MOTOR_PIN_A = 35
RIGHT_MOTOR_PIN_B = 33
RIGHT_MOTOR_PWM = 37

# Encoders
LEFT_ENCODER_PIN = 40
RIGHT_ENCODER_PIN = 38

# Camera (IMX708 - Pi Camera Module 3 via CSI)
CAMERA_PATH = "/dev/video0"
CAMERA_INDEX = 0

# LIDAR
LIDAR_PORT = "/dev/ttyUSB0"

# =============================================================================
# ROBOT PARAMETERS (App Level)
# =============================================================================
# Drift Compensation
DRIFT_COMPENSATION = -0.10

# Detection Configuration
KNOWN_HEIGHT_BOTTLE = 20.0
KNOWN_HEIGHT_CAN = 15.7  # Updated to 15.7cm (16oz Coke can)
FOCAL_LENGTH = 1298
TARGET_CLASSES = [0]  # 0=can

# Camera Settings
CAMERA_HFOV_DEG = 66.0  # IMX708 Standard FOV
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 864

# Performance Settings
VIDEO_FPS_CAP = 20
JPEG_QUALITY = 70
DETECTION_INTERVAL = 1
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 640

# YOLO Model
YOLO_MODEL = 'yolo11n_cans.onnx'
YOLO_FALLBACK = 'yolo11n_cans.pt'


# =============================================================================
# GLOBAL STATE
# =============================================================================

left_motor = None
right_motor = None
left_encoder = None
right_encoder = None
camera = None
lidar = None
imu = None
power_sensor = None
robot_state = RobotState()

# Navigation
fsm = None
detection_enabled = False
last_detections = []
frame_count = 0
is_auto_driving = False

# Status flags
is_stuck = False
is_tilted = False
_stuck_start_time = None
_last_encoder_count = 0

connected_clients = set()


# =============================================================================
# DETECTION ENGINE
# =============================================================================
model = None

def initialize_detection():
    """Load YOLO model."""
    global model
    try:
        print(f"  ⚡ Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ⚠ Failed to load ONNX model: {e}")
        try:
            print(f"  ⚡ Loading fallback model: {YOLO_FALLBACK}")
            model = YOLO(YOLO_FALLBACK)
            print("  ✓ Fallback model loaded")
        except Exception as e2:
            print(f"  ✗ Failed to load fallback model: {e2}")
            model = None

def process_detection(frame):
    """Run object detection on the frame."""
    global model
    if model is None:
        return frame, []
    
    detections = []
    
    try:
        results = model.predict(
            frame, 
            conf=CONFIDENCE_THRESHOLD, 
            imgsz=INFERENCE_SIZE, 
            verbose=False,
            classes=TARGET_CLASSES
        )
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Calculate center and distance
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Height-based distance estimation
                height_px = y2 - y1
                # Distance = (Real Height * Focal Length) / Image Height
                distance_cm = (KNOWN_HEIGHT_CAN * FOCAL_LENGTH) / height_px
                
                label = f"Can: {distance_cm:.1f}cm ({conf:.2f})"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'distance_cm': float(distance_cm)
                })
                
    except Exception as e:
        print(f"Detection error: {e}")
        
    return frame, detections


# =============================================================================
# INITIALIZATION
# =============================================================================
def initialize_hardware():
    """Initialize all hardware components."""
    global left_motor, right_motor, left_encoder, right_encoder, camera, lidar, imu, power_sensor, fsm
    
    print("\n" + "="*50)
    print("Initializing Hardware (Native GPIO)")
    print("="*50)
    
    # Motors
    print("\nMotors:")
    left_motor = NativeMotor(LEFT_MOTOR_PIN_A, LEFT_MOTOR_PIN_B, LEFT_MOTOR_PWM, sim_mode=SIM_MODE, name="left_motor")
    right_motor = NativeMotor(RIGHT_MOTOR_PIN_A, RIGHT_MOTOR_PIN_B, RIGHT_MOTOR_PWM, sim_mode=SIM_MODE, name="right_motor")
    
    # Encoders
    print("\nEncoders:")
    left_encoder = NativeEncoder(LEFT_ENCODER_PIN, sim_mode=SIM_MODE, name="left_encoder")
    right_encoder = NativeEncoder(RIGHT_ENCODER_PIN, sim_mode=SIM_MODE, name="right_encoder")
    
    # Link motors to encoders
    left_motor.set_encoder(left_encoder)
    right_motor.set_encoder(right_encoder)
    print("  ✓ Motors linked to encoders for direction tracking")
    
    # IMU
    print("\nIMU (MPU6050):")
    imu = NativeIMU(IMU_I2C_BUS, IMU_I2C_ADDRESS, sim_mode=SIM_MODE, name="imu")
    
    # Camera
    print("\nCamera:")
    # Camera
    print("\nCamera:")
    # camera = NativeCamera(CAMERA_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, sim_mode=SIM_MODE)
    # Using Picamera2 for Zone Focusing
    camera = Picamera2Driver(IMAGE_WIDTH, IMAGE_HEIGHT, sim_mode=SIM_MODE)
    
    # LIDAR
    print("\nLIDAR:")
    lidar = NativeLidar(LIDAR_PORT, sim_mode=SIM_MODE)
    
    # Power Sensor (INA219)
    print("\nPower Sensor:")
    power_sensor = NativePowerSensor(sim_mode=SIM_MODE)
    
    # Detection model
    print("\nDetection:")
    initialize_detection()
    
    # Initialize Navigation FSM
    print("\nNavigation FSM:")
    nav_config = NavigationConfig()
    nav_config.camera_hfov_deg = CAMERA_HFOV_DEG
    nav_config.frame_width = IMAGE_WIDTH
    nav_config.drift_compensation = DRIFT_COMPENSATION
    fsm = NavigationFSM(left_motor, right_motor, camera=camera, imu=imu, config=nav_config)
    
    # Wire up callbacks
    def on_arrived():
        global is_auto_driving
        if fsm.config.auto_return:
            print("🎉 FSM Callback: TARGET REACHED! Waiting 5s before return...")
        else:
            print("🎉 FSM Callback: TARGET REACHED! Disengaging auto-drive.")
            is_auto_driving = False

    def on_returned():
        global is_auto_driving
        print("🎉 FSM Callback: RETURNED TO START! Disengaging auto-drive.")
        is_auto_driving = False
    
    fsm.on_arrived = on_arrived
    fsm.on_returned = on_returned
    print("✓ FSM initialized (IMU enabled)" if imu else "✓ FSM initialized (Camera only)")
    
    print("\n" + "="*50)
    print("✓ Hardware initialization complete")
    print("="*50 + "\n")

def cleanup():
    """Cleanup all hardware resources."""
    print("\nCleaning up...")
    if left_motor: left_motor.cleanup()
    if right_motor: right_motor.cleanup()
    if left_encoder: left_encoder.cleanup()
    if right_encoder: right_encoder.cleanup()
    if camera: camera.cleanup()
    if lidar: lidar.cleanup()
    if imu: imu.cleanup()
    print("Cleanup complete.")


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================
async def handle_client(websocket):
    """Handle incoming WebSocket connections."""
    print("Client connected")
    connected_clients.add(websocket)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "set_power":
                    if "motor" in data:
                        # Individual motor control (Sliders/D-Pad)
                        motor = data.get("motor")
                        power = float(data.get("power", 0.0))
                        
                        if motor == "left" and left_motor:
                            if abs(power) > 0.1: power *= (1.0 + DRIFT_COMPENSATION)
                            left_motor.set_power(power)
                        elif motor == "right" and right_motor:
                            right_motor.set_power(power)
                    else:
                        # Dual motor control (Legacy or Batch)
                        l_pow = float(data.get("left", 0))
                        r_pow = float(data.get("right", 0))
                        
                        # Apply drift compensation
                        if abs(l_pow) > 0.1:
                            l_pow *= (1.0 + DRIFT_COMPENSATION)
                        
                        if left_motor: left_motor.set_power(l_pow)
                        if right_motor: right_motor.set_power(r_pow)
                    
                elif msg_type == "stop":
                    if left_motor: left_motor.stop()
                    if right_motor: right_motor.stop()
                    
                elif msg_type == "toggle_detection":
                    global detection_enabled
                    detection_enabled = data.get("enabled", False)
                    print(f"Detection {'Enabled' if detection_enabled else 'Disabled'}")
                    
                    # Reset to autofocus when detection is disabled
                    if not detection_enabled and camera and hasattr(camera, 'set_focus'):
                        camera.set_focus(0.0)
                        print("  📷 Camera set to autofocus")
                    
                elif msg_type == "start_auto_drive":
                    global is_auto_driving
                    is_auto_driving = True
                    print("🚀 AUTO-DRIVE ENGAGED")
                    
                    # RESET ALL STATE (Zero Heading/Position)
                    print("  🔄 Resetting Robot State & IMU to (0,0)")
                    if robot_state:
                        robot_state.x = 0.0
                        robot_state.y = 0.0
                        robot_state.theta = 0.0
                        robot_state.initialized = False # Force re-init of encoder baseline
                    
                    if imu:
                        imu.reset_heading()
                        
                    if fsm:
                        # Reset FSM state
                        await fsm.start()
                        
                elif msg_type == "stop_auto_drive":
                    is_auto_driving = False
                    print("🛑 AUTO-DRIVE DISENGAGED")
                    if left_motor: left_motor.stop()
                    if right_motor: right_motor.stop()
                    
                elif msg_type == "capture_image":
                    # Capture current frame and save for training
                    if camera:
                        frame = camera.get_frame()
                        if frame is not None:
                            # Create directory structure for organized training data
                            import os
                            from datetime import datetime
                            
                            base_dir = "training_images"
                            
                            # Categorize by distance if detection is available
                            distance_cm = None
                            detection_info = None
                            if last_detections and len(last_detections) > 0:
                                det = last_detections[0]  # Primary detection
                                distance_cm = det.get('distance_cm')
                                detection_info = {
                                    'center_x': det.get('center_x'),
                                    'center_y': det.get('center_y'),
                                    'width': det.get('width'),
                                    'height': det.get('height'),
                                    'confidence': det.get('confidence'),
                                    'distance_cm': distance_cm
                                }
                            
                            # Determine subdirectory based on distance
                            if distance_cm is not None:
                                if distance_cm < 30:
                                    subdir = "close"
                                elif distance_cm < 100:
                                    subdir = "medium"
                                else:
                                    subdir = "far"
                            else:
                                subdir = "uncategorized"
                            
                            capture_dir = os.path.join(base_dir, subdir)
                            os.makedirs(capture_dir, exist_ok=True)
                            
                            # Save image with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            img_filename = f"can_{timestamp}.jpg"
                            filepath = os.path.join(capture_dir, img_filename)
                            cv2.imwrite(filepath, frame)
                            
                            # Save metadata JSON
                            if detection_info:
                                meta_filepath = os.path.join(capture_dir, f"can_{timestamp}.json")
                                with open(meta_filepath, 'w') as f:
                                    json.dump({
                                        'image': img_filename,
                                        'frame_width': frame.shape[1],
                                        'frame_height': frame.shape[0],
                                        'detection': detection_info
                                    }, f, indent=2)
                            
                            # Count total images
                            total_count = 0
                            for root, dirs, files in os.walk(base_dir):
                                total_count += len([f for f in files if f.endswith('.jpg')])
                            
                            dist_str = f" ({distance_cm:.0f}cm)" if distance_cm else ""
                            print(f"📸 Captured: {filepath}{dist_str} (total: {total_count})")
                            
                            # Send response back to client
                            await websocket.send(json.dumps({
                                "type": "capture_response",
                                "count": total_count,
                                "filename": filepath,
                                "category": subdir,
                                "distance_cm": distance_cm
                            }))
                            
                elif msg_type == "download_images":
                    # Zip all training images and send to client
                    should_clear = data.get("clear", False)
                    import zipfile
                    import io
                    import shutil
                    from datetime import datetime
                    
                    base_dir = "training_images"
                    
                    if not os.path.exists(base_dir):
                        await websocket.send(json.dumps({
                            "type": "download_images_response",
                            "success": False,
                            "error": "No training_images folder found"
                        }))
                    else:
                        # Create zip in memory
                        zip_buffer = io.BytesIO()
                        image_count = 0
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for root, dirs, files in os.walk(base_dir):
                                for file in files:
                                    if file.endswith(('.jpg', '.jpeg', '.png', '.json')):
                                        filepath = os.path.join(root, file)
                                        # Preserve folder structure in zip
                                        arcname = os.path.relpath(filepath, base_dir)
                                        zf.write(filepath, arcname)
                                        if file.endswith(('.jpg', '.jpeg', '.png')):
                                            image_count += 1
                        
                        if image_count == 0:
                            await websocket.send(json.dumps({
                                "type": "download_images_response",
                                "success": False,
                                "error": "No images found in training_images folder"
                            }))
                        else:
                            # Convert to base64
                            zip_buffer.seek(0)
                            zip_b64 = base64.b64encode(zip_buffer.read()).decode('utf-8')
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"training_images_{timestamp}.zip"
                            
                            print(f"📦 Sending {image_count} images as {filename}")
                            
                            await websocket.send(json.dumps({
                                "type": "download_images_response",
                                "success": True,
                                "zip_data": zip_b64,
                                "filename": filename,
                                "image_count": image_count
                            }))
                            
                            # Clear images if requested (AFTER successful zip)
                            if should_clear:
                                print(f"  🗑️ Clearing {image_count} images from {base_dir}...")
                                try:
                                    # Delete contents of training_images but keep the directory
                                    for root, dirs, files in os.walk(base_dir, topdown=False):
                                        for name in files:
                                            os.remove(os.path.join(root, name))
                                        for name in dirs:
                                            os.rmdir(os.path.join(root, name))
                                    print("  ✅ Images cleared.")
                                except Exception as e:
                                    print(f"  ❌ Failed to clear images: {e}")
                             
                            
                elif msg_type == "collect_blur_dataset":
                    if camera:
                        print("  📸 Starting Blur Dataset Sweep...")
                        
                        # Determine efficient index
                        import os
                        dataset_index = 0
                        if os.path.exists(DEFAULT_SAVE_DIR):
                            existing_files = [f for f in os.listdir(DEFAULT_SAVE_DIR) if f.startswith("sweep_")]
                            if existing_files:
                                try:
                                    indices = [int(f.split('_')[1]) for f in existing_files]
                                    if indices:
                                        dataset_index = max(indices) + 1
                                except:
                                    pass

                        await websocket.send(json.dumps({
                            "type": "blur_dataset_response",
                            "status": "started",
                            "index": dataset_index
                        }))
                        
                        # Run sweep in thread
                        count = await asyncio.to_thread(capture_sweep, camera, DEFAULT_SAVE_DIR, dataset_index)
                        
                        await websocket.send(json.dumps({
                            "type": "blur_dataset_response",
                            "status": "complete",
                            "count": count,
                            "index": dataset_index
                        }))
                        print(f"  ✓ Sweep #{dataset_index} complete ({count} images)")
                    else:
                         await websocket.send(json.dumps({
                            "type": "blur_dataset_response",
                            "status": "error",
                            "message": "Camera not initialized"
                        }))

                            
            except Exception as e:
                print(f"Message handling error: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(connected_clients)}")


async def broadcast_loop():
    """Broadcast sensor data to all connected clients."""
    global frame_count, last_detections, is_auto_driving, is_stuck, is_tilted
    global _stuck_start_time, _last_encoder_count
    
    last_video_time = 0
    video_interval = 1.0 / VIDEO_FPS_CAP
    last_imu_time = 0
    imu_interval = 1.0 / IMU_SAMPLE_RATE
    
    # FPS tracking
    fps_frame_count = 0
    fps_detection_count = 0
    fps_last_time = time.time()
    fps_camera = 0.0
    fps_detection = 0.0
    
    while True:
        if connected_clients:
            current_time = time.time()
            
            # === IMU UPDATE (high frequency) ===
            if imu and current_time - last_imu_time >= imu_interval:
                last_imu_time = current_time
                imu.update()
                
                # --- TILT SAFETY CHECK ---
                if TILT_SAFETY_ENABLED and imu.is_tilted_unsafe():
                    if not is_tilted:
                        is_tilted = True
                        print("⚠️ TILT SAFETY: Rover tilted too far! Emergency stop.")
                        if left_motor: left_motor.stop()
                        if right_motor: right_motor.stop()
                        is_auto_driving = False
                elif is_tilted:
                    is_tilted = False
                    print("✓ Tilt returned to safe range")
                
                # --- STUCK DETECTION ---
                if STUCK_DETECTION_ENABLED and left_motor and right_motor:
                    motor_power = max(abs(left_motor.power), abs(right_motor.power))
                    
                    if motor_power > STUCK_MOTOR_THRESHOLD:
                        encoder_count = 0
                        if left_encoder: encoder_count += abs(left_encoder.get_count())
                        if right_encoder: encoder_count += abs(right_encoder.get_count())
                        
                        imu_moving = imu.is_moving() if imu else True
                        encoder_moving = abs(encoder_count - _last_encoder_count) > STUCK_ENCODER_THRESHOLD
                        
                        if not imu_moving and not encoder_moving:
                            if _stuck_start_time is None:
                                _stuck_start_time = current_time
                            elif current_time - _stuck_start_time > STUCK_TIME_THRESHOLD:
                                if not is_stuck:
                                    is_stuck = True
                                    print("⚠️ STUCK: Motors running but no movement detected!")
                        else:
                            _stuck_start_time = None
                            if is_stuck:
                                is_stuck = False
                                print("✓ Movement detected, no longer stuck")
                        
                        _last_encoder_count = encoder_count
                    else:
                        _stuck_start_time = None
            
            # Calculate velocities (units/sec)
            dt_video = current_time - last_video_time if last_video_time > 0 else 0.1 # Approx
            
            # --- UPDATE ODOMETRY ---
            l_pos = left_encoder.get_position() if left_encoder else 0
            r_pos = right_encoder.get_position() if right_encoder else 0
            
            # Simple velocity estimation (could be noisy, good for debug graph)
            # Store previous positions to calculate delta
            if not hasattr(broadcast_loop, "last_pos_l"):
                broadcast_loop.last_pos_l = l_pos
                broadcast_loop.last_pos_r = r_pos
                broadcast_loop.last_vel_time = current_time
            
            vel_l = 0.0
            vel_r = 0.0
            
            if current_time - broadcast_loop.last_vel_time > 0.0:
                dt_vel = current_time - broadcast_loop.last_vel_time
                vel_l = (l_pos - broadcast_loop.last_pos_l) / dt_vel
                vel_r = (r_pos - broadcast_loop.last_pos_r) / dt_vel
                
                broadcast_loop.last_pos_l = l_pos
                broadcast_loop.last_pos_r = r_pos
                broadcast_loop.last_vel_time = current_time

            if left_encoder and right_encoder:
                if imu:
                    robot_state.update_with_imu(
                        left_encoder.get_position(),
                        right_encoder.get_position(),
                        imu.get_heading()
                    )
                else:
                    robot_state.update(
                        left_encoder.get_position(),
                        right_encoder.get_position()
                    )
            
            # Get camera frame
            frame = camera.get_frame() if camera else None
            image_b64 = None
            
            if frame is not None:
                frame_count += 1
                fps_frame_count += 1
                
                # Run detection
                if detection_enabled and frame_count % DETECTION_INTERVAL == 0:
                    frame, last_detections = process_detection(frame)
                    fps_detection_count += 1
                elif detection_enabled:
                    for d in last_detections:
                        x1, y1, x2, y2 = d['bbox']
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                
                # Calculate FPS
                fps_elapsed = current_time - fps_last_time
                if fps_elapsed >= 1.0:
                    fps_camera = fps_frame_count / fps_elapsed
                    fps_detection = fps_detection_count / fps_elapsed
                    fps_frame_count = 0
                    fps_detection_count = 0
                    fps_last_time = current_time
                
                # === AUTO-DRIVE CONTROL (FSM) ===
                if is_auto_driving and fsm:
                    detection = None
                    target_pose = None
                    
                    if last_detections:
                        target = min(last_detections, key=lambda d: d['distance_cm'])
                        detection = {
                            'distance_cm': target['distance_cm'],
                            'center_x': target['center_x']
                        }
                        
                        # Calculate target_pose (world coordinates) for map-based navigation
                        det_distance = target['distance_cm']
                        det_center_x = target['center_x']
                        pixel_offset = det_center_x - (IMAGE_WIDTH / 2)
                        bearing = pixel_offset * (CAMERA_HFOV_DEG / IMAGE_WIDTH) * (np.pi / 180.0)
                        
                        # Local frame (Y-forward)
                        target_local_x = det_distance * np.sin(bearing)
                        target_local_y = det_distance * np.cos(bearing)
                        
                        # World frame
                        cos_theta = np.cos(robot_state.theta)
                        sin_theta = np.sin(robot_state.theta)
                        target_world_x = robot_state.x + (target_local_x * cos_theta + target_local_y * sin_theta)
                        target_world_y = robot_state.y + (-target_local_x * sin_theta + target_local_y * cos_theta)
                        
                        target_pose = {
                            'x': target_world_x,
                            'y': target_world_y,
                            'distance_cm': det_distance
                        }
                    
                    lidar_min = None
                    if lidar:
                        scan = lidar.get_scan()
                        front_dists = [d for a, d in scan if -0.78 < a < 0.78]
                        if front_dists:
                            lidar_min = min(front_dists) * 100.0
                    
                    await fsm.update(
                        detection,
                        target_pose=target_pose,
                        lidar_min_distance_cm=lidar_min,
                        current_pose={
                            'x': robot_state.x,
                            'y': robot_state.y,
                            'theta': robot_state.theta
                        }
                    )
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            last_video_time = current_time

            try:
                # Build data packet
                # Gather Raw IMU
                raw_accel = {'x': 0, 'y': 0, 'z': 0}
                raw_gyro = {'x': 0, 'y': 0, 'z': 0}
                if imu:
                    ax, ay, az = imu.get_accel()
                    gx, gy, gz = imu.get_gyro()
                    raw_accel = {'x': float(ax), 'y': float(ay), 'z': float(az)}
                    raw_gyro = {'x': float(gx), 'y': float(gy), 'z': float(gz)}

                data = {
                    "type": "readout",
                    "left_pos": float(l_pos),
                    "left_power": float(left_motor.power) if left_motor else 0,
                    "right_pos": float(r_pos),
                    "right_power": float(right_motor.power) if right_motor else 0,
                    "motor_velocity": {
                        "left": float(vel_l),
                        "right": float(vel_r)
                    },
                    "image": image_b64,
                    "detection_enabled": bool(detection_enabled),
                    "detections": last_detections,
                    "is_auto_driving": bool(is_auto_driving),
                    "is_stuck": bool(is_stuck),
                    "is_tilted": bool(is_tilted),
                    "fps_camera": float(fps_camera),
                    "fps_detection": float(fps_detection),
                    "robot_pose": {
                        "x": float(robot_state.x),
                        "y": float(robot_state.y),
                        "theta": float(robot_state.theta)
                    },
                    "target_pose": {
                        "x": float(fsm.ctx.goal_x),
                        "y": float(fsm.ctx.goal_y),
                        "distance_cm": float(fsm.ctx.goal_distance)
                    } if fsm and fsm.ctx.goal_x is not None else None,
                    "imu": {
                        "pitch": float(imu.get_tilt()[0]) if imu else 0,
                        "roll": float(imu.get_tilt()[1]) if imu else 0,
                        "heading_deg": float(np.degrees(imu.get_heading())) if imu else 0,
                        "yaw_rate": float(imu.get_gyro()[2]) if imu else 0,
                        "raw_accel": raw_accel,
                        "raw_gyro": raw_gyro
                    } if imu else None,
                    "auto_drive_start": {
                        "x": float(fsm.ctx.start_pose['x']),
                        "y": float(fsm.ctx.start_pose['y'])
                    } if fsm and (is_auto_driving or fsm.state != "IDLE") else None,
                    "lidar_points": lidar.get_points_xy()[:360] if lidar else [],
                    "fsm_state": fsm.state_summary if fsm else "IDLE",
                    "power": power_sensor.get_all() if power_sensor else None
                }
                
                # Update target_pose from best detection (live refinement)
                if last_detections and len(last_detections) > 0:
                    best_det = last_detections[0]
                    det_distance = best_det.get('distance_cm', 0)
                    det_center_x = best_det.get('center_x', IMAGE_WIDTH / 2)
                    
                    # Calculate bearing angle
                    # Pixel Offset: Negative = Left, Positive = Right
                    # We want Negative Bearing for Left (-X in World)
                    pixel_offset = det_center_x - (IMAGE_WIDTH / 2)
                    bearing = pixel_offset * (CAMERA_HFOV_DEG / IMAGE_WIDTH) * (np.pi / 180.0)
                    
                    # Calculate Local Position
                    target_local_x = det_distance * np.sin(bearing)
                    target_local_y = det_distance * np.cos(bearing)
                    
                    # Transform to World Frame
                    cos_theta = np.cos(robot_state.theta)
                    sin_theta = np.sin(robot_state.theta)
                    
                    target_world_x = robot_state.x + (target_local_x * cos_theta + target_local_y * sin_theta)
                    target_world_y = robot_state.y + (-target_local_x * sin_theta + target_local_y * cos_theta)
                    
                    # Update data packet
                    data["target_pose"] = {
                        "x": float(target_world_x),
                        "y": float(target_world_y),
                        "distance_cm": float(det_distance)
                    }
                
                # Calculate trajectory arc for 3D visualization
                trajectory_points = []
                if fsm and fsm.ctx.goal_x is not None and fsm.ctx.goal_y is not None:
                    # Generate arc points from robot to goal
                    rx, ry = robot_state.x, robot_state.y
                    gx, gy = fsm.ctx.goal_x, fsm.ctx.goal_y
                    
                    # Simple interpolation with curve based on bearing
                    dx = gx - rx
                    dy = gy - ry
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 5:  # Only show trajectory if far enough
                        # Calculate bearing for arc curvature
                        cos_t = np.cos(robot_state.theta)
                        sin_t = np.sin(robot_state.theta)
                        local_x = -(dx * cos_t - dy * sin_t)
                        local_y = dx * sin_t + dy * cos_t
                        bearing = np.arctan2(local_x, local_y)
                        
                        # Generate 5 points along a curved path
                        for i in range(6):
                            t = i / 5.0  # 0 to 1
                            # Quadratic Bezier curve for smooth arc
                            # Control point offset based on bearing
                            ctrl_offset = dist * 0.3 * np.sin(bearing)
                            
                            # Control point perpendicular to direct line
                            mid_x = (rx + gx) / 2 + ctrl_offset * cos_t
                            mid_y = (ry + gy) / 2 + ctrl_offset * sin_t
                            
                            # Quadratic Bezier: B(t) = (1-t)²*P0 + 2(1-t)*t*P1 + t²*P2
                            px = (1-t)**2 * rx + 2*(1-t)*t * mid_x + t**2 * gx
                            py = (1-t)**2 * ry + 2*(1-t)*t * mid_y + t**2 * gy
                            
                            trajectory_points.append({"x": float(px), "y": float(py)})
                
                data["trajectory"] = trajectory_points
                
                message = json.dumps(data)
                await asyncio.gather(
                    *[client.send(message) for client in connected_clients],
                    return_exceptions=True
                )

            except Exception as e:
                print(f"Broadcast Error: {e}")
                import traceback
                traceback.print_exc()
        
        await asyncio.sleep(0.001)


async def main():
    print("\n" + "="*60)
    print("  NATIVE RASPBERRY PI SERVER (Refactored)")
    print("  Zero API limits | Zero network latency")
    print("="*60)
    
    if SIM_MODE:
        print("\n⚠ SIMULATION MODE - No hardware control")
    
    initialize_hardware()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    server = await websockets.serve(
        handle_client, "0.0.0.0", 8081,
        ping_interval=20, ping_timeout=60
    )
    
    print(f"\n{'='*50}")
    print(f"WebSocket server running on ws://0.0.0.0:8081")
    print(f"{'='*50}\n")
    
    broadcast_task = asyncio.create_task(broadcast_loop())
    
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        broadcast_task.cancel()
        server.close()
        await server.wait_closed()
        cleanup()

async def shutdown():
    print("\nShutting down...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks: task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
