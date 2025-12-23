"""
Viam Rover Control Server - JETSON ORIN NANO OPTIMIZED

This WebSocket server connects to a Viam-powered rover and provides:
- Motor control (left/right)
- Camera feed streaming
- Lidar data streaming
- Real-time bottle/can detection using YOLOv8 (GPU accelerated)

Optimized for Jetson Orin Nano with NVIDIA GPU (CUDA inference).
For Raspberry Pi 5 (CPU), use server_pi.py instead.

Usage:
    python server_jetson.py
"""

import asyncio
import time
import json
import websockets
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.motor import Motor
from viam.components.camera import Camera
from viam.components.sensor import Sensor
from viam.components.power_sensor import PowerSensor
from viam.components.encoder import Encoder
import base64
import io
import struct
import numpy as np
import cv2
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Robot Connection Details
ROBOT_ADDRESS = "yeep-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "d55dcbc4-6c31-4d78-97d9-57792293a0b7"
API_KEY = "3u88u6fsuowp1wv4inpyebnv13k6dkhn"

# Component Names
LEFT_MOTOR_NAME = "left"
RIGHT_MOTOR_NAME = "right"
CAMERA_NAME = "cam"
LIDAR_NAME = None  # No lidar on this robot
BATTERY_NAME = "ina219"
IMU_NAME = "imu" # Assuming an IMU named "imu"

# Detection Configuration
KNOWN_HEIGHT_BOTTLE = 20.0  # Standard water bottle height in cm
KNOWN_HEIGHT_CAN = 12.0     # Standard soda can height in cm
FOCAL_LENGTH = 600          # Webcam focal length (calibrate for accuracy)
TARGET_CLASSES = [0]        # Custom model: 0=can (for yolov8n_cans.pt)

# Camera Calibration (HYPER hypercam HD 1080 = 76.5° HFOV)
CAMERA_HFOV_DEG = 76.5      # Horizontal field of view in degrees
IMAGE_WIDTH = 640           # Camera resolution width
IMAGE_HEIGHT = 480          # Camera resolution height

# Wheel Parameters (CALIBRATE THESE! - Measured for your rover)
WHEEL_DIAMETER_CM = 5.5     # Wheel diameter in cm
WHEEL_BASE_CM = 19.55       # Distance between wheel contact points in cm

# =============================================================================
# JETSON ORIN NANO GPU OPTIMIZATION SETTINGS
# =============================================================================
# These settings are tuned for GPU-accelerated inference on Orin Nano

# Use GPU acceleration
USE_GPU = True              # Enable CUDA acceleration

# YOLO Model - use custom trained can detection model
YOLO_MODEL = 'yolov8n_cans.pt'   # Custom trained for soda cans

# Inference resolution - higher = better accuracy with GPU
INFERENCE_SIZE = 640        # 640px for Jetson (Pi: 320)

# Detection interval - run YOLO every N frames
DETECTION_INTERVAL = 1      # Every frame with GPU (Pi: 3)

# Video frame rate cap (GPU can handle higher)
VIDEO_FPS_CAP = 30          # 30 FPS for Jetson (Pi: 15)

# JPEG quality (higher quality with GPU headroom)
JPEG_QUALITY = 75           # 75% for Jetson (Pi: 65)

# =============================================================================
# GLOBAL STATE
# =============================================================================

robot: RobotClient = None
left_motor: Motor = None
right_motor: Motor = None
camera: Camera = None
lidar: Camera = None
battery: PowerSensor = None
imu: Sensor = None
left_encoder: Encoder = None
right_encoder: Encoder = None

# Auto-drive state
is_auto_driving: bool = False
target_distance_cm: float = 12.5  # Target distance (between 10-15cm)
dist_threshold_cm: float = 2.5    # +/- 2.5cm tolerance
center_threshold_px: int = 50     # Pixel error tolerance for centering
  
# FPS Optimization
frame_count: int = 0
detection_interval: int = DETECTION_INTERVAL  # Use config constant (every frame with GPU)
last_detections: list = []

# Trackers
tracker = None
tracker_label = ""
tracker_init_time = 0

# Detection state
detection_model: YOLO = None
detection_enabled: bool = False

# Connected WebSocket clients
connected_clients = set()

# Trajectory tracking for GUI visualization
trajectory_history = []  # List of (x, y) positions
auto_drive_start_pos = None  # Starting position when auto-drive began
MAX_TRAJECTORY_POINTS = 200  # Limit points to avoid memory issues

# =============================================================================
# STATE TRACKING CLASSES (Odometry + Target Memory)
# =============================================================================

class RobotState:
    """Track robot pose using wheel encoder odometry."""
    
    def __init__(self):
        self.x = 0.0  # Robot X position (cm)
        self.y = 0.0  # Robot Y position (cm)
        self.theta = 0.0  # Robot heading (radians)
        self.last_left_pos = 0.0
        self.last_right_pos = 0.0
        self.initialized = False  # Flag to capture initial encoder values
    
    def update_odometry(self, left_pos, right_pos, 
                        wheel_base_cm=15.0, wheel_diameter_cm=6.5):
        """Update robot pose from wheel encoders."""
        # On first call, capture initial encoder values (don't compute delta)
        if not self.initialized:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.initialized = True
            print(f"Odometry initialized: L={left_pos:.2f}, R={right_pos:.2f}")
            return 0.0, 0.0
        
        # Calculate wheel displacement (encoder returns rotations)
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        # Convert rotations to linear distance
        left_dist = left_delta * np.pi * wheel_diameter_cm
        right_dist = right_delta * np.pi * wheel_diameter_cm
        
        # Calculate robot motion
        linear_dist = (left_dist + right_dist) / 2.0
        angular_change = (right_dist - left_dist) / wheel_base_cm
        
        # Update pose using robotics convention:
        # theta=0 means facing +Y (forward), X is lateral (right)
        # Motion in world frame: X += sin(theta), Y += cos(theta)
        self.x += linear_dist * np.sin(self.theta)  # Lateral motion
        self.y += linear_dist * np.cos(self.theta)  # Forward motion
        self.theta += angular_change
        
        # Normalize theta to [-pi, pi] to prevent explosion
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos
        
        return linear_dist, angular_change


class TargetState:
    """Track target position in world frame for memory during detection gaps."""
    
    def __init__(self):
        self.world_x = None  # Target X in world frame
        self.world_y = None  # Target Y in world frame
        self.last_update_time = 0
        self.confidence = 0.0
    
    def update_from_detection(self, robot_state, detection):
        """Convert camera detection to world coordinates."""
        # Get distance and bearing to target
        distance_cm = detection['distance_cm']
        
        # Calculate bearing using proper tangent projection
        # This accounts for lens distortion better than linear approximation
        pixel_error_x = detection['center_x'] - (IMAGE_WIDTH / 2)
        fov_rad = np.deg2rad(CAMERA_HFOV_DEG)
        bearing_offset = np.arctan(2 * pixel_error_x * np.tan(fov_rad / 2) / IMAGE_WIDTH)
        
        # Calculate target position in world frame
        # Using same convention as odometry: X = sin(theta), Y = cos(theta)
        target_bearing = robot_state.theta + bearing_offset
        self.world_x = robot_state.x + distance_cm * np.sin(target_bearing)
        self.world_y = robot_state.y + distance_cm * np.cos(target_bearing)
        
        self.last_update_time = time.time()
        self.confidence = detection['confidence']

    
    def get_robot_relative_position(self, robot_state):
        """Get target position relative to current robot pose."""
        if self.world_x is None:
            return None, None
        
        # Calculate vector from robot to target
        dx = self.world_x - robot_state.x
        dy = self.world_y - robot_state.y
        
        # Distance
        distance = np.sqrt(dx**2 + dy**2)
        
        # Bearing angle relative to robot heading
        target_angle = np.arctan2(dy, dx)
        bearing_error = target_angle - robot_state.theta
        
        # Normalize to [-pi, pi]
        bearing_error = np.arctan2(np.sin(bearing_error), np.cos(bearing_error))
        
        return distance, bearing_error


class ControlState:
    """Track control history for derivative term."""
    
    def __init__(self):
        self.last_bearing_error = 0.0
        self.last_distance_error = 0.0
        self.last_time = time.time()
        self.smoothed_angular = 0.0
        self.smoothed_linear = 0.0


# Initialize state tracking objects
robot_state = RobotState()
target_state = TargetState()
control_state = ControlState()  # Add to global state


async def update_robot_state_with_imu():
    """Fuse IMU heading with encoder odometry using complementary filter."""
    global robot_state, imu
    
    if not imu:
        return
    
    try:
        readings = await imu.get_readings()
        
        # IMU may provide orientation as Euler angles or quaternion
        if 'orientation' in readings:
            orientation = readings['orientation']
            
            # Extract yaw/heading (adjust based on your IMU's output format)
            # Common formats: dict with 'z' key, or object with euler.yaw
            imu_heading = None
            if isinstance(orientation, dict) and 'z' in orientation:
                imu_heading = np.deg2rad(orientation['z'])
            elif hasattr(orientation, 'euler'):
                imu_heading = np.deg2rad(orientation.euler.yaw)
            
            if imu_heading is not None:
                # Complementary filter: Trust encoders for short-term, IMU for drift correction
                ALPHA = 0.98  # 98% encoder, 2% IMU
                robot_state.theta = (ALPHA * robot_state.theta + 
                                     (1 - ALPHA) * imu_heading)
    except Exception as e:
        pass  # Continue without IMU if read fails


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def initialize_detection_model():
    """Load the YOLO model for object detection with GPU support."""
    global detection_model
    try:
        import torch
        
        detection_model = YOLO(YOLO_MODEL)
        
        # Move model to GPU if available and USE_GPU is enabled
        if USE_GPU and torch.cuda.is_available():
            detection_model.to('cuda')
            device = next(detection_model.model.parameters()).device
            print(f"✓ YOLO {YOLO_MODEL} loaded on GPU ({device})")
        else:
            if USE_GPU:
                print(f"⚠ CUDA not available, running {YOLO_MODEL} on CPU")
            else:
                print(f"✓ YOLO {YOLO_MODEL} loaded on CPU (GPU disabled)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False


def calculate_distance(bbox_height_px: float, real_height_cm: float) -> float:
    """
    Estimate distance using the Pinhole Camera Model.
    Distance = (Real_Height * Focal_Length) / Object_Pixel_Height
    """
    if bbox_height_px <= 0:
        return 0.0
    return (real_height_cm * FOCAL_LENGTH) / bbox_height_px


def process_detection(image_bytes: bytes) -> tuple:
    """
    Process an image for bottle/can detection.
    
    Args:
        image_bytes: JPEG image as bytes
        
    Returns:
        Tuple of (annotated_image_b64, detections_list)
    """
    global detection_model
    
    if detection_model is None:
        return None, []
    
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, []
        
        # Run YOLO inference
        results = detection_model(frame, verbose=False, stream=True)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # Only process target classes (bottle, cup/can)
                if cls_id not in TARGET_CLASSES:
                    continue
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                
                # Calculate dimensions
                width_px = x2 - x1
                height_px = y2 - y1
                
                # Get label and confidence
                label = detection_model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Calculate center point
                center_x = int(x1 + width_px / 2)
                center_y = int(y1 + height_px / 2)
                
                # Estimate distance
                real_height = KNOWN_HEIGHT_BOTTLE if label == 'bottle' else KNOWN_HEIGHT_CAN
                distance_cm = calculate_distance(height_px, real_height)
                
                # Store detection data
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "distance_cm": round(distance_cm, 1),
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": [x1, y1, x2, y2],
                    "area_px": width_px * height_px
                })
                
                # Draw annotations on frame
                color = (0, 255, 0)  # Green for detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw label text
                label_text = f"{label} {confidence:.0%}"
                dist_text = f"{int(distance_cm)}cm"
                cv2.putText(frame, label_text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, dist_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode annotated frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return annotated_b64, detections
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None, []

# =============================================================================
# LIDAR PARSING
# =============================================================================

def parse_pcd(data: bytes) -> list:
    """
    Parse a PCD (Point Cloud Data) file in binary format.
    Returns a list of [x, y] points.
    """
    try:
        # Find the end of the header
        header_end_index = data.find(b"DATA binary\n")
        if header_end_index == -1:
            return []
        
        header = data[:header_end_index].decode('ascii')
        raw_data = data[header_end_index + 12:]
        
        # Parse header fields
        lines = header.split('\n')
        fields = []
        size = []
        type_ = []
        
        for line in lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                size = [int(s) for s in line.split()[1:]]
            elif line.startswith("TYPE"):
                type_ = line.split()[1:]
        
        # Parse as float32 array if all fields are floats
        if all(s == 4 for s in size) and all(t == 'F' for t in type_):
            num_floats = len(fields)
            arr = np.frombuffer(raw_data, dtype=np.float32)
            if arr.size % num_floats == 0:
                arr = arr.reshape(-1, num_floats)
                return arr[:, :2].tolist()
        
        return []
        
    except Exception as e:
        print(f"PCD parsing error: {e}")
        return []

# =============================================================================
# ROBOT CONNECTION
# =============================================================================

async def connect_to_robot() -> bool:
    """Establish connection to the Viam robot."""
    global robot, left_motor, right_motor, camera, lidar, battery, imu
    global left_encoder, right_encoder
    
    print("Connecting to robot...")
    
    try:
        options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
        robot = await RobotClient.at_address(ROBOT_ADDRESS, options)
        print(f"✓ Connected to {ROBOT_ADDRESS}")
        
        print("Available resources:")
        for name in robot.resource_names:
            print(f" - {name.name} ({name.subtype})")
        
        # Initialize motors (optional)
        try:
            left_motor = Motor.from_robot(robot, LEFT_MOTOR_NAME)
            right_motor = Motor.from_robot(robot, RIGHT_MOTOR_NAME)
            print("✓ Motors initialized")
        except Exception:
            print("✗ Motors not found")
            left_motor = None
            right_motor = None
        
        # Initialize camera (optional)
        try:
            camera = Camera.from_robot(robot, CAMERA_NAME)
            print("✓ Camera initialized")
        except Exception:
            print("✗ Camera not found")
            camera = None
        
        # Initialize lidar (optional)
        if LIDAR_NAME:
            try:
                lidar = Camera.from_robot(robot, LIDAR_NAME)
                print("✓ Lidar initialized")
            except Exception:
                print("✗ Lidar not found")
                lidar = None
        else:
            lidar = None
            
        # Initialize battery (optional)
        try:
            battery = PowerSensor.from_robot(robot, BATTERY_NAME)
            print("✓ Battery initialized")
        except Exception:
            print(f"✗ Battery '{BATTERY_NAME}' not found")
            battery = None
            
        try:
            imu = Sensor.from_robot(robot, IMU_NAME)
            print("✓ IMU initialized")
        except Exception:
            print(f"✗ IMU '{IMU_NAME}' not found")
            imu = None
        
        # Initialize encoders for odometry
        try:
            left_encoder = Encoder.from_robot(robot, "Lenc")
            right_encoder = Encoder.from_robot(robot, "Renc")
            print("✓ Encoders initialized (odometry enabled)")
        except Exception:
            print("✗ Encoders not found (odometry disabled)")
            left_encoder = None
            right_encoder = None
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        robot = None
        return False

# =============================================================================
# WEBSOCKET HANDLERS
# =============================================================================

async def producer_task():
    """
    Continuously broadcast sensor data to all connected clients.
    Runs at ~10Hz for smooth video/lidar updates.
    """
    global robot, left_motor, right_motor, camera, lidar, battery, imu
    global left_encoder, right_encoder, robot_state, target_state
    global connected_clients, detection_enabled, is_auto_driving
    global frame_count, last_detections, tracker, tracker_label, tracker_init_time
    
    last_video_time = 0
    VIDEO_INTERVAL = 1.0 / VIDEO_FPS_CAP  # Use config constant for Jetson optimization
    last_battery_time = 0
    BATTERY_INTERVAL = 5.0       # Update battery every 5s
    
    current_volts = 0.0
    current_amps = 0.0
    current_watts = 0.0
    current_pct = 0.0
    
    # Odometry-based memory (2.5D visual servoing)
    MEMORY_DURATION = 2.0  # Use target memory for 2 seconds max
    
    while True:
        current_time = time.time()
        if robot and connected_clients:
            try:
                # Gather motor data
                left_pos, left_power_data, right_pos, right_power_data = await asyncio.gather(
                    left_motor.get_position(),
                    left_motor.is_powered(),
                    right_motor.get_position(),
                    right_motor.is_powered()
                )
                
                # Build response data
                data = {
                    "type": "readout",
                    "left_pos": left_pos,
                    "left_power": left_power_data[1],
                    "right_pos": right_pos,
                    "right_power": right_power_data[1],
                    "detection_enabled": detection_enabled,
                    "is_auto_driving": is_auto_driving,
                    # Trajectory data for GUI visualization
                    "robot_pose": {
                        "x": robot_state.x,
                        "y": robot_state.y,
                        "theta": robot_state.theta
                    },
                    "target_pose": {
                        "x": target_state.world_x,
                        "y": target_state.world_y
                    } if target_state.world_x is not None else None,
                    "trajectory": trajectory_history[-MAX_TRAJECTORY_POINTS:],
                    "auto_drive_start": auto_drive_start_pos
                }
                
                # Update odometry from wheel encoders (if available)
                if left_encoder and right_encoder:
                    try:
                        left_enc_pos, _ = await left_encoder.get_position()
                        right_enc_pos, _ = await right_encoder.get_position()
                        robot_state.update_odometry(left_enc_pos, right_enc_pos,
                                                    wheel_base_cm=WHEEL_BASE_CM,
                                                    wheel_diameter_cm=WHEEL_DIAMETER_CM)
                        
                        # Fuse IMU heading to correct encoder drift
                        await update_robot_state_with_imu()
                        
                        # DEBUG: Print every 50 frames to verify encoders are working
                        if frame_count % 50 == 0:
                            print(f"Odom: x={robot_state.x:.1f}, y={robot_state.y:.1f}, θ={np.degrees(robot_state.theta):.1f}°")
                            print(f"Encoders: L={left_enc_pos:.2f}, R={right_enc_pos:.2f}")
                    except Exception as e:
                        print(f"Encoder error: {e}")
                
                # Get battery data
                if battery:
                    try:
                        # For PowerSensor, get_voltage() returns (volts, is_ac)
                        volts, is_ac = await battery.get_voltage()
                        # Also could get watts, current, etc.
                        data["battery"] = {"volts": volts}
                    except Exception:
                        pass
                
                if camera:
                    try:
                        # --- Battery Logic ---
                        if battery and (current_time - last_battery_time > BATTERY_INTERVAL):
                            try:
                                # get_voltage returns (volts, is_ac)
                                volts_data = await battery.get_voltage()
                                volts = volts_data[0] if isinstance(volts_data, tuple) else volts_data
                                current_volts = round(volts, 2)
                                
                                # get_current returns (amps, is_ac)
                                amps_data = await battery.get_current()
                                amps = amps_data[0] if isinstance(amps_data, tuple) else amps_data
                                current_amps = round(amps, 3)
                                
                                # get_power returns watts (float usually)
                                watts = await battery.get_power()
                                current_watts = round(watts, 2)

                                # LiPo Estimate: 12.6V = 100%, 11.1V = ~20%, 9.0V = 0% typically for 3S
                                # Adjust based on actual battery. Assuming 3S LiPo for Rover.
                                # Simple linear map 9.6 - 12.6
                                pct = (current_volts - 9.6) / (12.6 - 9.6) * 100
                                current_pct = int(max(0, min(100, pct)))
                                
                                last_battery_time = current_time
                            except Exception as e:
                                print(f"Battery read error: {e}")

                        data["battery"] = {
                            "voltage": current_volts,
                            "amps": current_amps,
                            "watts": current_watts,
                            "percent": current_pct
                        }
                        
                        # Include auto-drive state
                        data["is_auto_driving"] = is_auto_driving

                        # --- Video Logic ---
                        # Throttle video sending to save bandwidth for controls
                        if (current_time - last_video_time) < VIDEO_INTERVAL:
                             # Skip video frame, sleep tiny bit and continue
                             await asyncio.sleep(0.005)
                             continue
                             
                        last_video_time = current_time

                        # Note: newer Viam SDK returns ViamImage object, use .data for bytes
                        viam_img = await camera.get_image(mime_type="image/jpeg")
                        img_bytes = viam_img.data
                        
                        # Resize for performance (optional, but good practice if camera is 1080p)
                        # We need to decode to resize anyway if we are detecting.
                        # If NOT detecting, we just pass through.
                        
                        if detection_enabled:
                            frame_count += 1
                            
                            # Decode once
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Resize if huge
                            h, w = frame.shape[:2]
                            if w > 640:
                                frame = cv2.resize(frame, (640, 480))
                                
                            # Run detection only on interval
                            if frame_count % detection_interval == 0:
                                # Run detection on this frame
                                # We need to pass the ARRAY to process_detection now, or refactor process_detection
                                # Let's optimize process_detection to accept an image array instead of bytes if possible, 
                                # or just do it inline here to avoid re-encoding/decoding.
                                
                                if detection_model:
                                    # HYBRID TRACKING LOGIC
                                    # 1. Run YOLO periodically (e.g. every 30 frames) OR if we lost tracking
                                    run_yolo = (frame_count % detection_interval == 0) or (tracker is None)
                                    
                                    if run_yolo:
                                        # Use 320 for speed (Pi 4 Optimization)
                                        results = detection_model(frame, imgsz=320, verbose=False, stream=True)
                                        best_det = None
                                        
                                        # Process YOLO results
                                        for r in results:
                                            for box in r.boxes:
                                                cls_id = int(box.cls[0])
                                                if cls_id not in TARGET_CLASSES: continue
                                                
                                                conf = float(box.conf[0])
                                                if conf < 0.4: continue # Skip weak
                                                
                                                # Select highest confidence object
                                                if best_det is None or conf > best_det['confidence']:
                                                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                                                    width_px, height_px = x2-x1, y2-y1
                                                    label = detection_model.names[cls_id]
                                                    
                                                    # Fix bounding box for tracker init
                                                    # CSRT needs (x, y, w, h)
                                                    best_det = {
                                                        "bbox_tracker": (x1, y1, width_px, height_px),
                                                        "label": label,
                                                        "confidence": conf,
                                                        "bbox_viz": [x1, y1, x2, y2]
                                                    }

                                        if best_det:
                                            # Init Tracker
                                            # Switched to KCF for speed on Pi 4 (Optimization)
                                            tracker = cv2.TrackerKCF_create()
                                            tracker.init(frame, best_det["bbox_tracker"])
                                            tracker_label = best_det["label"]
                                            # We will use the tracker update block below to set 'last_detections'
                                            # to ensure consistency
                                        else:
                                            # No object found by YOLO
                                            # Keep tracker if it was working? No, if YOLO says nothing, 
                                            # and we are in a 'check' frame, maybe we should trust YOLO?
                                            # Actually, YOLO might miss frames. 
                                            # Strategy: Only reset tracker if it explicitly FAILED previous updates
                                            # OR if we want to re-lock. 
                                            # For simplicity: If YOLO finds nothing, we rely on tracker 
                                            # UNLESS tracker also fails.
                                            pass

                                    # 2. Run Tracker (Every Frame where we have a tracker)
                                    current_detections = []
                                    if tracker:
                                        success, box = tracker.update(frame)
                                        if success:
                                            x, y, w, h = [int(v) for v in box]
                                            # center
                                            center_x = int(x + w/2)
                                            center_y = int(y + h/2)
                                            
                                            # Re-calc distance
                                            real_h = KNOWN_HEIGHT_BOTTLE if tracker_label == 'bottle' else KNOWN_HEIGHT_CAN
                                            dist_cm = calculate_distance(h, real_h)
                                            
                                            current_detections.append({
                                                "label": tracker_label,
                                                "confidence": 1.0, # Tracker confidence is binary usually
                                                "distance_cm": round(dist_cm, 1),
                                                "center_x": center_x, 
                                                "center_y": center_y,
                                                "bbox": [x, y, x+w, y+h],
                                                "area_px": w * h
                                            })
                                        else:
                                            # Tracking lost
                                            tracker = None
                                            tracker_label = ""
                                    
                                    last_detections = current_detections
                                        
                            # Draw LAST known detections on CURRENT frame (Tracking effect)
                            for d in last_detections:
                                x1, y1, x2, y2 = d['bbox']
                                label = d['label']
                                # conf = d['confidence'] 
                                dist = d['distance_cm']
                                
                                # Use separate color for tracked vs detected?
                                color = (0, 255, 255) # Cyan for tracker
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.circle(frame, (d['center_x'], d['center_y']), 4, (0, 0, 255), -1)
                                cv2.putText(frame, f"{label} {dist}cm", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Re-encode to send
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            
                            data["image"] = base64.b64encode(buffer).decode('utf-8')
                            data["detections"] = last_detections
                                
                        else:
                            # Send raw image without detection
                            data["image"] = base64.b64encode(img_bytes).decode('utf-8')
                            data["detections"] = []


                        # === AUTO-DRIVE LOGIC (Improved PD Control) ===

                        if is_auto_driving:
                            # 1. Update target state from detections
                            if detection_enabled and data["detections"]:
                                valid_detections = [d for d in data["detections"] if d['confidence'] > 0.4]
                                if valid_detections:
                                    best_detection = max(valid_detections, key=lambda d: d['confidence'])
                                    target_state.update_from_detection(robot_state, best_detection)
                            
                            # Debug output every ~1 second (30 frames at ~30fps)
                            if frame_count % 30 == 0:
                                print(f"=== Debug State ===")
                                print(f"Robot pose: ({robot_state.x:.1f}, {robot_state.y:.1f}) @ {np.degrees(robot_state.theta):.1f}°")
                                if target_state.world_x is not None:
                                    print(f"Target pose: ({target_state.world_x:.1f}, {target_state.world_y:.1f})")
                                    dist, bearing = target_state.get_robot_relative_position(robot_state)
                                    if dist is not None:
                                        print(f"Relative: dist={dist:.1f}cm, bearing={np.degrees(bearing):.1f}°")
                                if data.get("detections"):
                                    det = data["detections"][0]
                                    print(f"Detection: center_x={det['center_x']}px, dist={det['distance_cm']}cm")
                            
                            # 2. Get target relative position (uses memory if no fresh detection)
                            distance, bearing_error = target_state.get_robot_relative_position(robot_state)

                            
                            if distance is not None:
                                time_since_update = time.time() - target_state.last_update_time
                                
                                # Only use state for 2 seconds max
                                if time_since_update < MEMORY_DURATION:
                                    current_time = time.time()
                                    dt = current_time - control_state.last_time
                                    
                                    # === PD CONTROLLER ===
                                    # Control parameters (TUNED for stability)
                                    target_dist = 6.0           # Target stopping distance
                                    dist_threshold = 2.0        # Distance tolerance
                                    
                                    # Distance thresholds for behavior modes
                                    CLOSE_RANGE_CM = 45.0       # 1.5 feet - reduce gains
                                    FINAL_APPROACH_CM = 30.0    # 1 foot - drive straight only
                                    MIN_MOVE = 0.18             # Minimum drive to overcome friction
                                    
                                    # FINAL APPROACH MODE: When very close, disable angular correction
                                    # This prevents veering off when the bottle takes up most of the frame
                                    if distance < FINAL_APPROACH_CM and abs(bearing_error) < 0.3:
                                        # Just drive straight toward target
                                        angular = 0.0
                                        linear = MIN_MOVE if distance > target_dist else 0.0
                                        
                                        # Check if target reached
                                        if distance <= target_dist + dist_threshold:
                                            if time_since_update < 0.5:
                                                print(f"✓ Target reached! Distance: {distance:.1f}cm")
                                                is_auto_driving = False
                                                linear = 0.0
                                        
                                        # Apply motor commands directly
                                        if left_motor and right_motor:
                                            await left_motor.set_power(linear)
                                            await right_motor.set_power(linear)
                                        
                                        # Update control state
                                        control_state.last_time = current_time
                                    else:
                                        # NORMAL PD CONTROL MODE
                                        # Distance-based gain scheduling (reduce gains when close)
                                        if distance < CLOSE_RANGE_CM:
                                            gain_scale = 0.3 + 0.7 * ((distance - target_dist) / (CLOSE_RANGE_CM - target_dist))
                                            gain_scale = np.clip(gain_scale, 0.3, 1.0)
                                        else:
                                            gain_scale = 1.0
                                        
                                        # P gains (scaled by distance)
                                        Kp_angular = 0.8 * gain_scale
                                        Kp_linear = 0.06 * gain_scale
                                        
                                        # D gains (adds damping, scaled by distance)
                                        Kd_angular = 0.4 * gain_scale
                                        Kd_linear = 0.02 * gain_scale
                                        
                                        # Velocity limits (also reduce at close range)
                                        MAX_ANGULAR = 0.40 * gain_scale
                                        MAX_LINEAR = 0.30 * gain_scale
                                        MIN_TURN = 0.20
                                        
                                        # Calculate error derivatives (damping)
                                        if dt > 0:
                                            bearing_rate = (bearing_error - control_state.last_bearing_error) / dt
                                            distance_error = distance - target_dist
                                            distance_rate = (distance_error - control_state.last_distance_error) / dt
                                        else:
                                            bearing_rate = 0.0
                                            distance_rate = 0.0
                                        
                                        # PD control for angular (turn)
                                        angular_raw = (Kp_angular * bearing_error) + (Kd_angular * bearing_rate)
                                        angular_raw = np.clip(angular_raw, -MAX_ANGULAR, MAX_ANGULAR)
                                        
                                        # PD control for linear (distance)
                                        distance_error = distance - target_dist
                                        linear_raw = (Kp_linear * distance_error) + (Kd_linear * distance_rate)
                                        linear_raw = np.clip(linear_raw, -MAX_LINEAR, MAX_LINEAR)
                                        
                                        # Apply smoothing (low-pass filter)
                                        ALPHA = 0.2 if distance < CLOSE_RANGE_CM else 0.3
                                        control_state.smoothed_angular = (ALPHA * angular_raw + 
                                                                          (1-ALPHA) * control_state.smoothed_angular)
                                        control_state.smoothed_linear = (ALPHA * linear_raw + 
                                                                         (1-ALPHA) * control_state.smoothed_linear)
                                        
                                        angular = control_state.smoothed_angular
                                        linear = control_state.smoothed_linear
                                        
                                        # Apply minimum thresholds (deadband compensation)
                                        if abs(angular) > 0.05 and abs(angular) < MIN_TURN:
                                            angular = MIN_TURN * np.sign(angular)
                                        elif abs(angular) <= 0.05:
                                            angular = 0.0
                                            
                                        if abs(linear) > 0.05 and abs(linear) < MIN_MOVE:
                                            linear = MIN_MOVE * np.sign(linear)
                                        elif abs(linear) <= 0.05:
                                            linear = 0.0
                                        
                                        # Check if target reached (normal mode)
                                        if abs(distance - target_dist) < dist_threshold and abs(bearing_error) < 0.1:
                                            if time_since_update < 0.5:  # Fresh detection confirms arrival
                                                print(f"✓ Target reached! Distance: {distance:.1f}cm")
                                                is_auto_driving = False
                                                linear = 0.0
                                                angular = 0.0
                                        
                                        # Apply differential drive mixing
                                        l_pow = np.clip(linear + angular, -1.0, 1.0)
                                        r_pow = np.clip(linear - angular, -1.0, 1.0)
                                        
                                        if left_motor and right_motor:
                                            await left_motor.set_power(l_pow)
                                            await right_motor.set_power(r_pow)
                                        
                                        # Update control state for next iteration
                                        control_state.last_bearing_error = bearing_error
                                        control_state.last_distance_error = distance_error
                                        control_state.last_time = current_time
                                        
                                        # Record trajectory point (every 5th frame to reduce data)
                                        if frame_count % 5 == 0:
                                            trajectory_history.append((robot_state.x, robot_state.y))
                                    
                                else:
                                    # Memory expired - stop and disable auto-drive
                                    print("Lost target - memory expired")
                                    is_auto_driving = False
                                    if left_motor and right_motor:
                                        await left_motor.set_power(0)
                                        await right_motor.set_power(0)
                            else:
                                # No target state - stop
                                if left_motor and right_motor:
                                    await left_motor.set_power(0)
                                    await right_motor.set_power(0)
                            
                    except Exception as e:
                        print(f"Camera/Auto-drive loop error: {e}")
                        pass
                
                # Process lidar data
                if lidar:
                    try:
                        pc_bytes, _ = await lidar.get_point_cloud()
                        points = parse_pcd(pc_bytes)
                        data["lidar_points"] = points
                    except Exception:
                        pass
                
                # Check IMU for confirmation
                if imu:
                    try:
                        # get_readings() returns dict of values
                        readings = await imu.get_readings()
                        # data["imu"] = readings # Optional: send to GUI
                        
                        # Stall Detection Check (Basic)
                        # If motors are powered > 0.5 but linear_accel is low
                        # This requires checking current motor power state
                        pass
                    except Exception:
                        pass

                # Broadcast to all clients
                message = json.dumps(data)
                send_tasks = [client.send(message) for client in connected_clients]
                if send_tasks:
                    await asyncio.gather(*send_tasks)
                    
            except Exception as e:
                print(f"Producer error: {e}")
        
        # Rate limit: 10Hz
        # Rate limit: ~60Hz (limited by camera speed roughly)
        await asyncio.sleep(0.01)


async def consumer_task(websocket):
    """Handle incoming messages from a WebSocket client."""
    global robot, left_motor, right_motor, detection_enabled, is_auto_driving
    
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if not robot:
                continue
            
            if msg_type == "set_power":
                motor = data.get("motor")
                power = float(data.get("power", 0.0))
                
                if motor == "left":
                    await left_motor.set_power(power)
                elif motor == "right":
                    await right_motor.set_power(power)
            
            elif msg_type == "stop":
                await left_motor.set_power(0)
                await right_motor.set_power(0)
            
            elif msg_type == "toggle_detection":
                detection_enabled = data.get("enabled", False)
                print(f"Detection {'enabled' if detection_enabled else 'disabled'}")

            elif msg_type == "start_auto_drive":
                 print("Starting Auto-Drive...")
                 is_auto_driving = True
                 detection_enabled = True # Force detection on
                 # Reset trajectory and record start position
                 trajectory_history.clear()
                 auto_drive_start_pos = {"x": robot_state.x, "y": robot_state.y}
                 trajectory_history.append((robot_state.x, robot_state.y))
            
            elif msg_type == "stop_auto_drive":
                 print("Stopping Auto-Drive...")
                 is_auto_driving = False
                 await left_motor.set_power(0)
                 await right_motor.set_power(0)
            
            elif msg_type == "disconnect":
                 print("Client requested disconnect - stopping motors")
                 is_auto_driving = False
                 detection_enabled = False
                 await left_motor.set_power(0)
                 await right_motor.set_power(0)
                 await websocket.close()
                
        except Exception as e:
            print(f"Consumer error: {e}")


async def handler(websocket):
    """Handle a new WebSocket connection."""
    global connected_clients
    
    print(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    
    try:
        await consumer_task(websocket)
    finally:
        print(f"Client disconnected: {websocket.remote_address}")
        connected_clients.remove(websocket)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Initialize and run the server."""
    
    # Load detection model
    initialize_detection_model()
    
    # Connect to robot
    if not await connect_to_robot():
        return
    
    # Start producer task
    asyncio.create_task(producer_task())
    
    # Start WebSocket server
    # Listen on 0.0.0.0 to allow access from other machines (e.g. Laptop -> Pi)
    server_address = "0.0.0.0"
    server_port = 8081
    
    print(f"\n{'=' * 50}")
    print(f"WebSocket server running on ws://{server_address}:{server_port}")
    print(f"{'=' * 50}\n")
    
    async with websockets.serve(handler, server_address, server_port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())