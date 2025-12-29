"""
Viam Rover Control Server - RASPBERRY PI 5 OPTIMIZED

This WebSocket server connects to a Viam-powered rover and provides:
- Motor control (left/right)
- Camera feed streaming
- Lidar data streaming
- Real-time bottle/can detection using YOLOv8

Optimized for Raspberry Pi 5 (CPU-only inference).
For Jetson Orin (GPU), use server_jetson.py instead.

Usage:
    python server_pi.py          # Normal mode (connects to robot)
    python server_pi.py --sim    # Simulation mode (no hardware needed)
"""

import asyncio
import time
import json
import signal
import argparse
import websockets
from navigation_fsm import NavigationFSM, NavigationConfig, NavigationState
import base64
import io
import struct
import numpy as np
import cv2
from ultralytics import YOLO

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Viam Rover Control Server')
parser.add_argument('--sim', action='store_true', help='Run in simulation mode (no hardware)')
parser.add_argument('--profile', action='store_true', help='Enable performance profiling (sends metrics to GUI)')
args = parser.parse_args()
SIM_MODE = args.sim
PROFILE_MODE = args.profile

# Conditional Viam imports (not needed in sim mode)
if not SIM_MODE:
    from viam.robot.client import RobotClient
    from viam.rpc.dial import DialOptions
    from viam.components.motor import Motor
    from viam.components.camera import Camera
    from viam.components.sensor import Sensor
    from viam.components.movement_sensor import MovementSensor
    from viam.components.power_sensor import PowerSensor
    from viam.components.encoder import Encoder

# =============================================================================
# CONFIGURATION
# =============================================================================

# Robot Connection Details (Pi 5)
ROBOT_ADDRESS = "pi5-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "bd131b00-768a-4147-839a-4dae24169224"
API_KEY = "0eg1lzk5jg3x2c3cjh3i17aoljf76989"

# Component Names (Pi 5 configuration)
LEFT_MOTOR_NAME = "left"
RIGHT_MOTOR_NAME = "right"
CAMERA_NAME = "cam"
LIDAR_NAME = "lidar"  # RPLIDAR A1 - set to None if not connected
BATTERY_NAME = "ina219"
IMU_NAME = "imu"
LEFT_ENCODER_NAME = "left-enc"   # Pi 5 encoder name
RIGHT_ENCODER_NAME = "right-enc" # Pi 5 encoder name

# Lidar Configuration (for obstacle avoidance and distance fusion)
LIDAR_READ_INTERVAL = 0.5       # Read lidar at 2Hz to save API calls
LIDAR_FORWARD_ARC_DEG = 60      # Forward arc to check for obstacles (±30°)
LIDAR_OBSTACLE_DIST_CM = 25     # Trigger avoidance if obstacle closer than this

# Detection Configuration
KNOWN_HEIGHT_BOTTLE = 20.0  # Standard water bottle height in cm
KNOWN_HEIGHT_CAN = 12.0     # Standard soda can height in cm
FOCAL_LENGTH = 600          # Webcam focal length (calibrate for accuracy)
TARGET_CLASSES = [0]        # Custom model: 0=can (for yolo11n_cans.pt)

# Camera Calibration (HYPER hypercam HD 1080 = 76.5° HFOV)
CAMERA_HFOV_DEG = 76.5      # Horizontal field of view in degrees
IMAGE_WIDTH = 640           # Camera resolution width
IMAGE_HEIGHT = 480          # Camera resolution height

# Wheel Parameters (CALIBRATE THESE! - Measured for your rover)
WHEEL_DIAMETER_CM = 5.5     # Wheel diameter in cm
WHEEL_BASE_CM = 19.55       # Distance between wheel contact points in cm

# Camera Mount Parameters (for Homography distance estimation)
CAMERA_HEIGHT_CM = 12.0     # Height of camera lens above ground (cm)
CAMERA_TILT_DEG = 15.0      # Camera tilt angle (degrees, positive = looking down)

# =============================================================================
# RASPBERRY PI 5 OPTIMIZATION SETTINGS
# =============================================================================
# These settings are tuned for CPU-only inference on Pi 5

# YOLO Model - use custom trained can detection model
YOLO_MODEL = 'yolo11n_cans.pt'   # YOLO11 - trained on combined can datasets

# Inference resolution - lower = faster
INFERENCE_SIZE = 320        # 320px for Pi 5 (GPU: 640)

# Detection interval - run YOLO every N frames (reduces CPU load)
DETECTION_INTERVAL = 3      # Every 3rd frame (GPU: 1)

# Video frame rate cap (reduces bandwidth and CPU)
VIDEO_FPS_CAP = 15          # 15 FPS for Pi 5 (GPU: 24-30)

# JPEG quality (lower = smaller files, faster transfer)
JPEG_QUALITY = 65           # 65% for Pi 5 (GPU: 70-85)

# =============================================================================
# TIMEOUT AND RECONNECTION SETTINGS
# =============================================================================
# Increased timeouts for network latency tolerance
MOTOR_CMD_TIMEOUT = 2.5     # Timeout for motor set_power (seconds)
API_TIMEOUT = 3.0           # Timeout for encoder/camera API calls (seconds)

# Auto-reconnection settings
MAX_CONSECUTIVE_TIMEOUTS = 5    # Reconnect after this many consecutive timeouts
RECONNECT_DELAY = 2.0           # Delay before reconnection attempt (seconds)

# =============================================================================
# PERFORMANCE PROFILING
# =============================================================================

class PerformanceMetrics:
    """
    Tracks API call latencies and performance metrics.
    Enabled with --profile flag, sends data over WebSocket to GUI.
    """
    
    def __init__(self):
        self.enabled = PROFILE_MODE
        self.start_time = time.time()
        
        # Timing collections (last N samples)
        self.max_samples = 100
        self.timings = {
            "motor_read": [],
            "encoder_read": [],
            "camera_read": [],
            "lidar_read": [],
            "detection": [],
            "websocket_send": [],
            "frame_total": []
        }
        
        # Counters
        self.timeout_count = 0
        self.frame_count = 0
        self.api_call_count = 0
        
        # Summary stats (updated every N frames)
        self.summary_interval = 50
        self.last_summary = {}
    
    def record(self, category: str, duration_sec: float):
        """Record a timing sample."""
        if not self.enabled:
            return
        
        samples = self.timings.get(category, [])
        samples.append(duration_sec * 1000)  # Store in milliseconds
        if len(samples) > self.max_samples:
            samples.pop(0)
        self.timings[category] = samples
        self.api_call_count += 1
    
    def record_timeout(self):
        """Record a timeout event."""
        self.timeout_count += 1
    
    def get_summary(self) -> dict:
        """Get summary statistics for all metrics."""
        summary = {
            "uptime_sec": round(time.time() - self.start_time, 1),
            "frame_count": self.frame_count,
            "api_call_count": self.api_call_count,
            "timeout_count": self.timeout_count,
            "categories": {}
        }
        
        for category, samples in self.timings.items():
            if samples:
                summary["categories"][category] = {
                    "avg_ms": round(sum(samples) / len(samples), 2),
                    "max_ms": round(max(samples), 2),
                    "min_ms": round(min(samples), 2),
                    "count": len(samples)
                }
        
        self.last_summary = summary
        return summary
    
    def should_send_summary(self) -> bool:
        """Check if it's time to send a summary."""
        return self.enabled and (self.frame_count % self.summary_interval == 0)


# Global profiler instance
perf = PerformanceMetrics()


# =============================================================================
# SIMULATION MODE - Mock Classes and Ghost Target
# =============================================================================

# Ghost Target for testing navigation in simulation
GHOST_TARGET = {
    "x": 100.0,      # cm from start
    "y": 50.0,       # cm from start
    "label": "bottle",
    "height_cm": KNOWN_HEIGHT_BOTTLE
}


class MockMotor:
    """Simulated motor for testing without hardware."""
    def __init__(self, name: str):
        self.name = name
        self.power = 0.0
        self._position = 0.0  # Encoder ticks (simulated)
    
    async def set_power(self, power: float):
        self.power = max(-1.0, min(1.0, power))
    
    async def get_position(self):
        return self._position
    
    async def is_powered(self):
        return (True, self.power)
    
    async def stop(self):
        self.power = 0.0
    
    def update_position(self, dt: float, ticks_per_second: float = 100.0):
        """Called by physics loop to update encoder position."""
        self._position += self.power * ticks_per_second * dt


class MockEncoder:
    """Simulated encoder for testing without hardware."""
    def __init__(self, motor: MockMotor):
        self.motor = motor
    
    async def get_position(self):
        return self.motor._position


class MockCamera:
    """Simulated camera that generates test frames."""
    def __init__(self):
        self.frame_count = 0
    
    async def get_image(self):
        """Return a test pattern image."""
        # Create a simple test frame with grid pattern
        frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        
        # Draw grid
        for i in range(0, IMAGE_WIDTH, 40):
            cv2.line(frame, (i, 0), (i, IMAGE_HEIGHT), (50, 50, 50), 1)
        for i in range(0, IMAGE_HEIGHT, 40):
            cv2.line(frame, (0, i), (IMAGE_WIDTH, i), (50, 50, 50), 1)
        
        # Add "SIM MODE" text
        cv2.putText(frame, "SIM MODE", (IMAGE_WIDTH//2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        self.frame_count += 1
        
        # Return as PIL-like object with tobytes method
        class FakeImage:
            def __init__(self, data):
                self._data = data
            def tobytes(self):
                return self._data.tobytes()
        
        return FakeImage(frame)


# Physics simulation state
sim_physics_running = False

async def sim_physics_loop():
    """Background task that simulates robot physics at 50Hz."""
    global sim_physics_running, left_motor, right_motor, robot_state
    
    sim_physics_running = True
    last_time = time.time()
    
    while sim_physics_running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        if left_motor and right_motor and hasattr(left_motor, 'update_position'):
            # Update encoder positions based on motor power
            left_motor.update_position(dt)
            right_motor.update_position(dt)
            
            # Differential drive kinematics
            # Convert motor power to velocity (cm/s)
            max_speed_cm_s = 30.0  # Max robot speed
            v_left = left_motor.power * max_speed_cm_s
            v_right = right_motor.power * max_speed_cm_s
            
            # Calculate linear and angular velocity
            v_linear = (v_left + v_right) / 2.0
            v_angular = (v_right - v_left) / WHEEL_BASE_CM
            
            # Update robot pose
            robot_state.theta += v_angular * dt
            robot_state.x += v_linear * np.cos(robot_state.theta) * dt
            robot_state.y += v_linear * np.sin(robot_state.theta) * dt
        
        await asyncio.sleep(0.02)  # 50Hz


def generate_ghost_detection(robot_state) -> dict:
    """Generate fake detection data for the ghost target."""
    # Calculate relative position of ghost target
    dx = GHOST_TARGET["x"] - robot_state.x
    dy = GHOST_TARGET["y"] - robot_state.y
    
    # Distance to target
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Bearing to target (relative to robot heading)
    target_angle = np.arctan2(dy, dx)
    bearing = target_angle - robot_state.theta
    
    # Normalize bearing to [-pi, pi]
    while bearing > np.pi: bearing -= 2*np.pi
    while bearing < -np.pi: bearing += 2*np.pi
    
    # Check if target is in camera FOV
    fov_rad = CAMERA_HFOV_DEG * np.pi / 180.0
    if abs(bearing) > fov_rad / 2:
        return None  # Target not visible
    
    # Check if target is within reasonable range
    if distance > 300 or distance < 10:
        return None
    
    # Calculate pixel position in frame
    center_x = IMAGE_WIDTH / 2 + (bearing / (fov_rad / 2)) * (IMAGE_WIDTH / 2)
    
    # Estimate bounding box size based on distance
    apparent_height = (GHOST_TARGET["height_cm"] * FOCAL_LENGTH) / distance
    bbox_height = int(apparent_height)
    bbox_width = int(bbox_height * 0.4)  # Bottle aspect ratio
    
    center_y = IMAGE_HEIGHT / 2 + 50  # Slightly below center
    
    return {
        "label": GHOST_TARGET["label"],
        "confidence": 0.95,
        "center_x": int(center_x),
        "center_y": int(center_y),
        "width": bbox_width,
        "height": bbox_height,
        "distance_cm": distance
    }


# =============================================================================
# GLOBAL STATE
# =============================================================================

robot = None
left_motor = None
right_motor = None
camera = None
lidar = None
battery = None
imu = None
left_encoder = None
right_encoder = None

# Timeout tracking for auto-reconnection
consecutive_timeout_count = 0
is_reconnecting = False

# Motor command coalescing (prevents hitting Viam 100-request limit)
# Only send motor commands if power value changed or interval elapsed
last_motor_power = {"left": None, "right": None}
last_motor_time = {"left": 0.0, "right": 0.0}
MOTOR_CMD_MIN_INTERVAL = 0.05  # 50ms = 20Hz max per motor
MOTOR_POWER_DEADBAND = 0.02   # Ignore changes smaller than 2%

# Auto-drive state
is_auto_driving: bool = False
target_distance_cm: float = 15.0  # Target stopping distance from object
dist_threshold_cm: float = 3.0    # Distance tolerance for "arrived"

# Navigation State Machine for Path Planning
class NavPhase:
    IDLE = "IDLE"
    ACQUIRE = "ACQUIRE"    # Stop and detect target multiple times
    ROTATE = "ROTATE"      # Turn toward target bearing
    DRIVE = "DRIVE"        # Drive straight to target
    ARRIVED = "ARRIVED"    # At target

nav_phase: str = NavPhase.IDLE
nav_target_distance: float = 0.0      # Acquired target distance (cm)
nav_target_bearing: float = 0.0       # Acquired target bearing (radians from center)
nav_acquire_samples: list = []        # List of (distance, bearing) samples
NAV_ACQUIRE_COUNT = 3                 # Number of samples to average
NAV_BEARING_THRESHOLD = 0.12          # ~7 degrees - aligned enough to drive
NAV_BEARING_HYSTERESIS = 0.08         # ~4.5 degrees - hysteresis to prevent oscillation
NAV_LARGE_TURN_THRESHOLD = 0.35       # ~20 degrees - use tank turn above this
NAV_ROTATE_SPEED = 0.22               # Motor power for tank rotation (both wheels)
NAV_PIVOT_SPEED = 0.20                # Motor power for pivot turn (one wheel only)
NAV_DRIVE_SPEED = 0.22                # Motor power for driving
nav_last_turn_dir: int = 0            # Remember last turn direction to prevent oscillation
  
# FPS Optimization
frame_count: int = 0
detection_interval: int = DETECTION_INTERVAL  # Use config constant
last_detections: list = []

# FPS Tracking for performance monitoring
fps_camera: float = 0.0           # Camera frame rate
fps_detection: float = 0.0        # YOLO detection rate
last_fps_update_time: float = 0.0 # Last time FPS was calculated
fps_frame_count: int = 0          # Frames since last FPS calc
fps_detection_count: int = 0      # Detections since last FPS calc

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

# Navigation FSM instance (initialized after robot connects)
nav_fsm: NavigationFSM = None

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
# LIDAR FUNCTIONS
# =============================================================================

# Cache for lidar scan data
cached_lidar_distances = []
cached_lidar_angles = []


async def get_lidar_scan_data():
    """
    Get lidar scan and extract useful data.
    
    Returns:
        tuple: (min_forward_distance_cm, distances_array, angles_array)
               Returns (None, [], []) if lidar unavailable
    """
    global lidar, cached_lidar_distances, cached_lidar_angles
    
    if not lidar:
        return None, [], []
    
    try:
        # Get point cloud from lidar (Viam returns PCD format)
        pcd_bytes, _ = await asyncio.wait_for(
            lidar.get_point_cloud(),
            timeout=API_TIMEOUT
        )
        
        # Parse PCD binary data
        distances, angles = parse_pcd_to_polar(pcd_bytes)
        cached_lidar_distances = distances
        cached_lidar_angles = angles
        
        # Find minimum distance in forward arc (±LIDAR_FORWARD_ARC_DEG/2)
        half_arc = np.radians(LIDAR_FORWARD_ARC_DEG / 2)
        min_forward_dist = float('inf')
        
        for dist, angle in zip(distances, angles):
            # Normalize angle to [-pi, pi] where 0 is forward
            if -half_arc <= angle <= half_arc:
                if dist > 0 and dist < min_forward_dist:
                    min_forward_dist = dist
        
        # Convert to cm and return
        min_dist_cm = min_forward_dist * 100 if min_forward_dist != float('inf') else None
        reset_timeout_counter()  # Success
        
        return min_dist_cm, distances, angles
        
    except asyncio.TimeoutError:
        await handle_api_timeout("Lidar read")
        return None, cached_lidar_distances, cached_lidar_angles
    except Exception as e:
        print(f"Lidar error: {e}")
        return None, [], []


def parse_pcd_to_polar(pcd_bytes: bytes):
    """
    Parse PCD binary data to polar coordinates.
    
    Returns:
        tuple: (distances in meters, angles in radians)
    """
    distances = []
    angles = []
    
    try:
        # Skip PCD header (find DATA binary or ascii)
        data_start = pcd_bytes.find(b'DATA binary')
        if data_start == -1:
            data_start = pcd_bytes.find(b'DATA ascii')
            if data_start == -1:
                return [], []
        
        # Find newline after DATA line
        newline_pos = pcd_bytes.find(b'\n', data_start)
        if newline_pos == -1:
            return [], []
        
        # Binary data starts after newline
        binary_data = pcd_bytes[newline_pos + 1:]
        
        # Assume XYZ float32 format (12 bytes per point)
        point_size = 12
        num_points = len(binary_data) // point_size
        
        for i in range(num_points):
            offset = i * point_size
            x = np.frombuffer(binary_data[offset:offset+4], dtype=np.float32)[0]
            y = np.frombuffer(binary_data[offset+4:offset+8], dtype=np.float32)[0]
            # z is height, we mainly care about x,y for 2D navigation
            
            # Convert to polar
            dist = np.sqrt(x*x + y*y)
            angle = np.arctan2(y, x)  # Angle from robot's X axis
            
            if dist > 0.01:  # Filter out near-zero readings
                distances.append(dist)
                angles.append(angle)
                
    except Exception as e:
        print(f"PCD parse error: {e}")
    
    return distances, angles


def get_lidar_distance_at_angle(target_angle_rad: float, tolerance_rad: float = 0.1):
    """
    Get lidar distance at a specific angle (for fusing with camera detection).
    
    Args:
        target_angle_rad: Angle to check (0 = forward, positive = left)
        tolerance_rad: Angular tolerance (~6 degrees default)
    
    Returns:
        float: Average distance in cm at that angle, or None if no data
    """
    global cached_lidar_distances, cached_lidar_angles
    
    matching_dists = []
    for dist, angle in zip(cached_lidar_distances, cached_lidar_angles):
        if abs(angle - target_angle_rad) <= tolerance_rad:
            matching_dists.append(dist * 100)  # Convert to cm
    
    if not matching_dists:
        return None
    
    return sum(matching_dists) / len(matching_dists)


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def initialize_detection_model():
    """Load the YOLO model for object detection."""
    global detection_model
    try:
        detection_model = YOLO(YOLO_MODEL)  # Use config constant
        print(f"✓ YOLO detection model ({YOLO_MODEL}) loaded successfully.")
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


def calculate_distance_homography(bbox_bottom_y: int, image_height: int = IMAGE_HEIGHT) -> float:
    """
    Calculate distance using ground plane projection (Homography).
    
    Uses the bottom of the bounding box (assumed to be on the ground) to estimate
    distance more accurately than the pinhole model at close range.
    
    Formula: D = H_cam / tan(θ_cam + α_pixel)
    
    Args:
        bbox_bottom_y: Y coordinate of bottom of bounding box (pixels from top)
        image_height: Total image height in pixels
        
    Returns:
        Estimated distance in centimeters
    """
    # Calculate vertical FOV from horizontal FOV and aspect ratio
    aspect_ratio = image_height / IMAGE_WIDTH
    vfov_deg = CAMERA_HFOV_DEG * aspect_ratio
    
    # Pixel offset from image center (positive = below center = closer)
    image_center_y = image_height / 2
    pixel_offset = bbox_bottom_y - image_center_y
    
    # Convert pixel offset to angle (radians)
    # Each pixel represents vfov_deg / image_height degrees
    degrees_per_pixel = vfov_deg / image_height
    alpha_pixel_deg = pixel_offset * degrees_per_pixel
    alpha_pixel_rad = alpha_pixel_deg * (np.pi / 180.0)
    
    # Camera tilt in radians (positive = looking down)
    theta_cam_rad = CAMERA_TILT_DEG * (np.pi / 180.0)
    
    # Total angle from horizontal to the ground point
    total_angle = theta_cam_rad + alpha_pixel_rad
    
    # Check for invalid geometry (object above horizon or angle too shallow)
    if total_angle <= 0.01:  # ~0.5 degree minimum
        return 999.0  # Return large distance for objects above horizon
    
    # Ground plane projection
    distance = CAMERA_HEIGHT_CM / np.tan(total_angle)
    
    # Clamp to reasonable range
    return max(5.0, min(distance, 500.0))


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
    """Establish connection to the Viam robot (or create mock in SIM_MODE)."""
    global robot, left_motor, right_motor, camera, lidar, battery, imu
    global left_encoder, right_encoder
    
    # =========================================================================
    # SIMULATION MODE - Create mock components
    # =========================================================================
    if SIM_MODE:
        print("\n" + "="*50)
        print("  🎮 SIMULATION MODE - No Hardware Required")
        print("="*50 + "\n")
        
        robot = "SIMULATED"  # Dummy value to indicate connected
        
        # Create mock motors
        left_motor = MockMotor("left")
        right_motor = MockMotor("right")
        print("✓ Mock Motors created")
        
        # Create mock encoders (linked to motors)
        left_encoder = MockEncoder(left_motor)
        right_encoder = MockEncoder(right_motor)
        print("✓ Mock Encoders created")
        
        # Create mock camera
        camera = MockCamera()
        print("✓ Mock Camera created")
        
        # No lidar, battery, or IMU in sim mode
        lidar = None
        battery = None
        imu = None
        
        print(f"\n🎯 Ghost Target at ({GHOST_TARGET['x']}, {GHOST_TARGET['y']}) cm")
        print("   Robot will try to navigate to it when auto-drive is enabled\n")
        
        return True
    
    # =========================================================================
    # REAL MODE - Connect to Viam robot
    # =========================================================================
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
            imu = MovementSensor.from_robot(robot, IMU_NAME)
            print("✓ IMU initialized (MovementSensor)")
        except Exception as e:
            print(f"✗ IMU '{IMU_NAME}' not found: {e}")
            imu = None
        
        # Initialize encoders for odometry
        try:
            left_encoder = Encoder.from_robot(robot, LEFT_ENCODER_NAME)
            right_encoder = Encoder.from_robot(robot, RIGHT_ENCODER_NAME)
            print(f"✓ Encoders initialized ({LEFT_ENCODER_NAME}, {RIGHT_ENCODER_NAME})")
        except Exception:
            print("✗ Encoders not found (odometry disabled)")
            left_encoder = None
            right_encoder = None
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        robot = None
        return False


async def reconnect_to_robot():
    """
    Attempt to reconnect to the robot after consecutive timeouts.
    Closes existing connection and re-establishes it.
    """
    global robot, is_reconnecting, consecutive_timeout_count
    
    if is_reconnecting or SIM_MODE:
        return False
    
    is_reconnecting = True
    consecutive_timeout_count = 0
    
    print("\n" + "="*50)
    print("  🔄 AUTO-RECONNECTING TO ROBOT")
    print("="*50 + "\n")
    
    # Close existing connection
    try:
        if robot and robot != "SIMULATED":
            await robot.close()
            print("✓ Closed existing connection")
    except Exception as e:
        print(f"⚠ Error closing connection: {e}")
    
    # Wait before reconnecting
    await asyncio.sleep(RECONNECT_DELAY)
    
    # Attempt reconnection
    success = await connect_to_robot()
    
    if success:
        # Update NavigationFSM with new motor references
        if nav_fsm and left_motor and right_motor:
            nav_fsm.update_motors(left_motor, right_motor)
        print("✓ Reconnection successful!\n")
    else:
        print("✗ Reconnection failed - will retry on next timeout\n")
    
    is_reconnecting = False
    return success


async def handle_api_timeout(operation_name: str = "API call"):
    """
    Called when an API timeout occurs. Tracks consecutive timeouts
    and triggers reconnection if threshold exceeded.
    """
    global consecutive_timeout_count
    
    consecutive_timeout_count += 1
    print(f"⚠ {operation_name} timeout ({consecutive_timeout_count}/{MAX_CONSECUTIVE_TIMEOUTS})")
    
    if consecutive_timeout_count >= MAX_CONSECUTIVE_TIMEOUTS:
        print(f"⚠ Max consecutive timeouts reached - triggering reconnection")
        await reconnect_to_robot()


def reset_timeout_counter():
    """Called on successful API call to reset the timeout counter."""
    global consecutive_timeout_count
    if consecutive_timeout_count > 0:
        consecutive_timeout_count = 0


def should_send_motor_command(motor: str, power: float) -> bool:
    """
    Check if motor command should actually be sent (coalescing).
    Returns True if:
    - Power value changed beyond deadband, OR
    - Minimum interval elapsed since last command
    
    This prevents flooding the Viam API with redundant commands.
    """
    global last_motor_power, last_motor_time
    
    now = time.time()
    last_power = last_motor_power.get(motor)
    last_time = last_motor_time.get(motor, 0)
    
    # Always send if this is the first command or a stop (power=0)
    if last_power is None or power == 0:
        return True
    
    # Check if power changed beyond deadband
    power_changed = abs(power - last_power) > MOTOR_POWER_DEADBAND
    
    # Check if minimum interval elapsed
    interval_elapsed = (now - last_time) >= MOTOR_CMD_MIN_INTERVAL
    
    return power_changed or interval_elapsed


def update_motor_state(motor: str, power: float):
    """Update the last sent motor power and time."""
    global last_motor_power, last_motor_time
    last_motor_power[motor] = power
    last_motor_time[motor] = time.time()


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
    global fps_camera, fps_detection, last_fps_update_time, fps_frame_count, fps_detection_count
    global nav_phase, nav_acquire_samples, nav_target_distance, nav_target_bearing, nav_last_turn_dir
    
    last_video_time = 0
    VIDEO_INTERVAL = 1.0 / VIDEO_FPS_CAP  # Use config constant for Pi 5 optimization
    last_battery_time = 0
    BATTERY_INTERVAL = 5.0       # Update battery every 5s
    last_encoder_time = 0
    ENCODER_INTERVAL = 0.2       # Update encoders at 5Hz (was 10Hz) - reduces API calls
    last_motor_read_time = 0
    MOTOR_READ_INTERVAL = 1.0    # Update motor status every 1.0s (1Hz) - critical to avoid API overflow
    last_lidar_time = 0
    
    API_TIMEOUT = 2.0  # Timeout for Viam API calls to prevent freezes
    
    current_volts = 0.0
    current_amps = 0.0
    current_watts = 0.0
    current_pct = 0.0
    
    # Cached lidar data (for obstacle avoidance)
    cached_min_forward_dist_cm = None
    # Cached motor data (updated less frequently to reduce API load)
    cached_left_pos = 0.0
    cached_left_power = 0.0
    cached_right_pos = 0.0
    cached_right_power = 0.0
    
    # Odometry-based memory (2.5D visual servoing)
    MEMORY_DURATION = 2.0  # Use target memory for 2 seconds max
    
    while True:
        current_time = time.time()
        if robot and connected_clients:
            try:
                # Gather motor data with timeout (SKIP during auto-drive to save API calls)
                # During auto-drive, encoders provide position data for odometry
                if not is_auto_driving and current_time - last_motor_read_time > MOTOR_READ_INTERVAL:
                    try:
                        _t0 = time.time()
                        left_pos, left_power_data, right_pos, right_power_data = await asyncio.wait_for(
                            asyncio.gather(
                                left_motor.get_position(),
                                left_motor.is_powered(),
                                right_motor.get_position(),
                                right_motor.is_powered()
                            ),
                            timeout=API_TIMEOUT
                        )
                        perf.record("motor_read", time.time() - _t0)
                        cached_left_pos = left_pos
                        cached_left_power = left_power_data[1]
                        cached_right_pos = right_pos
                        cached_right_power = right_power_data[1]
                        last_motor_read_time = current_time
                        reset_timeout_counter()  # Successful API call
                    except asyncio.TimeoutError:
                        await handle_api_timeout("Motor read")
                    except Exception as e:
                        print(f"Motor read error: {e}")
                
                # Build response data (using cached values for reliability)
                data = {
                    "type": "readout",
                    "left_pos": cached_left_pos,
                    "left_power": cached_left_power,
                    "right_pos": cached_right_pos,
                    "right_power": cached_right_power,
                    "detection_enabled": detection_enabled,
                    "is_auto_driving": is_auto_driving,
                    "nav_phase": nav_fsm.state_summary if nav_fsm else nav_phase,  # Navigation state for GUI
                    # FPS data for GUI display
                    "fps_camera": round(fps_camera, 1),
                    "fps_detection": round(fps_detection, 1) if detection_enabled else 0,
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
                
                # Update odometry from wheel encoders (ONLY during auto-drive to save API calls)
                # Manual control doesn't need odometry - user drives by sight
                if is_auto_driving and left_encoder and right_encoder:
                    try:
                        _t0 = time.time()
                        left_enc_pos, _ = await left_encoder.get_position()
                        right_enc_pos, _ = await right_encoder.get_position()
                        perf.record("encoder_read", time.time() - _t0)
                        robot_state.update_odometry(left_enc_pos, right_enc_pos,
                                                    wheel_base_cm=WHEEL_BASE_CM,
                                                    wheel_diameter_cm=WHEEL_DIAMETER_CM)
                        
                        # Fuse IMU heading to correct encoder drift
                        await update_robot_state_with_imu()
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
                            fps_frame_count += 1
                            
                            # Calculate FPS every second
                            if current_time - last_fps_update_time >= 1.0:
                                fps_camera = fps_frame_count / (current_time - last_fps_update_time)
                                fps_detection = fps_detection_count / (current_time - last_fps_update_time)
                                fps_frame_count = 0
                                fps_detection_count = 0
                                last_fps_update_time = current_time
                            
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
                                        # Use INFERENCE_SIZE for Pi 5 performance
                                        results = detection_model(frame, imgsz=INFERENCE_SIZE, verbose=False, stream=True)
                                        best_det = None
                                        fps_detection_count += 1  # Count YOLO inference for FPS
                                        
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
                                            # Init Tracker - try multiple methods for opencv compatibility
                                            try:
                                                # Try legacy module (opencv-contrib-python)
                                                if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
                                                    tracker = cv2.legacy.TrackerKCF_create()
                                                elif hasattr(cv2, 'TrackerKCF_create'):
                                                    tracker = cv2.TrackerKCF_create()
                                                else:
                                                    # No tracker available - use YOLO-only mode
                                                    tracker = None
                                                    
                                                if tracker:
                                                    tracker.init(frame, best_det["bbox_tracker"])
                                                    tracker_label = best_det["label"]
                                            except Exception as e:
                                                print(f"Tracker init failed (using YOLO-only): {e}")
                                                tracker = None
                                            # We will use the tracker update block below to set 'last_detections'
                                            # to ensure consistency
                                        else:
                                            # No object found by YOLO
                                            pass

                                    # 2. Run Tracker (Every Frame where we have a tracker) OR use YOLO results directly
                                    current_detections = []
                                    if tracker:
                                        success, box = tracker.update(frame)
                                        if success:
                                            x, y, w, h = [int(v) for v in box]
                                            center_x = int(x + w/2)
                                            center_y = int(y + h/2)
                                            
                                            real_h = KNOWN_HEIGHT_BOTTLE if tracker_label == 'bottle' else KNOWN_HEIGHT_CAN
                                            dist_cm = calculate_distance(h, real_h)
                                            
                                            current_detections.append({
                                                "label": tracker_label,
                                                "confidence": 1.0,
                                                "distance_cm": round(dist_cm, 1),
                                                "center_x": center_x, 
                                                "center_y": center_y,
                                                "bbox": [x, y, x+w, y+h],
                                                "area_px": w * h
                                            })
                                        else:
                                            tracker = None
                                            tracker_label = ""
                                    elif best_det:
                                        # No tracker available - use YOLO detection directly
                                        x1, y1, w_px, h_px = best_det["bbox_tracker"]
                                        center_x = int(x1 + w_px/2)
                                        center_y = int(y1 + h_px/2)
                                        
                                        real_h = KNOWN_HEIGHT_BOTTLE if best_det["label"] == 'bottle' else KNOWN_HEIGHT_CAN
                                        dist_cm = calculate_distance(h_px, real_h)
                                        
                                        current_detections.append({
                                            "label": best_det["label"],
                                            "confidence": best_det["confidence"],
                                            "distance_cm": round(dist_cm, 1),
                                            "center_x": center_x,
                                            "center_y": center_y,
                                            "bbox": best_det["bbox_viz"],
                                            "area_px": w_px * h_px
                                        })
                                    
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
                            
                            # Re-encode to send with Pi 5 optimized quality
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                            
                            data["image"] = base64.b64encode(buffer).decode('utf-8')
                            data["detections"] = last_detections
                                
                        else:
                            # Send raw image without detection
                            data["image"] = base64.b64encode(img_bytes).decode('utf-8')
                            data["detections"] = []


                        # === AUTO-DRIVE LOGIC (using Navigation FSM) ===
                        if is_auto_driving and nav_fsm:
                            # Get detection for FSM
                            det = data["detections"][0] if data.get("detections") else None
                            
                            # Debug output every ~1 second
                            if frame_count % 30 == 0:
                                print(f"=== Nav State: {nav_fsm.state_summary} ===")
                                if det:
                                    print(f"  Detection: dist={det['distance_cm']:.1f}cm, center_x={det['center_x']}px")
                            
                            # Update Navigation FSM (handles all navigation logic)
                            # Pass lidar data for obstacle avoidance
                            await nav_fsm.update(detection=det, lidar_min_distance_cm=cached_min_forward_dist_cm)
                            
                            # Check if FSM reached ARRIVED state
                            if nav_fsm.state == NavigationState.ARRIVED:
                                is_auto_driving = False
                                print("🏁 Navigation complete!")
                            
                    except Exception as e:
                        print(f"Camera/Auto-drive loop error: {e}")
                        pass
                
                # Process lidar data (rate-limited to save API calls)
                if lidar and is_auto_driving and (current_time - last_lidar_time > LIDAR_READ_INTERVAL):
                    try:
                        cached_min_forward_dist_cm, distances, angles = await get_lidar_scan_data()
                        
                        # Also include in data for GUI visualization if needed
                        if distances:
                            # Convert to points for GUI (every 10th point to reduce data)
                            points = []
                            for i in range(0, len(distances), 10):
                                d, a = distances[i], angles[i]
                                points.append({"x": d * np.cos(a), "y": d * np.sin(a)})
                            data["lidar_points"] = points
                        
                        last_lidar_time = current_time
                    except Exception as e:
                        if "No such file" not in str(e):  # Don't spam if disconnected
                            print(f"Lidar error: {e}")
                
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
                perf.frame_count += 1
                
                # Add profile metrics to data if profiling enabled
                if perf.should_send_summary():
                    data["profile_metrics"] = perf.get_summary()
                
                _t0 = time.time()
                message = json.dumps(data)
                send_tasks = [client.send(message) for client in connected_clients]
                if send_tasks:
                    await asyncio.gather(*send_tasks)
                    perf.record("websocket_send", time.time() - _t0)
                    
            except Exception as e:
                print(f"Producer error: {e}")
        
        # Rate limit: 10Hz
        # Rate limit: ~60Hz (limited by camera speed roughly)
        await asyncio.sleep(0.01)


async def consumer_task(websocket):
    """Handle incoming messages from a WebSocket client."""
    global robot, left_motor, right_motor, detection_enabled, is_auto_driving
    global nav_phase, nav_acquire_samples, nav_target_distance, nav_target_bearing
    
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if not robot or is_reconnecting:
                continue
            
            if msg_type == "set_power":
                motor = data.get("motor")
                power = float(data.get("power", 0.0))
                
                # Coalescing: Skip if command is redundant
                if not should_send_motor_command(motor, power):
                    continue
                
                try:
                    if motor == "left":
                        await asyncio.wait_for(left_motor.set_power(power), timeout=MOTOR_CMD_TIMEOUT)
                        update_motor_state("left", power)
                    elif motor == "right":
                        await asyncio.wait_for(right_motor.set_power(power), timeout=MOTOR_CMD_TIMEOUT)
                        update_motor_state("right", power)
                    reset_timeout_counter()  # Success - reset counter
                except asyncio.TimeoutError:
                    await handle_api_timeout(f"Motor {motor}")
            
            elif msg_type == "stop":
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            left_motor.set_power(0),
                            right_motor.set_power(0)
                        ),
                        timeout=MOTOR_CMD_TIMEOUT
                    )
                    # Update motor state to stopped
                    update_motor_state("left", 0)
                    update_motor_state("right", 0)
                    reset_timeout_counter()
                except asyncio.TimeoutError:
                    await handle_api_timeout("Stop command")
                    # Try harder with stop()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                left_motor.stop(),
                                right_motor.stop()
                            ),
                            timeout=MOTOR_CMD_TIMEOUT
                        )
                        update_motor_state("left", 0)
                        update_motor_state("right", 0)
                    except:
                        print("✗ Motor stop failed!")
            
            elif msg_type == "toggle_detection":
                detection_enabled = data.get("enabled", False)
                print(f"Detection {'enabled' if detection_enabled else 'disabled'}")

            elif msg_type == "start_auto_drive":
                 print("Starting Auto-Drive (Navigation FSM)...")
                 
                 # Reset odometry to (0,0) to avoid drift issues
                 robot_state.x = 0.0
                 robot_state.y = 0.0
                 robot_state.theta = 0.0
                 robot_state.initialized = False
                 
                 is_auto_driving = True
                 detection_enabled = True  # Force detection on
                 
                 # Start Navigation FSM (uses APPROACHING mode directly since target visible)
                 if nav_fsm:
                     await nav_fsm.start_approach()
                 
                 # Reset trajectory
                 trajectory_history.clear()
                 auto_drive_start_pos = {"x": 0.0, "y": 0.0}
            
            elif msg_type == "stop_auto_drive":
                 print("Stopping Auto-Drive...")
                 is_auto_driving = False
                 
                 # Stop Navigation FSM
                 if nav_fsm:
                     await nav_fsm.stop()
                 
                 # Reset timeout counter and wait briefly for API queue to clear
                 reset_timeout_counter()
                 await asyncio.sleep(0.5)  # Let API queue drain before manual control
            
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

async def stop_motors():
    """Emergency stop - call when server exits to ensure motors stop."""
    global left_motor, right_motor
    print("\n⚠ Stopping motors...")
    try:
        if left_motor:
            await left_motor.stop()
        if right_motor:
            await right_motor.stop()
        print("✓ Motors stopped safely")
    except Exception as e:
        print(f"✗ Error stopping motors: {e}")


async def cleanup():
    """Cleanup on shutdown."""
    global robot, is_auto_driving
    is_auto_driving = False
    await stop_motors()
    if robot:
        await robot.close()
        print("✓ Robot connection closed")


async def main():
    """Initialize and run the server."""
    global nav_fsm
    
    # Load detection model
    initialize_detection_model()
    
    # Connect to robot
    if not await connect_to_robot():
        return
    
    # Initialize Navigation FSM with calibrated config
    nav_config = NavigationConfig()
    nav_config.camera_hfov_deg = CAMERA_HFOV_DEG
    nav_config.frame_width = IMAGE_WIDTH
    nav_config.target_distance_cm = target_distance_cm
    nav_config.dist_threshold_cm = dist_threshold_cm
    nav_fsm = NavigationFSM(left_motor, right_motor, nav_config)
    print("✓ Navigation FSM initialized")
    
    # Start physics simulation loop if in SIM_MODE
    if SIM_MODE:
        asyncio.create_task(sim_physics_loop())
        print("✓ Physics simulation running at 50Hz")
    
    # Start producer task
    asyncio.create_task(producer_task())
    
    # Start WebSocket server
    # Listen on 0.0.0.0 to allow access from other machines (e.g. Laptop -> Pi)
    server_address = "0.0.0.0"
    server_port = 8081
    
    print(f"\n{'=' * 50}")
    print(f"WebSocket server running on ws://{server_address}:{server_port}")
    if PROFILE_MODE:
        print(f"📊 PROFILING ENABLED - metrics sent to GUI every 50 frames")
    print(f"Press Ctrl+C to stop (motors will be stopped safely)")
    print(f"{'=' * 50}\n")
    
    try:
        async with websockets.serve(handler, server_address, server_port):
            await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        # Note: cleanup() is called in main()'s finally block