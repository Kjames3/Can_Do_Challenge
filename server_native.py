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
import cv2
from ultralytics import YOLO
from navigation_fsm import NavigationFSM, NavigationConfig

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Native Viam Rover Control Server')
parser.add_argument('--sim', action='store_true', help='Run in simulation mode (no hardware)')
args = parser.parse_args()
SIM_MODE = args.sim

# Import websockets first (always needed)
import websockets

# Conditional hardware imports
GPIO = None
GPIOZERO_AVAILABLE = False

if not SIM_MODE:
    # Try to set up gpiozero with a working pin factory for Pi 5
    pin_factory_found = False
    
    try:
        from gpiozero import PWMOutputDevice, DigitalOutputDevice, Device, Button
        GPIOZERO_AVAILABLE = True
        
        # Try pin factories in order of preference for Pi 5
        factories_to_try = [
            ("rpi-lgpio", "gpiozero.pins.lgpio", "LGPIOFactory"),
            ("lgpio", "gpiozero.pins.lgpio", "LGPIOFactory"),
            ("pigpio", "gpiozero.pins.pigpio", "PiGPIOFactory"),
            ("native", "gpiozero.pins.native", "NativeFactory"),
        ]
        
        for name, module_path, factory_class in factories_to_try:
            try:
                module = __import__(module_path, fromlist=[factory_class])
                factory = getattr(module, factory_class)
                Device.pin_factory = factory()
                print(f"✓ Using {name} pin factory")
                pin_factory_found = True
                break
            except (ImportError, Exception) as e:
                continue
        
        if not pin_factory_found:
            print("⚠ No suitable GPIO pin factory found!")
            print("  On Raspberry Pi 5, install: sudo apt install python3-lgpio")
            print("  Or: pip install rpi-lgpio")
            SIM_MODE = True
            
    except ImportError as e:
        print(f"⚠ gpiozero not available: {e}")
        print("  Install with: pip install gpiozero")
        print("  Running in simulation mode")
        SIM_MODE = True

# =============================================================================
# GPIO PIN CONFIGURATION (from Viam Dashboard)
# =============================================================================
# These are BOARD pin numbers (physical pins on the Pi header)

# Left Motor (In1/In2 + PWM)
LEFT_MOTOR_PIN_A = 35      # Forward/In1
LEFT_MOTOR_PIN_B = 33      # Backward/In2
LEFT_MOTOR_PWM = 37        # PWM speed control

# Right Motor (In1/In2 + PWM) 
RIGHT_MOTOR_PIN_A = 31     # Forward/In1
RIGHT_MOTOR_PIN_B = 29     # Backward/In2
RIGHT_MOTOR_PWM = 15       # PWM speed control

# Encoders (single channel)
LEFT_ENCODER_PIN = 38
RIGHT_ENCODER_PIN = 40

# I2C Devices
I2C_BUS = 1                # I2C bus number

# Camera
# TODO: If camera path is different, change it here. Check if camera is connected by running `ls /dev/v4l/by-id/` in the terminal.
CAMERA_PATH = "/dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0"
CAMERA_INDEX = 0           # Fallback to index if path fails

# LIDAR
LIDAR_PORT = "/dev/ttyUSB0"

# =============================================================================
# ROBOT PARAMETERS (from Viam config)
# =============================================================================
WHEEL_CIRCUMFERENCE_MM = 381    # mm
WHEEL_BASE_MM = 356             # mm (width between wheels)
WHEEL_DIAMETER_CM = WHEEL_CIRCUMFERENCE_MM / (np.pi * 10)  # Convert to cm
WHEEL_BASE_CM = WHEEL_BASE_MM / 10

# Motor drift compensation (adjust if rover drifts when driving straight)
# Positive = reduce right motor power, Negative = reduce left motor power
DRIFT_COMPENSATION = 0.05  # 5% reduction on right motor (rover was drifting right)

# Detection Configuration
KNOWN_HEIGHT_BOTTLE = 20.0  # Standard water bottle height in cm
KNOWN_HEIGHT_CAN = 12.0     # Standard soda can height in cm
FOCAL_LENGTH = 600          # Webcam focal length (calibrate for accuracy)
TARGET_CLASSES = [0]        # Custom model: 0=can

# Camera Settings
CAMERA_HFOV_DEG = 76.5
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Performance Settings
VIDEO_FPS_CAP = 20          # Reduced to avoid camera buffer issues
JPEG_QUALITY = 70
DETECTION_INTERVAL = 1      # Run detection every frame for better tracking
CONFIDENCE_THRESHOLD = 0.25 # Lower threshold for longer range detection

# Detection inference size
INFERENCE_SIZE = 640        # Larger = better long-range detection, slower

# YOLO Model (YOLO11 preferred, fallback to YOLOv8)
YOLO_MODEL = 'yolo11n_cans.pt'  # Will fallback to yolov8n_cans.pt if not found

# =============================================================================
# IMU CONFIGURATION (MPU6050)
# =============================================================================
IMU_I2C_BUS = 1              # I2C bus number (from Viam config)
IMU_I2C_ADDRESS = 0x68       # Default MPU6050 address
IMU_SAMPLE_RATE = 50         # Hz - how often to read IMU
IMU_GYRO_SCALE = 131.0       # LSB/(°/s) for ±250°/s range

# Drift Compensation Settings
DRIFT_CORRECTION_ENABLED = True
DRIFT_CORRECTION_GAIN = 0.02  # Proportional gain for heading correction
VELOCITY_MATCH_GAIN = 0.05    # Proportional gain for encoder velocity matching

# Tilt Safety Settings
# NOTE: Disabled because IMU calibration can cause false positives
# Re-enable after IMU is properly calibrated on a level surface
TILT_SAFETY_ENABLED = False
MAX_TILT_DEGREES = 45.0       # Emergency stop if pitch or roll exceeds this (increased from 30)

# Stuck Detection Settings
STUCK_DETECTION_ENABLED = True
STUCK_MOTOR_THRESHOLD = 0.15  # Min motor power to check for stuck
STUCK_TIME_THRESHOLD = 1.5    # Seconds of no movement before declaring stuck
STUCK_ACCEL_THRESHOLD = 0.1   # g's - min acceleration to indicate movement
STUCK_ENCODER_THRESHOLD = 5   # Min encoder pulses per second to indicate movement

# =============================================================================
# MOTOR DRIVER CLASS (In1/In2 + PWM Pattern)
# =============================================================================

class NativeMotor:
    """
    Direct GPIO motor control using In1/In2 + PWM pattern.
    
    In1/In2 Logic:
    - Forward:  In1=HIGH, In2=LOW, PWM controls speed
    - Backward: In1=LOW, In2=HIGH, PWM controls speed
    - Stop:     In1=LOW, In2=LOW (or PWM=0)
    """
    
    def __init__(self, pin_a, pin_b, pin_pwm, name="motor"):
        self.name = name
        self._power = 0.0
        self._encoder = None  # Associated encoder for direction tracking
        
        if SIM_MODE:
            return
        
        # Use BCM pin numbering internally
        # Convert BOARD pins to BCM
        self.pin_a_bcm = self._board_to_bcm(pin_a)
        self.pin_b_bcm = self._board_to_bcm(pin_b)
        self.pin_pwm_bcm = self._board_to_bcm(pin_pwm)
        
        # Initialize GPIO devices
        self.in1 = DigitalOutputDevice(self.pin_a_bcm)
        self.in2 = DigitalOutputDevice(self.pin_b_bcm)
        self.pwm = PWMOutputDevice(self.pin_pwm_bcm, frequency=1000)
        
        print(f"  ✓ {name}: In1=GPIO{self.pin_a_bcm}, In2=GPIO{self.pin_b_bcm}, PWM=GPIO{self.pin_pwm_bcm}")
    
    def set_encoder(self, encoder):
        """Associate an encoder with this motor for direction tracking."""
        self._encoder = encoder
    
    def _board_to_bcm(self, board_pin):
        """Convert physical BOARD pin number to BCM GPIO number."""
        # Raspberry Pi 5 BOARD to BCM mapping
        board_to_bcm = {
            3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
            15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11, 24: 8,
            26: 7, 27: 0, 28: 1, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19,
            36: 16, 37: 26, 38: 20, 40: 21
        }
        return board_to_bcm.get(board_pin, board_pin)
    
    def set_power(self, power: float):
        """
        Set motor power from -1.0 to 1.0.
        Positive = forward, Negative = backward.
        """
        self._power = max(-1.0, min(1.0, power))
        
        # Update encoder direction
        if self._encoder:
            self._encoder.set_direction(self._power >= 0)
        
        if SIM_MODE:
            return
        
        if abs(self._power) < 0.01:
            # Stop
            self.in1.off()
            self.in2.off()
            self.pwm.value = 0
        elif self._power > 0:
            # Forward
            self.in1.on()
            self.in2.off()
            self.pwm.value = abs(self._power)
        else:
            # Backward
            self.in1.off()
            self.in2.on()
            self.pwm.value = abs(self._power)
    
    def stop(self):
        """Emergency stop."""
        self.set_power(0)
    
    @property
    def power(self):
        return self._power
    
    def cleanup(self):
        """Release GPIO resources."""
        if not SIM_MODE:
            self.stop()
            self.in1.close()
            self.in2.close()
            self.pwm.close()


# =============================================================================
# ENCODER CLASS (Single Channel with Interrupt)
# =============================================================================

class NativeEncoder:
    """
    Single-channel encoder using gpiozero Button for Pi 5 compatibility.
    Counts pulses on button press events.
    """
    
    def __init__(self, pin, name="encoder", ppr=1000):
        """
        Args:
            pin: BOARD pin number
            name: Encoder name for logging
            ppr: Pulses per revolution (encoder resolution, default 1000)
        """
        self.name = name
        self.ppr = ppr
        self._count = 0
        self._direction = 1  # 1 = forward, -1 = backward
        self._lock = threading.Lock()
        self._button = None
        
        if SIM_MODE:
            return
        
        self.pin_bcm = self._board_to_bcm(pin)
        
        try:
            # Use gpiozero Button which works with lgpio on Pi 5
            from gpiozero import Button
            self._button = Button(self.pin_bcm, pull_up=True)
            self._button.when_pressed = self._pulse_callback
            self._button.when_released = self._pulse_callback  # Count both edges
            print(f"  ✓ {name}: GPIO{self.pin_bcm}")
        except Exception as e:
            print(f"  ⚠ {name} init failed: {e}")
    
    def _board_to_bcm(self, board_pin):
        """Convert physical BOARD pin to BCM GPIO number."""
        board_to_bcm = {
            3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
            15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11, 24: 8,
            26: 7, 27: 0, 28: 1, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19,
            36: 16, 37: 26, 38: 20, 40: 21
        }
        return board_to_bcm.get(board_pin, board_pin)
    
    def set_direction(self, forward: bool):
        """Set direction for pulse counting (call based on motor power)."""
        with self._lock:
            self._direction = 1 if forward else -1
    
    def _pulse_callback(self):
        """Called on each encoder pulse (edge)."""
        with self._lock:
            self._count += self._direction
    
    def get_position(self):
        """Get encoder position in revolutions (signed)."""
        with self._lock:
            return self._count / self.ppr
    
    def get_count(self):
        """Get raw pulse count (signed)."""
        with self._lock:
            return self._count
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._count = 0
    
    def cleanup(self):
        """Release GPIO resources."""
        if self._button:
            self._button.close()


# =============================================================================
# IMU CLASS (MPU6050 via I2C)
# =============================================================================

class NativeIMU:
    """
    Direct I2C access to MPU6050 IMU for heading, tilt, and motion detection.
    
    Features:
    - Gyroscope integration for heading (yaw)
    - Accelerometer for pitch/roll (tilt detection)
    - Motion detection for stuck detection
    """
    
    # MPU6050 Register addresses
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    
    def __init__(self, bus: int = 1, address: int = 0x68, name: str = "imu"):
        self.name = name
        self.bus = bus
        self.address = address
        self._smbus = None
        self._initialized = False
        
        # Heading tracking (integrated from gyro Z)
        self._heading = 0.0  # radians
        self._heading_offset = 0.0  # For calibration
        self._last_update = time.time()
        
        # Calibration offsets
        self._gyro_offset = [0.0, 0.0, 0.0]
        self._accel_offset = [0.0, 0.0, 0.0]
        
        # Motion state
        self._is_moving = False
        self._motion_threshold = STUCK_ACCEL_THRESHOLD
        
        self._lock = threading.Lock()
        
        if not SIM_MODE:
            self._initialize_hardware()
    
    def _initialize_hardware(self):
        """Initialize I2C connection to MPU6050."""
        try:
            import smbus2
            self._smbus = smbus2.SMBus(self.bus)
            
            # Wake up MPU6050 (it starts in sleep mode)
            self._smbus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Configure gyro for ±250°/s (most sensitive)
            self._smbus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
            
            # Configure accelerometer for ±2g (most sensitive)
            self._smbus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)
            
            self._initialized = True
            print(f"  ✓ {self.name} initialized on I2C bus {self.bus}, addr 0x{self.address:02x}")
            
            # Auto-calibrate (assumes stationary at startup)
            self._calibrate()
            
        except ImportError:
            print(f"  ✗ {self.name}: smbus2 not installed (pip install smbus2)")
            self._initialized = False
        except Exception as e:
            print(f"  ✗ {self.name}: I2C error - {e}")
            self._initialized = False
    
    def _calibrate(self, samples: int = 100):
        """Calibrate gyro/accel offsets (call when stationary)."""
        if not self._initialized:
            return
        
        print(f"  ⏳ Calibrating {self.name} (keep rover still)...")
        
        gyro_sum = [0.0, 0.0, 0.0]
        accel_sum = [0.0, 0.0, 0.0]
        
        for _ in range(samples):
            raw_gyro = self._read_raw_gyro()
            raw_accel = self._read_raw_accel()
            for i in range(3):
                gyro_sum[i] += raw_gyro[i]
                accel_sum[i] += raw_accel[i]
            time.sleep(0.01)
        
        self._gyro_offset = [g / samples for g in gyro_sum]
        # Don't offset Z accel (should be ~1g due to gravity)
        self._accel_offset = [accel_sum[0] / samples, accel_sum[1] / samples, 0.0]
        
        print(f"  ✓ {self.name} calibrated")
    
    def _read_raw_word(self, reg: int) -> int:
        """Read a signed 16-bit value from two registers."""
        high = self._smbus.read_byte_data(self.address, reg)
        low = self._smbus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value -= 0x10000
        return value
    
    def _read_raw_gyro(self) -> list:
        """Read raw gyroscope values (LSB units)."""
        if not self._initialized:
            return [0.0, 0.0, 0.0]
        try:
            gx = self._read_raw_word(self.GYRO_XOUT_H)
            gy = self._read_raw_word(self.GYRO_XOUT_H + 2)
            gz = self._read_raw_word(self.GYRO_XOUT_H + 4)
            return [gx, gy, gz]
        except:
            return [0.0, 0.0, 0.0]
    
    def _read_raw_accel(self) -> list:
        """Read raw accelerometer values (LSB units)."""
        if not self._initialized:
            return [0.0, 0.0, 16384.0]  # Fake 1g on Z
        try:
            ax = self._read_raw_word(self.ACCEL_XOUT_H)
            ay = self._read_raw_word(self.ACCEL_XOUT_H + 2)
            az = self._read_raw_word(self.ACCEL_XOUT_H + 4)
            return [ax, ay, az]
        except:
            return [0.0, 0.0, 16384.0]
    
    def get_gyro(self) -> tuple:
        """Get gyroscope readings in degrees/second."""
        raw = self._read_raw_gyro()
        # Convert to degrees/sec (131 LSB/(°/s) for ±250°/s range)
        gx = (raw[0] - self._gyro_offset[0]) / IMU_GYRO_SCALE
        gy = (raw[1] - self._gyro_offset[1]) / IMU_GYRO_SCALE
        gz = (raw[2] - self._gyro_offset[2]) / IMU_GYRO_SCALE
        return (gx, gy, gz)
    
    def get_accel(self) -> tuple:
        """Get accelerometer readings in g's."""
        raw = self._read_raw_accel()
        # Convert to g's (16384 LSB/g for ±2g range)
        ax = (raw[0] - self._accel_offset[0]) / 16384.0
        ay = (raw[1] - self._accel_offset[1]) / 16384.0
        az = raw[2] / 16384.0  # Don't offset Z
        return (ax, ay, az)
    
    def update(self):
        """
        Update heading by integrating gyroscope.
        Call this frequently (at least 50Hz) for accurate heading.
        """
        if not self._initialized:
            return
        
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        if dt > 0.5:
            # Too long since last update, skip integration
            return
        
        _, _, gz = self.get_gyro()
        
        with self._lock:
            # Integrate gyro Z to get heading (convert deg/s to rad)
            self._heading += np.radians(gz) * dt
            # Normalize to -π to π
            self._heading = np.arctan2(np.sin(self._heading), np.cos(self._heading))
        
        # Update motion state
        ax, ay, az = self.get_accel()
        accel_magnitude = np.sqrt(ax*ax + ay*ay)  # Ignore Z (gravity)
        self._is_moving = accel_magnitude > self._motion_threshold
    
    def get_heading(self) -> float:
        """Get current heading in radians (integrated from gyro Z)."""
        with self._lock:
            return self._heading - self._heading_offset
    
    def reset_heading(self):
        """Reset heading to zero (set current heading as reference)."""
        with self._lock:
            self._heading_offset = self._heading
    
    def get_tilt(self) -> tuple:
        """
        Get pitch and roll angles in degrees from accelerometer.
        
        Returns:
            (pitch, roll) - pitch is forward/back tilt, roll is side tilt
        """
        ax, ay, az = self.get_accel()
        
        # Calculate pitch (rotation around Y axis) and roll (rotation around X axis)
        pitch = np.degrees(np.arctan2(ax, np.sqrt(ay*ay + az*az)))
        roll = np.degrees(np.arctan2(ay, np.sqrt(ax*ax + az*az)))
        
        return (pitch, roll)
    
    def is_tilted_unsafe(self) -> bool:
        """Check if rover is tilted beyond safe limits."""
        pitch, roll = self.get_tilt()
        return abs(pitch) > MAX_TILT_DEGREES or abs(roll) > MAX_TILT_DEGREES
    
    def is_moving(self) -> bool:
        """Check if rover is moving based on accelerometer."""
        return self._is_moving
    
    def get_yaw_rate(self) -> float:
        """Get current yaw rate in degrees/second."""
        _, _, gz = self.get_gyro()
        return gz
    
    def cleanup(self):
        """Release I2C resources."""
        if self._smbus:
            self._smbus.close()


# =============================================================================
# CAMERA CLASS (OpenCV)
# =============================================================================

class NativeCamera:
    """
    Direct camera access via OpenCV VideoCapture.
    Much faster than Viam's network-based approach.
    """
    
    def __init__(self, device=0, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        
        if SIM_MODE:
            return
        
        # Try device path first, then index
        if isinstance(device, str):
            self.cap = cv2.VideoCapture(device)
            if not self.cap.isOpened():
                print(f"  ⚠ Camera path {device} failed, trying index 0")
                self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(device)
        
        if not self.cap.isOpened():
            print("  ✗ Failed to open camera")
            return
        
        # Use MJPEG format for faster capture and less tearing
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduce buffer size to minimize tearing (only keep 1 frame in buffer)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Flush initial stale frames from the buffer
        for _ in range(5):
            self.cap.grab()
        
        # Start capture thread for non-blocking reads
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        print(f"  ✓ Camera: {width}x{height} (MJPEG, buffer=1)")
    
    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        frame_errors = 0
        frame_ok_count = 0
        while self._running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self._frame_lock:
                        self._frame = frame
                    frame_ok_count += 1
                    frame_errors = 0  # Reset on success
                    # Log first successful frame
                    if frame_ok_count == 1:
                        print(f"  ✓ Camera: First frame captured successfully")
                else:
                    frame_errors += 1
                    if frame_errors == 10:
                        print(f"  ⚠ Camera: Failed to read frame (10 consecutive failures)")
            except Exception as e:
                frame_errors += 1
                if frame_errors == 1:
                    print(f"  ⚠ Camera capture error: {e}")
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def get_frame(self):
        """Get the latest frame (non-blocking)."""
        if SIM_MODE:
            # Generate test pattern
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "SIM MODE", (self.width//2 - 80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None
    
    def get_jpeg(self, quality=75):
        """Get frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    
    def cleanup(self):
        """Release camera resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# =============================================================================
# LIDAR CLASS (rplidar)
# =============================================================================

class NativeLidar:
    """
    RPLIDAR A1 access via rplidar-roboticia library.
    Runs scanning in background thread.
    """
    
    def __init__(self, port="/dev/ttyUSB0"):
        self.port = port
        self._scans = []
        self._scan_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._lidar = None
        
        if SIM_MODE:
            return
        
        try:
            from rplidar import RPLidar
            self._lidar = RPLidar(port)
            self._lidar.connect()
            
            # Start scanning thread
            self._running = True
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            
            print(f"  ✓ LIDAR: {port}")
        except Exception as e:
            print(f"  ⚠ LIDAR init failed: {e}")
            self._lidar = None
    
    def _scan_loop(self):
        """Background thread for continuous scanning."""
        try:
            for scan in self._lidar.iter_scans():
                if not self._running:
                    break
                
                # Convert to (angle_rad, distance_m) pairs
                points = []
                for _, angle, distance in scan:
                    if distance > 0:
                        angle_rad = np.radians(angle)
                        dist_m = distance / 1000.0
                        points.append((angle_rad, dist_m))
                
                with self._scan_lock:
                    self._scans = points
        except Exception as e:
            print(f"LIDAR scan error: {e}")
    
    def get_scan(self):
        """Get latest scan data as list of (angle, distance) tuples."""
        if SIM_MODE:
            return []
        
        with self._scan_lock:
            return list(self._scans)
    
    def get_points_xy(self):
        """Get scan as list of [x, y] coordinates in meters."""
        scan = self.get_scan()
        points = []
        for angle, dist in scan:
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            points.append([x, y])
        return points
    
    def cleanup(self):
        """Stop scanning and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.disconnect()
            except:
                pass


# =============================================================================
# ODOMETRY STATE
# =============================================================================

class RobotState:
    """Track robot pose using wheel encoder odometry with optional IMU heading fusion."""
    
    def __init__(self):
        self.x = 0.0  # cm
        self.y = 0.0  # cm
        self.theta = 0.0  # radians
        self.last_left_pos = 0.0
        self.last_right_pos = 0.0
        self.initialized = False
    
    def update(self, left_pos, right_pos):
        """Update pose from encoder positions (in revolutions)."""
        if not self.initialized:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.initialized = True
            return
        
        # Delta in revolutions
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        # Convert to distance (cm)
        left_dist = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        right_dist = right_delta * WHEEL_CIRCUMFERENCE_MM / 10
        
        # Differential drive kinematics
        linear = (left_dist + right_dist) / 2.0
        angular = (right_dist - left_dist) / WHEEL_BASE_CM
        
        # Update pose
        self.x += linear * np.sin(self.theta)
        self.y += linear * np.cos(self.theta)
        self.theta += angular
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos
    
    def update_with_imu(self, left_pos, right_pos, imu_heading):
        """
        Update pose using encoder distance + IMU heading (more accurate).
        
        Args:
            left_pos: Left encoder position in revolutions
            right_pos: Right encoder position in revolutions
            imu_heading: Heading from IMU in radians
        """
        if not self.initialized:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.initialized = True
            return
        
        # Delta in revolutions
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        # Convert to distance (cm)
        left_dist = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        right_dist = right_delta * WHEEL_CIRCUMFERENCE_MM / 10
        
        # Average distance for forward movement
        distance = (left_dist + right_dist) / 2
        
        # Use IMU heading directly (much more accurate than encoder-derived)
        self.theta = imu_heading
        
        # Update position using IMU heading
        # Convention: Y is forward (cos), X is lateral (sin)
        self.x += distance * np.sin(self.theta)
        self.y += distance * np.cos(self.theta)
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos


# =============================================================================
# GLOBAL STATE
# =============================================================================

left_motor: NativeMotor = None
right_motor: NativeMotor = None
left_encoder: NativeEncoder = None
right_encoder: NativeEncoder = None
camera: NativeCamera = None
lidar: NativeLidar = None
imu: NativeIMU = None  # MPU6050 IMU
robot_state = RobotState()
fsm: NavigationFSM = None  # Navigation FSM

detection_model: YOLO = None
detection_enabled = False
is_auto_driving = False
is_stuck = False  # Set True when motors running but no movement
is_tilted = False  # Set True if tilt exceeds safe limits
frame_count = 0
last_detections = []

# Stuck detection state
_stuck_start_time = None
_last_encoder_count = 0

connected_clients = set()

# =============================================================================
# DETECTION
# =============================================================================

def initialize_detection():
    """Load YOLO model."""
    global detection_model
    try:
        detection_model = YOLO(YOLO_MODEL)
        print(f"✓ YOLO model loaded: {YOLO_MODEL}")
        return True
    except Exception as e:
        print(f"✗ YOLO load failed: {e}")
        return False


def process_detection(frame):
    """Run YOLO detection on frame, return annotated frame and detections."""
    global detection_model
    
    if detection_model is None or frame is None:
        return frame, []
    
    results = detection_model(frame, imgsz=INFERENCE_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            label = detection_model.names[cls_id]
            
            # Calculate distance
            height_px = y2 - y1
            real_height = KNOWN_HEIGHT_CAN
            distance_cm = (real_height * FOCAL_LENGTH) / height_px if height_px > 0 else 0
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "distance_cm": round(distance_cm, 1),
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [x1, y1, x2, y2]
            })
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} {int(distance_cm)}cm", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detections


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

async def handle_client(websocket):
    """Handle WebSocket client connection (websockets 10+ API)."""
    global detection_enabled, is_auto_driving
    
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "set_power":
                    motor = data.get("motor")
                    power = float(data.get("power", 0))
                    
                    # Apply drift compensation for forward driving
                    # Only apply when both motors should be moving together
                    if motor == "left" and left_motor:
                        if DRIFT_COMPENSATION < 0:
                            # Reduce left motor power
                            power = power * (1.0 + DRIFT_COMPENSATION)
                        left_motor.set_power(power)
                    elif motor == "right" and right_motor:
                        if DRIFT_COMPENSATION > 0:
                            # Reduce right motor power
                            power = power * (1.0 - DRIFT_COMPENSATION)
                        right_motor.set_power(power)
                
                elif msg_type == "stop":
                    if left_motor:
                        left_motor.stop()
                    if right_motor:
                        right_motor.stop()
                    is_auto_driving = False
                    if fsm:
                        await fsm.stop()
                
                elif msg_type == "toggle_detection":
                    detection_enabled = data.get("enabled", False)
                
                elif msg_type == "start_auto_drive":
                    is_auto_driving = True
                    detection_enabled = True
                    if fsm:
                        if left_motor and right_motor:
                            fsm.update_motors(left_motor, right_motor)
                        await fsm.start()
                
                elif msg_type == "stop_auto_drive":
                    is_auto_driving = False
                    if fsm:
                        await fsm.stop()
                    if left_motor:
                        left_motor.stop()
                    if right_motor:
                        right_motor.stop()
                
                elif msg_type == "capture_image":
                    # Capture current frame and save for training
                    if camera:
                        frame = camera.get_frame()
                        if frame is not None:
                            # Create directory if needed
                            import os
                            from datetime import datetime
                            capture_dir = "captured_images"
                            os.makedirs(capture_dir, exist_ok=True)
                            
                            # Save with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filepath = os.path.join(capture_dir, f"can_{timestamp}.jpg")
                            cv2.imwrite(filepath, frame)
                            
                            # Count existing images
                            count = len([f for f in os.listdir(capture_dir) if f.endswith('.jpg')])
                            print(f"📸 Captured image: {filepath} (total: {count})")
                            
                            # Send count back to client
                            await websocket.send(json.dumps({
                                "type": "capture_response",
                                "count": count,
                                "filename": filepath
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
    
    # Auto-drive control parameters
    TARGET_DISTANCE_CM = 25.0  # Stop when this close to target
    CENTER_THRESHOLD_PX = 80   # Acceptable centering error
    DRIVE_SPEED = 0.25
    TURN_SPEED = 0.20
    
    # For drift correction during straight driving
    initial_heading = None
    
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
                        if left_motor:
                            left_motor.stop()
                        if right_motor:
                            right_motor.stop()
                        is_auto_driving = False
                elif is_tilted:
                    # Clear tilt flag when level again
                    is_tilted = False
                    print("✓ Tilt returned to safe range")
                
                # --- STUCK DETECTION ---
                if STUCK_DETECTION_ENABLED and left_motor and right_motor:
                    motor_power = max(abs(left_motor._power), abs(right_motor._power))
                    
                    if motor_power > STUCK_MOTOR_THRESHOLD:
                        # Check if actually moving
                        encoder_count = 0
                        if left_encoder:
                            encoder_count += abs(left_encoder.get_count())
                        if right_encoder:
                            encoder_count += abs(right_encoder.get_count())
                        
                        imu_moving = imu.is_moving() if imu else True
                        encoder_moving = abs(encoder_count - _last_encoder_count) > STUCK_ENCODER_THRESHOLD
                        
                        if not imu_moving and not encoder_moving:
                            # Possibly stuck
                            if _stuck_start_time is None:
                                _stuck_start_time = current_time
                            elif current_time - _stuck_start_time > STUCK_TIME_THRESHOLD:
                                if not is_stuck:
                                    is_stuck = True
                                    print("⚠️ STUCK: Motors running but no movement detected!")
                                    # Optionally stop motors
                                    # left_motor.stop()
                                    # right_motor.stop()
                        else:
                            _stuck_start_time = None
                            if is_stuck:
                                is_stuck = False
                                print("✓ Movement detected, no longer stuck")
                        
                        _last_encoder_count = encoder_count
                    else:
                        _stuck_start_time = None
            
            # Throttle video frame rate
            if current_time - last_video_time < video_interval:
                await asyncio.sleep(0.005)
                continue
            
            last_video_time = current_time
            
            # --- UPDATE ODOMETRY (with IMU heading if available) ---
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
                
                # Run detection on interval
                if detection_enabled and frame_count % DETECTION_INTERVAL == 0:
                    frame, last_detections = process_detection(frame)
                elif detection_enabled:
                    # Draw previous detections
                    for d in last_detections:
                        x1, y1, x2, y2 = d['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # === AUTO-DRIVE CONTROL (FSM) ===
                if is_auto_driving and fsm:
                    # Get closest detection
                    detection = None
                    if last_detections:
                        # Find closest target
                        target = min(last_detections, key=lambda d: d['distance_cm'])
                        detection = {
                            'distance_cm': target['distance_cm'],
                            'center_x': target['center_x']
                        }
                    
                    # Get lidar minimum distance for obstacle avoidance
                    lidar_min = None
                    if lidar:
                        # Simple check: min distance in front sector (-45 to +45 deg)
                        scan = lidar.get_scan()
                        front_dists = [d for a, d in scan if -0.78 < a < 0.78] # +/- 45 deg
                        if front_dists:
                            lidar_min = min(front_dists) * 100.0 # Convert to cm
                    
                    await fsm.update(detection, lidar_min)
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Build data packet
            data = {
                "type": "readout",
                "left_pos": left_encoder.get_position() if left_encoder else 0,
                "left_power": left_motor.power if left_motor else 0,
                "right_pos": right_encoder.get_position() if right_encoder else 0,
                "right_power": right_motor.power if right_motor else 0,
                "image": image_b64,
                "detection_enabled": detection_enabled,
                "detections": last_detections,
                "is_auto_driving": is_auto_driving,
                "is_stuck": is_stuck,
                "is_tilted": is_tilted,
                "robot_pose": {
                    "x": robot_state.x,
                    "y": robot_state.y,
                    "theta": robot_state.theta
                },
                "imu": {
                    "pitch": imu.get_tilt()[0] if imu else 0,
                    "roll": imu.get_tilt()[1] if imu else 0,
                    "heading_deg": np.degrees(imu.get_heading()) if imu else 0,
                    "yaw_rate": imu.get_yaw_rate() if imu else 0
                } if imu else None,
                "imu": {
                    "pitch": imu.get_tilt()[0] if imu else 0,
                    "roll": imu.get_tilt()[1] if imu else 0,
                    "heading_deg": np.degrees(imu.get_heading()) if imu else 0,
                    "yaw_rate": imu.get_yaw_rate() if imu else 0
                } if imu else None,
                "lidar_points": lidar.get_points_xy()[:360] if lidar else [],
                "fsm_state": fsm.state_summary if fsm else "IDLE"
            }
            
            # Broadcast to all clients
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in connected_clients],
                return_exceptions=True
            )
        
        await asyncio.sleep(0.001)


# =============================================================================
# INITIALIZATION AND MAIN
# =============================================================================

def initialize_hardware():
    """Initialize all hardware components."""
    global left_motor, right_motor, left_encoder, right_encoder, camera, lidar, imu
    
    print("\n" + "="*50)
    print("Initializing Hardware (Native GPIO)")
    print("="*50)
    
    # Motors
    print("\nMotors:")
    left_motor = NativeMotor(LEFT_MOTOR_PIN_A, LEFT_MOTOR_PIN_B, LEFT_MOTOR_PWM, "left_motor")
    right_motor = NativeMotor(RIGHT_MOTOR_PIN_A, RIGHT_MOTOR_PIN_B, RIGHT_MOTOR_PWM, "right_motor")
    
    # Encoders
    print("\nEncoders:")
    left_encoder = NativeEncoder(LEFT_ENCODER_PIN, "left_encoder")
    right_encoder = NativeEncoder(RIGHT_ENCODER_PIN, "right_encoder")
    
    # Link motors to encoders for direction tracking
    left_motor.set_encoder(left_encoder)
    right_motor.set_encoder(right_encoder)
    print("  ✓ Motors linked to encoders for direction tracking")
    
    # IMU
    print("\nIMU (MPU6050):")
    imu = NativeIMU(IMU_I2C_BUS, IMU_I2C_ADDRESS, "imu")
    
    # Camera
    print("\nCamera:")
    camera = NativeCamera(CAMERA_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)
    
    # LIDAR
    print("\nLIDAR:")
    lidar = NativeLidar(LIDAR_PORT)
    
    # Detection model
    print("\nDetection:")
    initialize_detection()
    
    # Initialize Navigation FSM
    global fsm
    print("\nNavigation FSM:")
    fsm = NavigationFSM(left_motor, right_motor, imu=imu)
    
    # Wire up callbacks
    def on_arrived():
        global is_auto_driving
        print("🎉 FSM Callback: TARGET REACHED! Disengaging auto-drive.")
        is_auto_driving = False
    
    fsm.on_arrived = on_arrived
    print("✓ FSM initialized (IMU enabled)" if imu else "✓ FSM initialized (Camera only)")
    
    print("\n" + "="*50)
    print("✓ Hardware initialization complete")
    print("="*50 + "\n")


def cleanup():
    """Cleanup all hardware resources."""
    print("\nCleaning up...")
    
    if left_motor:
        left_motor.cleanup()
    if right_motor:
        right_motor.cleanup()
    if left_encoder:
        left_encoder.cleanup()
    if right_encoder:
        right_encoder.cleanup()
    if camera:
        camera.cleanup()
    if lidar:
        lidar.cleanup()
    if imu:
        imu.cleanup()
    
    # Only cleanup GPIO if it was initialized (not None)
    if not SIM_MODE and GPIO is not None:
        try:
            GPIO.cleanup()
        except:
            pass  # GPIO may not have cleanup method depending on library
    
    print("Cleanup complete.")


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  NATIVE RASPBERRY PI SERVER (No Viam SDK)")
    print("  Zero API limits | Zero network latency")
    print("="*60)
    
    if SIM_MODE:
        print("\n⚠ SIMULATION MODE - No hardware control")
    
    initialize_hardware()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    # Start WebSocket server
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8081,
        ping_interval=20,
        ping_timeout=60
    )
    
    print(f"\n{'='*50}")
    print(f"WebSocket server running on ws://0.0.0.0:8081")
    print(f"{'='*50}\n")
    
    # Start broadcast loop
    broadcast_task = asyncio.create_task(broadcast_loop())
    
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        broadcast_task.cancel()
        server.close()
        await server.wait_closed()
        cleanup()


async def shutdown():
    """Graceful shutdown handler."""
    print("\nShutting down...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
