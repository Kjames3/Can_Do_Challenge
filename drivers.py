
import threading
import time
import numpy as np


try:
    from picamera2 import Picamera2
    from libcamera import controls
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False
    print("⚠ Picamera2 not found (Run: pip install picamera2)")

def configure_pin_factory():
    """
    Attempt to set up a working GPIO pin factory.
    Crucial for Raspberry Pi 5 which requires lgpio/rpi-lgpio.
    Returns True if successful, False otherwise.
    """
    try:
        from gpiozero import Device
        
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
                return True
            except (ImportError, Exception):
                continue
                
        print("⚠ No suitable GPIO pin factory found!")
        print("  On Raspberry Pi 5, install: sudo apt install python3-lgpio")
        return False
        
    except ImportError:
        print("⚠ gpiozero not installed")
        return False

# =============================================================================
# HARDWARE CONFIGURATION CONSTANTS
# =============================================================================

# IMU
IMU_I2C_BUS = 1
IMU_I2C_ADDRESS = 0x68
IMU_SAMPLE_RATE = 50
IMU_GYRO_SCALE = 131.0  # LSB/(°/s) for ±250°/s range

# Drift Compensation
DRIFT_CORRECTION_ENABLED = True
DRIFT_CORRECTION_GAIN = 0.02
VELOCITY_MATCH_GAIN = 0.05

# Safety
TILT_SAFETY_ENABLED = False
MAX_TILT_DEGREES = 45.0

# Stuck Detection
STUCK_DETECTION_ENABLED = True
STUCK_MOTOR_THRESHOLD = 0.15
STUCK_TIME_THRESHOLD = 1.5
STUCK_ACCEL_THRESHOLD = 0.1
STUCK_ENCODER_THRESHOLD = 5

# =============================================================================
# MOTOR DRIVER CLASS
# =============================================================================

class NativeMotor:
    """
    Direct GPIO motor control using In1/In2 + PWM pattern.
    """
    
    def __init__(self, pin_a, pin_b, pin_pwm, sim_mode=False, name="motor"):
        self.name = name
        self.sim_mode = sim_mode
        self._power = 0.0
        self._encoder = None
        
        if self.sim_mode:
            return
            
        from gpiozero import PWMOutputDevice, DigitalOutputDevice
        
        self.pin_a_bcm = self._board_to_bcm(pin_a)
        self.pin_b_bcm = self._board_to_bcm(pin_b)
        self.pin_pwm_bcm = self._board_to_bcm(pin_pwm)
        
        self.in1 = DigitalOutputDevice(self.pin_a_bcm)
        self.in2 = DigitalOutputDevice(self.pin_b_bcm)
        self.pwm = PWMOutputDevice(self.pin_pwm_bcm, frequency=1000)
        
        print(f"  ✓ {name}: In1=GPIO{self.pin_a_bcm}, In2=GPIO{self.pin_b_bcm}, PWM=GPIO{self.pin_pwm_bcm}")
    
    def set_encoder(self, encoder):
        self._encoder = encoder
    
    def _board_to_bcm(self, board_pin):
        board_to_bcm = {
            3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
            15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11, 24: 8,
            26: 7, 27: 0, 28: 1, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19,
            36: 16, 37: 26, 38: 20, 40: 21
        }
        return board_to_bcm.get(board_pin, board_pin)
    
    def set_power(self, power: float):
        self._power = max(-1.0, min(1.0, power))
        
        if self._encoder:
            self._encoder.set_direction(self._power >= 0)
        
        if self.sim_mode:
            return
        
        if abs(self._power) < 0.01:
            self.in1.off()
            self.in2.off()
            self.pwm.value = 0
        elif self._power > 0:
            self.in1.on()
            self.in2.off()
            self.pwm.value = abs(self._power)
        else:
            self.in1.off()
            self.in2.on()
            self.pwm.value = abs(self._power)
    
    def stop(self):
        self.set_power(0)
    
    @property
    def power(self):
        return self._power
    
    def cleanup(self):
        if not self.sim_mode:
            self.stop()
            self.in1.close()
            self.in2.close()
            self.pwm.close()

# =============================================================================
# ENCODER CLASS
# =============================================================================

class NativeEncoder:
    """
    Single-channel encoder using gpiozero Button.
    """
    
    def __init__(self, pin, sim_mode=False, name="encoder", ppr=1000):
        self.name = name
        self.sim_mode = sim_mode
        self.ppr = ppr
        self._count = 0
        self._direction = 1
        self._lock = threading.Lock()
        self._button = None
        
        if self.sim_mode:
            return
        
        self.pin_bcm = self._board_to_bcm(pin)
        
        try:
            from gpiozero import Button
            self._button = Button(self.pin_bcm, pull_up=True)
            self._button.when_pressed = self._pulse_callback
            self._button.when_released = self._pulse_callback
            print(f"  ✓ {name}: GPIO{self.pin_bcm}")
        except Exception as e:
            print(f"  ⚠ {name} init failed: {e}")
    
    def _board_to_bcm(self, board_pin):
        board_to_bcm = {
            3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
            15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11, 24: 8,
            26: 7, 27: 0, 28: 1, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19,
            36: 16, 37: 26, 38: 20, 40: 21
        }
        return board_to_bcm.get(board_pin, board_pin)
    
    def set_direction(self, forward: bool):
        with self._lock:
            self._direction = 1 if forward else -1
    
    def _pulse_callback(self):
        with self._lock:
            self._count += self._direction
    
    def get_position(self):
        with self._lock:
            return self._count / self.ppr
    
    def get_count(self):
        with self._lock:
            return self._count
    
    def reset(self):
        with self._lock:
            self._count = 0
    
    def cleanup(self):
        if self._button:
            self._button.close()

# =============================================================================
# IMU CLASS
# =============================================================================

class NativeIMU:
    """
    Direct I2C access to MPU6050.
    """
    
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    
    def __init__(self, bus=1, address=0x68, sim_mode=False, name="imu"):
        self.name = name
        self.bus = bus
        self.address = address
        self.sim_mode = sim_mode
        self._smbus = None
        self._initialized = False
        
        self._heading = 0.0
        self._heading_offset = 0.0
        self._last_update = time.time()
        
        self._gyro_offset = [0.0, 0.0, 0.0]
        self._accel_offset = [0.0, 0.0, 0.0]
        
        self._is_moving = False
        self._motion_threshold = STUCK_ACCEL_THRESHOLD
        
        self._lock = threading.Lock()
        
        if not self.sim_mode:
            self._initialize_hardware()
    
    def _initialize_hardware(self):
        try:
            import smbus2
            self._smbus = smbus2.SMBus(self.bus)
            
            self._smbus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Configure gyro for ±250°/s
            self._smbus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
            
            # Configure accelerometer for ±2g
            self._smbus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)
            
            self._initialized = True
            print(f"  ✓ {self.name} initialized on I2C bus {self.bus}, addr 0x{self.address:02x}")
            
            self._calibrate()
            
        except ImportError:
            print(f"  ✗ {self.name}: smbus2 not installed")
            self._initialized = False
        except Exception as e:
            print(f"  ✗ {self.name}: I2C error - {e}")
            self._initialized = False
    
    def _calibrate(self, samples=100):
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
        self._accel_offset = [accel_sum[0] / samples, accel_sum[1] / samples, 0.0]
        
        print(f"  ✓ {self.name} calibrated")
    
    def _read_raw_word(self, reg):
        high = self._smbus.read_byte_data(self.address, reg)
        low = self._smbus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value -= 0x10000
        return value
    
    def _read_raw_gyro(self):
        if not self._initialized:
            return [0.0, 0.0, 0.0]
        try:
            gx = self._read_raw_word(self.GYRO_XOUT_H)
            gy = self._read_raw_word(self.GYRO_XOUT_H + 2)
            gz = self._read_raw_word(self.GYRO_XOUT_H + 4)
            return [gx, gy, gz]
        except:
            return [0.0, 0.0, 0.0]
    
    def _read_raw_accel(self):
        if not self._initialized:
            return [0.0, 0.0, 16384.0]
        try:
            ax = self._read_raw_word(self.ACCEL_XOUT_H)
            ay = self._read_raw_word(self.ACCEL_XOUT_H + 2)
            az = self._read_raw_word(self.ACCEL_XOUT_H + 4)
            return [ax, ay, az]
        except:
            return [0.0, 0.0, 16384.0]
    
    def get_gyro(self):
        raw = self._read_raw_gyro()
        gx = (raw[0] - self._gyro_offset[0]) / IMU_GYRO_SCALE
        gy = (raw[1] - self._gyro_offset[1]) / IMU_GYRO_SCALE
        gz = (raw[2] - self._gyro_offset[2]) / IMU_GYRO_SCALE
        return (gx, gy, gz)
    
    def get_accel(self):
        raw = self._read_raw_accel()
        ax = (raw[0] - self._accel_offset[0]) / 16384.0
        ay = (raw[1] - self._accel_offset[1]) / 16384.0
        az = raw[2] / 16384.0
        return (ax, ay, az)
    
    def update(self):
        if not self._initialized:
            return
        
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        if dt > 0.5:
            return
        
        _, _, gz = self.get_gyro()
        
        with self._lock:
            self._heading += np.radians(gz) * dt
            self._heading = np.arctan2(np.sin(self._heading), np.cos(self._heading))
        
        ax, ay, az = self.get_accel()
        accel_magnitude = np.sqrt(ax*ax + ay*ay)
        self._is_moving = accel_magnitude > self._motion_threshold
    
    def get_heading(self):
        with self._lock:
            return self._heading - self._heading_offset
    
    def reset_heading(self):
        with self._lock:
            self._heading_offset = self._heading
    
    def get_tilt(self):
        ax, ay, az = self.get_accel()
        pitch = np.degrees(np.arctan2(ax, np.sqrt(ay*ay + az*az)))
        roll = np.degrees(np.arctan2(ay, np.sqrt(ax*ax + az*az)))
        return (pitch, roll)
    
    def is_tilted_unsafe(self):
        pitch, roll = self.get_tilt()
        return abs(pitch) > MAX_TILT_DEGREES or abs(roll) > MAX_TILT_DEGREES
    
    def is_moving(self):
        return self._is_moving
    
    def cleanup(self):
        if self._smbus:
            self._smbus.close()

# =============================================================================
# CAMERA CLASS
# =============================================================================

class Picamera2Driver:
    """
    Advanced Camera Driver using Picamera2 for Zone Focusing.
    """
    def __init__(self, width=1280, height=720, sim_mode=False):
        self.width = width
        self.height = height
        self.sim_mode = sim_mode
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self.picam2 = None
        
        if self.sim_mode or not HAS_PICAM2:
            print("  ⚠ Picamera2Driver in SIM MODE (Missing lib or --sim)")
            return

        try:
            print(f"  ⚡ Initializing Picamera2 ({width}x{height})...")
            self.picam2 = Picamera2()
            
            # Configure for video (BGR format for OpenCV compatibility)
            config = self.picam2.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            # Set Initial Focus to Infinity (0.0)
            self.set_focus(0.0)
            
            print(f"  ✓ Picamera2 started")
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            
        except Exception as e:
            print(f"  ✗ Picamera2 init failed: {e}")
            self.picam2 = None

    def set_focus(self, position):
        """
        Set lens position manually (Zone Focusing).
        0.0 = Infinity
        1.0 ≈ 1 meter
        4.0 ≈ 25 cm
        10.0 = Macro (Closest)
        """
        if self.picam2:
            try:
                # Set Manual Focus Mode and Position
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": float(position)
                })
                print(f"  🔍 Focus set to: {position}")
            except Exception as e:
                print(f"  ⚠ Focus error: {e}")

    def _capture_loop(self):
        import cv2
        while self._running and self.picam2:
            try:
                # Capture latest frame as numpy array
                frame = self.picam2.capture_array()
                if frame is not None:
                    # Fix Blue/Red swap (Picamera2 returns RGB, OpenCV expects BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    with self._frame_lock:
                        self._frame = frame
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        if self.sim_mode or not self.picam2:
            # Return dummy frame if sim/failed
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(img, "NO CAM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            return img
            
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def cleanup(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()

class NativeCamera:
    """
    Camera access supporting both CSI and USB cameras.
    """
    
    def __init__(self, device=0, width=1280, height=720, sim_mode=False):
        self.width = width
        self.height = height
        self.sim_mode = sim_mode
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._use_libcamera = False
        self._libcamera_proc = None
        self.cap = None
        
        if self.sim_mode:
            return
        
        try:
            import subprocess
            import shutil
            import cv2
            
            vid_cmd = None
            if shutil.which('rpicam-vid'):
                vid_cmd = 'rpicam-vid'
            elif shutil.which('libcamera-vid'):
                vid_cmd = 'libcamera-vid'
            
            if vid_cmd:
                cmd = [
                    vid_cmd,
                    '-t', '0',
                    '--width', str(width),
                    '--height', str(height),
                    '--framerate', '30',
                    '--codec', 'yuv420',
                    '--autofocus-mode', 'manual',
                    '--lens-position', '0.0',
                    '-n',
                    '-o', '-'
                ]
                
                self._libcamera_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=width * height * 3 // 2
                )
                
                self._use_libcamera = True
                self._yuv_size = width * height * 3 // 2
                print(f"  ✓ Camera ({vid_cmd}): {width}x{height}")
            else:
                raise Exception("rpicam-vid/libcamera-vid not found")
                
        except Exception as e:
            print(f"  ⚠ libcamera failed: {e}")
            print(f"  → Trying OpenCV fallback...")
            
            import cv2
            if isinstance(device, str):
                self.cap = cv2.VideoCapture(device)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(device)
            
            if not self.cap.isOpened():
                print("  ✗ Failed to open camera")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"  ✓ Camera (OpenCV): {width}x{height}")
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def _capture_loop(self):
        import cv2
        frame_errors = 0
        frame_ok_count = 0
        
        while self._running:
            try:
                if self._use_libcamera and self._libcamera_proc:
                    yuv_data = self._libcamera_proc.stdout.read(self._yuv_size)
                    if len(yuv_data) == self._yuv_size:
                        yuv_array = np.frombuffer(yuv_data, dtype=np.uint8).reshape(
                            (self.height * 3 // 2, self.width)
                        )
                        frame = cv2.cvtColor(yuv_array, cv2.COLOR_YUV2BGR_I420)
                        with self._frame_lock:
                            self._frame = frame
                        frame_ok_count += 1
                        frame_errors = 0
                    else:
                        frame_errors += 1
                        
                elif self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        with self._frame_lock:
                            self._frame = frame
                        frame_ok_count += 1
                        frame_errors = 0
                    else:
                        frame_errors += 1
                else:
                    time.sleep(0.1)
                    continue
                    
            except Exception as e:
                frame_errors += 1
            
            time.sleep(0.001)
    
    def get_frame(self):
        if self.sim_mode:
            import cv2
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "SIM MODE", (self.width//2 - 80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None
    
    def cleanup(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._libcamera_proc:
            try:
                self._libcamera_proc.terminate()
            except:
                pass
        if self.cap:
            self.cap.release()

# =============================================================================
# LIDAR CLASS
# =============================================================================

class NativeLidar:
    """
    RPLIDAR A1 access via rplidar-roboticia library.
    """
    
    def __init__(self, port="/dev/ttyUSB0", sim_mode=False):
        self.port = port
        self.sim_mode = sim_mode
        self._scans = []
        self._scan_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._lidar = None
        
        if self.sim_mode:
            return
        
        try:
            from rplidar import RPLidar
            self._lidar = RPLidar(port)
            self._lidar.connect()
            
            self._running = True
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            
            print(f"  ✓ LIDAR: {port}")
        except Exception as e:
            print(f"  ⚠ LIDAR init failed: {e}")
            self._lidar = None
    
    def _scan_loop(self):
        try:
            for scan in self._lidar.iter_scans():
                if not self._running:
                    break
                
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
        if self.sim_mode:
            return []
        
        with self._scan_lock:
            return list(self._scans)
    
    def get_points_xy(self):
        scan = self.get_scan()
        points = []
        for angle, dist in scan:
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            points.append([x, y])
        return points
    
    def cleanup(self):
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
# POWER SENSOR CLASS (INA219)
# =============================================================================

class NativePowerSensor:
    """
    INA219 voltage/current sensor via pi_ina219 library.
    """
    
    # Battery voltage thresholds (4S LiPo)
    VOLTAGE_FULL = 16.0  # 4.0V per cell
    VOLTAGE_EMPTY = 12.5  # ~3.1V per cell (safe cutoff)
    
    def __init__(self, sim_mode=False, name="power"):
        self.name = name
        self.sim_mode = sim_mode
        self._ina = None
        self._initialized = False
        
        # Cached values
        self._voltage = 0.0
        self._current = 0.0
        self._power = 0.0
        self._last_update = 0
        self._update_interval = 0.5  # Update every 500ms
        
        if self.sim_mode:
            print(f"  ⚠ {name}: SIM MODE")
            return
        
        try:
            from ina219 import INA219
            from ina219 import DeviceRangeError
            
            SHUNT_OHMS = 0.1  # Standard INA219 shunt resistor
            MAX_EXPECTED_AMPS = 3.2
            
            self._ina = INA219(SHUNT_OHMS, MAX_EXPECTED_AMPS)
            self._ina.configure()
            
            self._initialized = True
            print(f"  ✓ {name}: INA219 initialized")
            
            # Initial read
            self._update()
            
        except ImportError:
            print(f"  ⚠ {name}: pi_ina219 not installed (pip install pi-ina219)")
        except Exception as e:
            print(f"  ⚠ {name}: Init failed - {e}")
    
    def _update(self):
        """Read current sensor values."""
        if not self._initialized:
            return
        
        try:
            self._voltage = self._ina.voltage()
            self._current = self._ina.current() / 1000.0  # mA to A
            self._power = self._ina.power() / 1000.0  # mW to W
            self._last_update = time.time()
        except Exception as e:
            pass  # Silently ignore read errors
    
    def get_voltage(self):
        """Get battery voltage (V)."""
        if self.sim_mode:
            return 15.2  # Simulated ~60% battery
        
        if time.time() - self._last_update > self._update_interval:
            self._update()
        return self._voltage
    
    def get_current(self):
        """Get current draw (A)."""
        if self.sim_mode:
            return 0.8
        
        if time.time() - self._last_update > self._update_interval:
            self._update()
        return self._current
    
    def get_power(self):
        """Get power consumption (W)."""
        if self.sim_mode:
            return 12.0
        
        if time.time() - self._last_update > self._update_interval:
            self._update()
        return self._power
    
    def get_battery_percentage(self):
        """Calculate battery percentage based on voltage."""
        voltage = self.get_voltage()
        if voltage >= self.VOLTAGE_FULL:
            return 100.0
        elif voltage <= self.VOLTAGE_EMPTY:
            return 0.0
        else:
            return ((voltage - self.VOLTAGE_EMPTY) / (self.VOLTAGE_FULL - self.VOLTAGE_EMPTY)) * 100.0
    
    def get_all(self):
        """Get all power readings as a dict."""
        return {
            "voltage": self.get_voltage(),
            "current": self.get_current(),
            "power": self.get_power(),
            "battery_pct": self.get_battery_percentage()
        }
    
    def cleanup(self):
        pass  # No cleanup needed for I2C

