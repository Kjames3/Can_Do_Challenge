import time
import argparse
import numpy as np
import cv2
import sys
from collections import deque

# Import drivers
try:
    from drivers import (
        NativeEncoder, NativeIMU, NativeLidar, Picamera2Driver, NativeCamera,
        configure_pin_factory,
        IMU_I2C_BUS, IMU_I2C_ADDRESS
    )
except ImportError as e:
    print(f"Error importing drivers: {e}")
    sys.exit(1)

# =============================================================================
# CONSTANTS (Matched to server_native.py)
# =============================================================================
# Encoders
LEFT_ENCODER_PIN = 38
RIGHT_ENCODER_PIN = 40

# Lidar
LIDAR_PORT = "/dev/ttyUSB0"

# Camera
IMAGE_WIDTH = 640  # Lower res for faster processing in stats loop
IMAGE_HEIGHT = 480

# Monitoring Settings
WINDOW_SIZE = 50  # Number of samples for sliding window statistics

# =============================================================================
# UTILITIES
# =============================================================================
class RollingStats:
    def __init__(self, window_size=WINDOW_SIZE, label="Value"):
        self.window_size = window_size
        self.label = label
        self.data = deque(maxlen=window_size)
        
    def update(self, value):
        if value is not None:
            self.data.append(value)
            
    def get_stats(self):
        if not self.data:
            return 0.0, 0.0, 0.0, 0.0
        
        arr = np.array(self.data)
        mean = np.mean(arr)
        std = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        return mean, std, min_val, max_val

def print_header():
    print(f"\n{'SENSOR':<15} | {'METRIC':<15} | {'MEAN':<10} | {'STD DEV (NOISE)':<15} | {'MIN':<10} | {'MAX':<10}")
    print("-" * 85)

def print_row(sensor, metric, mean, std, min_val, max_val):
    print(f"{sensor:<15} | {metric:<15} | {mean:>10.4f} | {std:>15.4f} | {min_val:>10.4f} | {max_val:>10.4f}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Sensor Noise Monitor')
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
    args = parser.parse_args()
    
    sim_mode = args.sim
    
    if not sim_mode:
        if not configure_pin_factory():
            print("Falling back to simulation mode due to GPIO failure.")
            sim_mode = True

    print("\nInitializing Sensors...")
    
    # --- Initialize IMU ---
    imu = NativeIMU(IMU_I2C_BUS, IMU_I2C_ADDRESS, sim_mode=sim_mode)
    
    # --- Initialize Encoders ---
    left_encoder = NativeEncoder(LEFT_ENCODER_PIN, sim_mode=sim_mode, name="Left Enc")
    right_encoder = NativeEncoder(RIGHT_ENCODER_PIN, sim_mode=sim_mode, name="Right Enc")
    
    # --- Initialize Lidar ---
    lidar = NativeLidar(LIDAR_PORT, sim_mode=sim_mode)
    
    # --- Initialize Camera ---
    # Try Picamera2 first, then NativeCamera
    camera = Picamera2Driver(IMAGE_WIDTH, IMAGE_HEIGHT, sim_mode=sim_mode)
    if not args.sim and (camera.picam2 is None):
        print("Falling back to standard NativeCamera driver...")
        camera = NativeCamera(0, IMAGE_WIDTH, IMAGE_HEIGHT, sim_mode=sim_mode)

    # --- Stats Containers ---
    # IMU
    accel_x_stats = RollingStats(label="Accel X")
    accel_y_stats = RollingStats(label="Accel Y")
    accel_z_stats = RollingStats(label="Accel Z")
    gyro_z_stats = RollingStats(label="Gyro Z")
    
    # Encoders (Monitoring count changes per loop - should be 0 if still)
    left_enc_stats = RollingStats(label="Left Pos")
    right_enc_stats = RollingStats(label="Right Pos")
    
    # Lidar (Monitoring distance to nearest object in front)
    lidar_front_stats = RollingStats(label="Front Dist")
    
    # Camera (Monitoring average brightness stability)
    cam_bright_stats = RollingStats(label="Brightness")

    print("\nStarting Monitoring Loop... Press Ctrl+C to stop.")
    print("NOTE: Keep the robot STATIONARY for accurate noise measurement.\n")
    time.sleep(2)

    try:
        while True:
            # 1. Update IMU
            if imu:
                imu.update() # Update heading integration
                ax, ay, az = imu.get_accel()
                gx, gy, gz = imu.get_gyro()
                
                accel_x_stats.update(ax)
                accel_y_stats.update(ay)
                accel_z_stats.update(az)
                gyro_z_stats.update(gz)

            # 2. Update Encoders
            if left_encoder:
                left_enc_stats.update(left_encoder.get_position())
            if right_encoder:
                right_enc_stats.update(right_encoder.get_position())

            # 3. Update Lidar
            if lidar:
                scan = lidar.get_scan()
                # Filter for front sector (-10 to +10 degrees)
                front_dists = [d for angle, d in scan if -0.17 < angle < 0.17] 
                if front_dists:
                    # Take average of front sector to be robust, or just nearest
                    avg_front = np.mean(front_dists)
                    lidar_front_stats.update(avg_front)
            
            # 4. Update Camera
            if camera:
                frame = camera.get_frame()
                if frame is not None:
                    # Calculate average brightness (V in HSV or just mean of grayscale)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray)
                    cam_bright_stats.update(avg_brightness)

            # --- Display ---
            # Use ANSI escape codes to clear screen and reset cursor (simpler than curses)
            print("\033[2J\033[H", end="") # Clear screen, Go Home
            print("SENSOR NOISE MONITOR (Ctrl+C to Quit)")
            print(f"Sampling Window: {WINDOW_SIZE} frames")
            print("=" * 85)
            print_header()
            
            # IMU Rows
            for label, stat in [("IMU Accel X", accel_x_stats), 
                                ("IMU Accel Y", accel_y_stats), 
                                ("IMU Accel Z", accel_z_stats), 
                                ("IMU Gyro Z", gyro_z_stats)]:
                mean, std, min_v, max_v = stat.get_stats()
                print_row("IMU", label, mean, std, min_v, max_v)
            
            print("-" * 85)
            
            # Encoder Rows
            for label, stat in [("Left Enc", left_enc_stats), ("Right Enc", right_enc_stats)]:
                mean, std, min_v, max_v = stat.get_stats()
                # Std Dev > 0 on stationary encoder means electrical noise triggering interrupts
                print_row("Encoder", label, mean, std, min_v, max_v)
                
            print("-" * 85)

            # Lidar Rows
            mean, std, min_v, max_v = lidar_front_stats.get_stats()
            print_row("Lidar", "Front Dist (m)", mean, std, min_v, max_v)
            
            print("-" * 85)
            
            # Camera Rows
            mean, std, min_v, max_v = cam_bright_stats.get_stats()
            print_row("Camera", "Brightness (0-255)", mean, std, min_v, max_v)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        if imu: imu.cleanup()
        if left_encoder: left_encoder.cleanup()
        if right_encoder: right_encoder.cleanup()
        if lidar: lidar.cleanup()
        if camera: camera.cleanup()
        print("Sensors released.")

if __name__ == "__main__":
    main()
