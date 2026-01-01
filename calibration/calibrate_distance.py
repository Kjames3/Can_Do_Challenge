#!/usr/bin/env python3
"""
Distance Calibration Script for Viam Rover

This script performs a scientific calibration of the robot's odometry.
1. Drives the robot forward for a set duration.
2. Calculates the distance the robot *thinks* it traveled.
3. Asks the user to input the *actual* measured distance.
4. Calculates the correct WHEEL_CIRCUMFERENCE_MM to perform the calibration.

Usage:
    python3 calibration/calibrate_distance.py
"""

import time
import sys
import os
import argparse
import numpy as np

# Add parent directory to path to import drivers/robot_state
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers import NativeMotor, NativeEncoder, configure_pin_factory
from robot_state import WHEEL_CIRCUMFERENCE_MM, WHEEL_DIAMETER_CM

# Parameters
DRIVE_POWER = 0.4
DRIVE_DURATION = 3.0  # seconds

# COPY GPIO CONFIG FROM server_native.py
LEFT_MOTOR_PIN_A = 35
LEFT_MOTOR_PIN_B = 33
LEFT_MOTOR_PWM = 37

RIGHT_MOTOR_PIN_A = 31
RIGHT_MOTOR_PIN_B = 29
RIGHT_MOTOR_PWM = 15

LEFT_ENCODER_PIN = 38
RIGHT_ENCODER_PIN = 40

def main():
    print("\n" + "="*60)
    print("  DISTANCE CALIBRATION UTILITY")
    print("="*60)
    print(f"  Current Config:")
    print(f"    - Wheel Diameter:      {WHEEL_DIAMETER_CM:.2f} cm")
    print(f"    - Wheel Circumference: {WHEEL_CIRCUMFERENCE_MM:.2f} mm")
    print("-" * 60)

    # 1. Setup
    print("\nInitializing Hardware...")
    if not configure_pin_factory():
        print("⚠ GPIO setup failed. Falling back to simulation (results will be fake).")
        sim_mode = True
    else:
        sim_mode = False

    # Initialize with ppr=12 (Fixed from previous step)
    left_encoder = NativeEncoder(LEFT_ENCODER_PIN, sim_mode=sim_mode, name="left_enc", ppr=12)
    right_encoder = NativeEncoder(RIGHT_ENCODER_PIN, sim_mode=sim_mode, name="right_enc", ppr=12)
    
    left_motor = NativeMotor(LEFT_MOTOR_PIN_A, LEFT_MOTOR_PIN_B, LEFT_MOTOR_PWM, sim_mode=sim_mode, name="left_mot")
    right_motor = NativeMotor(RIGHT_MOTOR_PIN_A, RIGHT_MOTOR_PIN_B, RIGHT_MOTOR_PWM, sim_mode=sim_mode, name="right_mot")
    
    left_motor.set_encoder(left_encoder)
    right_motor.set_encoder(right_encoder)

    # 2. Instructions
    print("\nINSTRUCTIONS:")
    print("1. Place the robot on a flat surface with a Start Line.")
    print("2. Mark the Start Line with tape.")
    print("3. Ensure the robot has enough space to drive forward (~1 meter).")
    input("\nPress ENTER when ready to start driving...")

    # 3. Drive
    print(f"\nDriving forward for {DRIVE_DURATION} seconds...")
    left_encoder.reset()
    right_encoder.reset()
    
    start_time = time.time()
    
    # Simple straight drive loop
    while time.time() - start_time < DRIVE_DURATION:
        # Drift compensation (if needed, simplified here)
        left_motor.set_power(DRIVE_POWER)  
        right_motor.set_power(DRIVE_POWER)
        time.sleep(0.01)
        
    left_motor.stop()
    right_motor.stop()
    
    time.sleep(0.5) # Wait for stop

    # 4. Calculate Robot Distance
    # Ticks are already converted to revolutions inside NativeEncoder (if get_position used)
    # But let's check exact count to be transparent
    l_revs = left_encoder.get_position()
    r_revs = right_encoder.get_position()
    avg_revs = (l_revs + r_revs) / 2.0
    
    # Distance = Revolutions * Circumference
    robot_dist_mm = avg_revs * WHEEL_CIRCUMFERENCE_MM
    robot_dist_cm = robot_dist_mm / 10.0
    
    print("\n" + "-"*60)
    print("  ROBOT REPORT:")
    print(f"    Left Revolutions:  {l_revs:.2f}")
    print(f"    Right Revolutions: {r_revs:.2f}")
    print(f"    Avg Revolutions:   {avg_revs:.2f}")
    print(f"    Calculated Dist:   {robot_dist_cm:.2f} cm")
    print("-" * 60)

    # 5. User Measurement
    print("\nACTION REQUIRED:")
    print(f"  Please measure the ACTUAL distance traveled from the start line.")
    
    while True:
        try:
            actual_dist_str = input("  Enter ACTUAL distance (in cm): ")
            actual_dist_cm = float(actual_dist_str)
            if actual_dist_cm <= 0:
                print("  Please enter a positive number.")
                continue
            break
        except ValueError:
            print("  Invalid input. Please enter a number (e.g., 55.0).")

    # 6. Calculate Correction
    if robot_dist_cm == 0:
        print("\n⚠ Error: Robot thinks it didn't move. Cannot calculate correction.")
        print("  Check encoder connections or ppr settings.")
    else:
        # Formula: New = Old * (Actual / Robot)
        correction_factor = actual_dist_cm / robot_dist_cm
        new_circumference_mm = WHEEL_CIRCUMFERENCE_MM * correction_factor
        
        # Also calculate new diameter for reference
        new_diameter_cm = (new_circumference_mm / np.pi) / 10.0

        print("\n" + "="*60)
        print("  CALIBRATION RESULTS")
        print("="*60)
        print(f"  Old Circumference: {WHEEL_CIRCUMFERENCE_MM:.2f} mm")
        print(f"  Actual Distance:   {actual_dist_cm:.2f} cm")
        print(f"  Robot Distance:    {robot_dist_cm:.2f} cm")
        print(f"  Correction Factor: {correction_factor:.4f}")
        print("-" * 60)
        print(f"  NEW CIRCUMFERENCE: {new_circumference_mm:.2f} mm")
        print(f"  (Equiv Diameter:   {new_diameter_cm:.2f} cm)")
        print("="*60)
        
        print("\nTO APPLY FIX:")
        print(f"1. Open 'robot_state.py'")
        print(f"2. Update line 15 to:")
        print(f"     WHEEL_CIRCUMFERENCE_MM = {new_circumference_mm:.2f}")
        print("   (Or obtain it from diameter: WHEEL_DIAMETER_CM = {new_diameter_cm:.2f})")

    # Cleanup
    left_motor.cleanup()
    right_motor.cleanup()
    left_encoder.cleanup()
    right_encoder.cleanup()

if __name__ == "__main__":
    main()
