#!/usr/bin/env python3
"""
Motor Calibration Script for Viam Rover

This script finds the minimum motor power needed to overcome friction
and actually move the robot. It uses the encoders to detect movement.

Usage:
    python calibrate_motors.py [--step 0.01] [--max 0.5] [--duration 0.5]
"""

import argparse
import time
import sys

# Check if running on Raspberry Pi
try:
    import RPi.GPIO as GPIO
    SIM_MODE = False
except ImportError:
    print("⚠ RPi.GPIO not available - running in simulation mode")
    GPIO = None
    SIM_MODE = True


# =============================================================================
# HARDWARE CONFIGURATION (from server_native.py)
# =============================================================================

# Motor pins (BCM numbering)
LEFT_MOTOR_PIN_A = 23
LEFT_MOTOR_PIN_B = 24
LEFT_MOTOR_PWM = 12

RIGHT_MOTOR_PIN_A = 27
RIGHT_MOTOR_PIN_B = 22
RIGHT_MOTOR_PWM = 13

# Encoder pins
LEFT_ENCODER_PIN = 4
RIGHT_ENCODER_PIN = 17

# PWM frequency
PWM_FREQUENCY = 1000


# =============================================================================
# HARDWARE CLASSES (simplified from server_native.py)
# =============================================================================

class CalibrationMotor:
    """Simplified motor class for calibration"""
    
    def __init__(self, pin_a, pin_b, pwm_pin, name):
        self.name = name
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.pwm_pin = pwm_pin
        self._power = 0.0
        self.pwm = None
        
        if not SIM_MODE and GPIO:
            GPIO.setup(pin_a, GPIO.OUT)
            GPIO.setup(pin_b, GPIO.OUT)
            GPIO.setup(pwm_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(pwm_pin, PWM_FREQUENCY)
            self.pwm.start(0)
            print(f"  ✓ {name} initialized")
        else:
            print(f"  ✓ {name} (simulated)")
    
    def set_power(self, power):
        """Set motor power (-1.0 to 1.0)"""
        self._power = max(-1.0, min(1.0, power))
        
        if SIM_MODE or not GPIO:
            return
        
        if power >= 0:
            GPIO.output(self.pin_a, GPIO.HIGH)
            GPIO.output(self.pin_b, GPIO.LOW)
        else:
            GPIO.output(self.pin_a, GPIO.LOW)
            GPIO.output(self.pin_b, GPIO.HIGH)
        
        duty = abs(self._power) * 100
        self.pwm.ChangeDutyCycle(duty)
    
    def stop(self):
        self.set_power(0)
        if not SIM_MODE and GPIO:
            GPIO.output(self.pin_a, GPIO.LOW)
            GPIO.output(self.pin_b, GPIO.LOW)
    
    def cleanup(self):
        self.stop()
        if self.pwm:
            self.pwm.stop()


class CalibrationEncoder:
    """Simplified encoder class for calibration"""
    
    def __init__(self, pin, name):
        self.name = name
        self.pin = pin
        self._count = 0
        self._last_state = 0
        
        if not SIM_MODE and GPIO:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(pin, GPIO.BOTH, callback=self._callback)
            print(f"  ✓ {name} initialized")
        else:
            print(f"  ✓ {name} (simulated)")
    
    def _callback(self, channel):
        self._count += 1
    
    def get_count(self):
        return self._count
    
    def reset(self):
        self._count = 0
    
    def cleanup(self):
        if not SIM_MODE and GPIO:
            try:
                GPIO.remove_event_detect(self.pin)
            except:
                pass


# =============================================================================
# CALIBRATION LOGIC
# =============================================================================

def calibrate_motor(motor, encoder, direction, step_size, max_power, hold_duration):
    """
    Find minimum power to move a motor.
    
    Args:
        motor: Motor instance
        encoder: Encoder instance
        direction: 1 for forward, -1 for backward
        step_size: Power increment per step
        max_power: Maximum power to try
        hold_duration: How long to hold each power level (seconds)
    
    Returns:
        Minimum power that caused movement, or None if no movement detected
    """
    dir_name = "FORWARD" if direction > 0 else "BACKWARD"
    print(f"\n  Testing {motor.name} {dir_name}...")
    
    power = 0.0
    
    while power <= max_power:
        # Reset encoder
        encoder.reset()
        initial_count = encoder.get_count()
        
        # Apply power
        motor.set_power(power * direction)
        
        # Wait for movement
        time.sleep(hold_duration)
        
        # Check encoder
        final_count = encoder.get_count()
        movement = final_count - initial_count
        
        # Stop motor
        motor.stop()
        
        if SIM_MODE:
            # Simulate movement detection above threshold
            movement = 5 if power >= 0.15 else 0
        
        print(f"    Power: {power:.2f} | Encoder ticks: {movement}", end="")
        
        if movement > 2:  # Threshold for "real" movement
            print(" ← MOVEMENT DETECTED!")
            return power
        else:
            print()
        
        power += step_size
        time.sleep(0.1)  # Brief pause between tests
    
    return None


def run_calibration(step_size=0.01, max_power=0.5, hold_duration=0.5):
    """Run full motor calibration"""
    
    print("\n" + "="*60)
    print("  MOTOR CALIBRATION SCRIPT")
    print("  Finding minimum power to overcome friction")
    print("="*60)
    
    if SIM_MODE:
        print("\n⚠ SIMULATION MODE - Results are simulated\n")
    
    # Initialize GPIO
    if not SIM_MODE and GPIO:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
    
    print("\nInitializing hardware...")
    
    # Create motor and encoder instances
    left_motor = CalibrationMotor(LEFT_MOTOR_PIN_A, LEFT_MOTOR_PIN_B, LEFT_MOTOR_PWM, "left_motor")
    right_motor = CalibrationMotor(RIGHT_MOTOR_PIN_A, RIGHT_MOTOR_PIN_B, RIGHT_MOTOR_PWM, "right_motor")
    left_encoder = CalibrationEncoder(LEFT_ENCODER_PIN, "left_encoder")
    right_encoder = CalibrationEncoder(RIGHT_ENCODER_PIN, "right_encoder")
    
    print("\n" + "-"*60)
    print("Starting calibration...")
    print(f"  Step size: {step_size}")
    print(f"  Max power: {max_power}")
    print(f"  Hold duration: {hold_duration}s")
    print("-"*60)
    
    results = {}
    
    try:
        # Test left motor forward
        results["left_forward"] = calibrate_motor(
            left_motor, left_encoder, 1, step_size, max_power, hold_duration
        )
        
        # Test left motor backward
        results["left_backward"] = calibrate_motor(
            left_motor, left_encoder, -1, step_size, max_power, hold_duration
        )
        
        # Test right motor forward
        results["right_forward"] = calibrate_motor(
            right_motor, right_encoder, 1, step_size, max_power, hold_duration
        )
        
        # Test right motor backward
        results["right_backward"] = calibrate_motor(
            right_motor, right_encoder, -1, step_size, max_power, hold_duration
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠ Calibration interrupted!")
    
    finally:
        # Cleanup
        left_motor.cleanup()
        right_motor.cleanup()
        left_encoder.cleanup()
        right_encoder.cleanup()
        
        if not SIM_MODE and GPIO:
            GPIO.cleanup()
    
    # Print results
    print("\n" + "="*60)
    print("  CALIBRATION RESULTS")
    print("="*60)
    
    for key, value in results.items():
        if value is not None:
            print(f"  {key:20s}: {value:.2f}")
        else:
            print(f"  {key:20s}: NOT FOUND (try increasing max_power)")
    
    # Calculate recommendations
    valid_results = [v for v in results.values() if v is not None]
    
    if valid_results:
        min_power = max(valid_results)  # Use highest minimum as the safe threshold
        recommended = min_power + 0.02  # Add small margin
        
        print("\n" + "-"*60)
        print(f"  RECOMMENDED MIN_MOVING_POWER: {recommended:.2f}")
        print("-"*60)
        print("\n  To apply, update in navigation_fsm.py:")
        print(f"    MIN_MOVING_POWER = {recommended:.2f}")
    else:
        print("\n  ⚠ No movement detected - check hardware connections")
    
    print("\n✓ Calibration complete!\n")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate motors to find minimum power threshold"
    )
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="Power increment per step (default: 0.01)"
    )
    parser.add_argument(
        "--max", type=float, default=0.5,
        help="Maximum power to test (default: 0.5)"
    )
    parser.add_argument(
        "--duration", type=float, default=0.5,
        help="How long to hold each power level in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    run_calibration(
        step_size=args.step,
        max_power=args.max,
        hold_duration=args.duration
    )
