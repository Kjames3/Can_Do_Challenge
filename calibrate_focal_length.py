"""
Focal Length Calibration Script for Viam Rover

This script helps calculate the correct FOCAL_LENGTH value for distance estimation.
It uses the YOLO model to detect a standard object (Coke can) and calculates
the focal length based on its known height and a known distance.

Usage:
    1. Place a standard Coke can (12cm height) exactly 50cm away from the camera.
    2. Run this script: python calibrate_focal_length.py
    3. Ensure the can is detected (green box).
    4. Press SPACE to capture and calculate.
    5. Update FOCAL_LENGTH in server_native.py with the result.
"""

import time
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from drivers import NativeCamera, configure_pin_factory

# Constants
KNOWN_HEIGHT_CAN_CM = 12.0  # Standard 12oz/355ml Coke can
DEFAULT_TEST_DIST_CM = 50.0 # Standard calibration distance

# Camera Settings (Must match server_native.py)
# NEW (Update to match your target resolution)
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 864

def main():
    parser = argparse.ArgumentParser(description='Calibrate Camera Focal Length')
    parser.add_argument('--dist', type=float, default=DEFAULT_TEST_DIST_CM,
                        help=f'Distance to object in cm (default: {DEFAULT_TEST_DIST_CM})')
    parser.add_argument('--height', type=float, default=KNOWN_HEIGHT_CAN_CM,
                        help=f'Real height of object in cm (default: {KNOWN_HEIGHT_CAN_CM})')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  CAMERA FOCAL LENGTH CALIBRATION")
    print("="*60)
    print(f"  Target Object: Standard Can ({args.height} cm)")
    print(f"  Target Distance: {args.dist} cm")
    print("-" * 60)
    
    # 1. Setup GPIO/Camera
    if not configure_pin_factory():
        print("⚠ Warning: GPIO setup failed (running in limited mode)")
        
    print("\nInitializing Camera...")
    camera = NativeCamera(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    # Allow camera to warm up
    time.sleep(2.0)
    
    # 2. Load Model
    print("Loading YOLO Model...")
    model_name = 'yolo11n_cans.onnx'
    try:
        model = YOLO(model_name)
    except Exception:
        print(f"⚠ Failed to load {model_name}, trying fallback...")
        model = YOLO('yolo11n_cans.pt')
    
    print("\n" + "="*60)
    print("  INSTRUCTIONS:")
    print("  1. Place the Coke can upright on the floor.")
    print(f"  2. Measure exactly {args.dist} cm from the camera lens to the can.")
    print("  3. Ensure the green box detections are stable.")
    print("  4. Press [SPACE] to capture and calculate.")
    print("  5. Press [Q] to quit without saving.")
    print("="*60 + "\n")
    
    running = True
    samples = []
    
    while running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Run detection
        results = model.predict(frame, conf=0.4, verbose=False, classes=[0])
        
        can_detected = False
        height_px = 0
        
        for r in results:
            boxes = r.boxes
            # Find largest box (presumably the closest can)
            best_box = None
            max_area = 0
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = box
            
            if best_box:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                height_px = y2 - y1
                can_detected = True
                
                # Draw box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Display live pixel height
                label = f"H: {height_px:.1f}px"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # UI Overlay
        cv2.putText(frame, f"Target Dist: {args.dist}cm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if can_detected:
            # Calculate provisional focal length
            curr_focal = (height_px * args.dist) / args.height
            cv2.putText(frame, f"Est. Focal: {curr_focal:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to Calculate", (10, IMAGE_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "NO CAN DETECTED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        # Note: server environment might not support cv2.imshow
        # We will save snapshot on calculation instead of relying on GUI window
        # But for logic we assume user might be running this in a way they can see output 
        # or we just rely on console print after capture.
        # Since this is "server native", usually headless.
        # Let's add logic to print status to console periodically.
        
        # Display logic for headless:
        # We can't use cv2.imshow on a headless Pi easily via SSH.
        # We'll rely on console output.
        pass
        
        # Since we can't easily show window, we'll auto-capture if stable for 2 seconds?
        # Or just wait for ENTER key in console?
        # Actually, let's look for a keypress if possible, otherwise input() in a separate thread?
        # Since cv2.waitKey requires a window, we can't use it headlessly.
        
        # PLAN B: Simple Console Interaction
        # We will take 10 samples immediately once a stable detection is found.
    
    camera.cleanup()

# Redefining Main for Headless Execution
def main_headless():
    parser = argparse.ArgumentParser(description='Calibrate Camera Focal Length (Headless)')
    parser.add_argument('--dist', type=float, default=DEFAULT_TEST_DIST_CM,
                        help=f'Distance to object in cm (default: {DEFAULT_TEST_DIST_CM})')
    parser.add_argument('--height', type=float, default=KNOWN_HEIGHT_CAN_CM,
                        help=f'Real height of object in cm (default: {KNOWN_HEIGHT_CAN_CM})')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CAMERA FOCAL LENGTH CALIBRATION (HEADLESS)")
    print("="*60)
    print(f"  Target Object: Standard Can ({args.height} cm)")
    print(f"  Target Distance: {args.dist} cm")
    print("-" * 60)
    
    if not configure_pin_factory():
        print("⚠ Warning: GPIO setup failed")
        
    print("\nInitializing Camera...")
    camera = NativeCamera(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    time.sleep(2.0)
    
    print("Loading YOLO Model...")
    try:
        model = YOLO('yolo11n_cans.onnx')
    except:
        model = YOLO('yolo11n_cans.pt')
    
    print("\n" + "="*60)
    print("  INSTRUCTIONS:")
    print(f"  1. Place the Coke can exactly {args.dist} cm away.")
    print("  2. The script will take 20 samples once it sees the can.")
    print("  3. Make sure the can is clearly visible and upright.")
    print("="*60 + "\n")
    
    input("Press Enter to start sampling...")
    print("Sampling...")
    
    samples = []
    required_samples = 20
    
    start_time = time.time()
    
    while len(samples) < required_samples:
        if time.time() - start_time > 30:
            print("Timeout: Could not collect enough samples.")
            break
            
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        results = model.predict(frame, conf=0.4, verbose=False, classes=[0])
        
        detected = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                height_px = y2 - y1
                
                # Filter small noise
                if height_px > 20: 
                    samples.append(height_px)
                    print(f"  Sample {len(samples)}/{required_samples}: Height = {height_px:.1f} px", end='\r')
                    detected = True
                    break # Take one per frame
        
        if not detected:
            print(f"  Searching... (Seen {len(samples)})", end='\r')
            
        time.sleep(0.1)
    
    camera.cleanup()
    
    print("\n" + "-"*60)
    if len(samples) > 0:
        avg_height_px = sum(samples) / len(samples)
        
        # Focal Length Formula: F = (P * D) / H
        # P = Pixel Height, D = Distance, H = Real Height
        focal_length = (avg_height_px * args.dist) / args.height
        
        print("\nRESULTS:")
        print(f"  Average Pixel Height: {avg_height_px:.1f} px")
        print(f"  Real Distance:        {args.dist} cm")
        print(f"  Real Height:          {args.height} cm")
        print("-" * 30)
        print(f"  CALCULATED FOCAL LENGTH: {focal_length:.1f}")
        print("-" * 30)
        print("\nACTION:")
        print(f"  Update 'server_native.py' line 136:")
        print(f"  FOCAL_LENGTH = {focal_length:.0f}")
        print("="*60)
    else:
        print("Failed to detect object.")

if __name__ == "__main__":
    main_headless()
