#!/usr/bin/env python3
"""
Blurry Dataset Collector (Library & Script)

This module provides functionality to capture a sequence of images at varying 
focus levels to train the YOLO model to recognize objects even when out of focus.
It can be run as a standalone script or imported by the server.
"""

import time
import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add parent directory to path to import drivers if running standalone
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from drivers import Picamera2Driver

DEFAULT_SAVE_DIR = "training_images/blur_sweep"

def capture_sweep(camera, save_dir, dataset_index, start=0.0, end=12.0, step=0.5):
    """
    Executes a focus sweep and saves images.
    
    Args:
        camera: The camera driver instance.
        save_dir: Directory to save images.
        dataset_index: Index number for this sweep (e.g. 0, 1, 2).
        start: Starting focus value.
        end: Ending focus value.
        step: Step size for focus iteration.
        
    Returns:
        int: Number of images captured.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    count = 0
    # Include the end value in the range
    focus_values = np.arange(start, end + 0.1, step)
    
    print(f"  📸 Starting sweep #{dataset_index} with {len(focus_values)} steps...")
    
    try:
        for focus_val in focus_values:
            # Set Focus
            camera.set_focus(focus_val)
            time.sleep(0.2) # Wait for lens to move & frame to update
            
            # Capture
            frame = camera.get_frame()
            
            if frame is not None:
                # Save
                filename = f"{save_dir}/sweep_{dataset_index}_focus_{focus_val:.1f}.jpg"
                cv2.imwrite(filename, frame)
                count += 1
                # print(f"    Saved: {filename}")
            else:
                print("    ⚠ Failed to get frame")
                
    except Exception as e:
        print(f"  ⚠ Sweep failed: {e}")
        
    # Reset focus to infinity
    camera.set_focus(0.0)
    return count

def main():
    print("\n" + "="*60)
    print("  BLURRY DATASET COLLECTOR")
    print("="*60)
    
    # Config
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640
    SAVE_DIR = DEFAULT_SAVE_DIR
    
    # Initialize Camera
    print("  Initializing Camera...")
    try:
        cam = Picamera2Driver(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return

    time.sleep(2)  # Warmup
    
    print("\nINSTRUCTIONS:")
    print("1. Place the soda can at a fixed distance (e.g., 20cm).")
    print("2. The robot will sweep focus from 0.0 to 12.0.")
    print("3. Move the can to a new distance/angle and repeat.")
    
    dataset_index = 0
    
    # Check for existing files to auto-increment index
    if os.path.exists(SAVE_DIR):
        existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("sweep_")]
        if existing_files:
            indices = [int(f.split('_')[1]) for f in existing_files]
            if indices:
                dataset_index = max(indices) + 1
    
    try:
        while True:
            action = input(f"\nPress [ENTER] to capture sweep #{dataset_index}, or [Q] to quit: ").strip().upper()
            if action == 'Q':
                break
                
            count = capture_sweep(cam, SAVE_DIR, dataset_index)
            print(f"  ✓ Sweep complete. Captured {count} images.")
            dataset_index += 1
            print("  Move the can!")
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing camera...")
        cam.cleanup()

if __name__ == "__main__":
    main()
