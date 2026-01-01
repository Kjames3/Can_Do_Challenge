#!/usr/bin/env python3
"""
Focus Calibration Tool for IMX708 (Viam Rover)

Usage:
    python3 calibration/calibrate_focus_tool.py

Features:
1. Manual Mode: Enter a lens position to test.
2. Sweep Mode: Automatically scan all focus values to find the sharpest one.
"""

import time
import sys
import os
import cv2
import numpy as np

# Add parent directory to path so we can import drivers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers import Picamera2Driver

# Config
IMAGE_WIDTH = 1536  # Matches your server_native.py
IMAGE_HEIGHT = 864
SAVE_DIR = "focus_test_images"

def calculate_sharpness(image):
    """
    Calculate the sharpness of an image using the Variance of Laplacian method.
    Higher value = Sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a center crop to focus on the object (avoid background noise)
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    crop_size = 300  # 300x300 center box
    
    # Ensure crop implies within bounds
    if crop_size > h or crop_size > w:
        crop_size = min(h, w)
        
    crop = gray[center_h-crop_size//2:center_h+crop_size//2, 
                center_w-crop_size//2:center_w+crop_size//2]
    
    score = cv2.Laplacian(crop, cv2.CV_64F).var()
    return score

def main():
    print("\n" + "="*60)
    print("  IMX708 FOCUS CALIBRATION TOOL")
    print("="*60)
    
    # 1. Initialize Camera
    print("Initializing Camera...")
    # NOTE: Picamera2Driver might fail if libcamera is not available or X server is missing
    # ensuring we handle that gracefully if possible, but driver prints error.
    cam = Picamera2Driver(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    time.sleep(2)  # Warmup
    
    # Check if camera initialized
    if not cam.picam2 and not cam.sim_mode:
        print("Camera failed to initialize. Check connections or run with --sim?")
        # We proceed anyway, but get_frame will return dummy.
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"\nImages will be saved to: {os.path.abspath(SAVE_DIR)}")

    while True:
        print("\n" + "-"*60)
        print("MODES:")
        print("  [M] Manual Entry  (Type a specific float value)")
        print("  [S] Auto-Sweep    (Scan 0.0 to 12.0 to find peak sharpness)")
        print("  [Q] Quit")
        choice = input("Select Mode: ").upper().strip()
        
        if choice == 'Q':
            break
            
        elif choice == 'M':
            try:
                val_str = input("  Enter Lens Position (0.0=Inf, 10.0=Macro): ")
                lens_pos = float(val_str)
                cam.set_focus(lens_pos)
                time.sleep(1.0) # Wait for lens to settle
                
                frame = cam.get_frame()
                score = calculate_sharpness(frame)
                
                filename = f"{SAVE_DIR}/manual_pos_{lens_pos:.1f}_score_{int(score)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  📸 Saved: {filename}")
                print(f"  ✨ Sharpness Score: {score:.1f}")
            except ValueError:
                print("  Invalid number.")

        elif choice == 'S':
            print("\n  STARTING SWEEP (0.0 -> 12.0)...")
            print("  Make sure your target object is CENTERED in the frame.")
            input("  Press ENTER to begin...")
            
            best_score = -1
            best_pos = -1
            
            # Scan range: 0.0 to 12.0 in steps of 0.5
            # IMX708 lens positions typically range 0.0 (inf) to ~15.0 (macro)
            for lens_pos in np.arange(0.0, 12.5, 0.5):
                cam.set_focus(lens_pos)
                time.sleep(0.5) # Settle time
                
                frame = cam.get_frame()
                score = calculate_sharpness(frame)
                
                print(f"    Pos: {lens_pos:>4.1f} | Score: {score:>7.1f} | Bar: {'#' * int(score/50)}")
                
                if score > best_score:
                    best_score = score
                    best_pos = lens_pos
                    # Save the new "best" image
                    cv2.imwrite(f"{SAVE_DIR}/best_sweep_pos_{lens_pos:.1f}.jpg", frame)
            
            print(f"\n  🏆 WINNER: Lens Position {best_pos} (Score: {best_score:.1f})")
            print(f"  (Check {SAVE_DIR} to visually verify the image)")

    print("\nClosing camera...")
    cam.cleanup()

if __name__ == "__main__":
    main()
