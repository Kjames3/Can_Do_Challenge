import cv2
import time

def test_camera(index):
    print(f"Testing Camera Index {index}...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Index {index}: Failed to open.")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Index {index}: Opened, but failed to read frame.")
        cap.release()
        return False
        
    print(f"Index {index}: Success! Resolution {frame.shape[1]}x{frame.shape[0]}")
    
    print(f"Press 'q' to close the window for Camera {index}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow(f"Camera Index {index}", frame)
        
        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyWindow(f"Camera Index {index}")
    return True

def main():
    print("========================================")
    print("    Astra Camera Local Verification     ")
    print("========================================")
    print("This script will attempt to open video devices (0, 1, 2...)")
    print("The Astra usually shows up as 2 devices (RGB and IR/Depth)")
    print("Press 'q' in the window to move to the next camera.\n")
    
    found_any = False
    for i in range(5):
        if test_camera(i):
            found_any = True
            
    if not found_any:
        print("\nNo cameras found! Check USB connection.")
    else:
        print("\nTest complete.")

if __name__ == "__main__":
    main()
