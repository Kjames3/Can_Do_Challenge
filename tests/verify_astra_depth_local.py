import cv2
import time
import sys

def check_backend_support():
    build_info = cv2.getBuildInformation()
    openni_support = "OpenNI" in build_info or "OpenNI2" in build_info
    obsensor_support = "OBSENSOR" in build_info
    
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"OpenNI2 Support: {openni_support}")
    print(f"Orbbec (OBSENSOR) Support: {obsensor_support}")
    return openni_support, obsensor_support

def try_open_depth(index, backend_id, backend_name):
    print(f"\nAttempting index {index} with {backend_name}...")
    try:
        cap = cv2.VideoCapture(index, backend_id)
    except Exception as e:
        print(f" - Error init: {e}")
        return False

    if not cap.isOpened():
        print(" - Failed to open.")
        return False
        
    # Try to grab a frame
    # For Depth, we often need to set grab mode or channel
    cap.grab()
    ret, frame = cap.retrieve(0, cv2.CAP_OPENNI_DEPTH_MAP)
    
    if ret and frame is not None:
        print(f" - Success! Grayscale Depth Frame: {frame.shape}")
        # Normalize for display
        norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow(f"Depth {backend_name} Idx {index}", norm_frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        return True
    
    # Try standard read if retrieve specific channel failed
    ret, frame = cap.read()
    if ret and frame is not None:
         print(f" - Success (Standard Read)! Frame: {frame.shape}")
         cv2.imshow(f"Stream {backend_name} Idx {index}", frame)
         cv2.waitKey(2000)
         cv2.destroyAllWindows()
         return True
         
    print(" - Opened, but no frame.")
    return False

def main():
    print("========================================")
    print("    Astra Depth Verification (Local)    ")
    print("========================================")
    
    openni, obsensor = check_backend_support()
    
    # Check standard indices again just for Depth specific channel
    # Usually Depth is a virtual index if using OpenNI (e.g. cv2.CAP_OPENNI2 + 0)
    
    print("\n--- Testing Backends ---")
    
    # Test OBSENSOR if available
    # It might treat the camera as index 0 for the backend
    if hasattr(cv2, 'CAP_OBSENSOR'):
        print("\nUsing cv2.CAP_OBSENSOR:")
        try_open_depth(0, cv2.CAP_OBSENSOR, "OBSENSOR")
    else:
        print("\ncv2.CAP_OBSENSOR not found in this OpenCV build.")

    # Test OPENNI2
    if hasattr(cv2, 'CAP_OPENNI2'):
        print("\nUsing cv2.CAP_OPENNI2:")
        # OpenNI usually enumerates devices itself, so Index 0 or 1
        try_open_depth(0, cv2.CAP_OPENNI2, "OPENNI2")
        try_open_depth(1, cv2.CAP_OPENNI2, "OPENNI2")
    else:
        print("\ncv2.CAP_OPENNI2 not found in this OpenCV build.")

    print("\nNote: If these fail, you likely need to install the OpenNI2 SDK and drivers")
    print("separately, or use the 'openni' python package.")
    print("However, getting RGB at Index 2 via standard USB is a GOOD sign.")

if __name__ == "__main__":
    main()
