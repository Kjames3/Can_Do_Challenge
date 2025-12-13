import cv2
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
# Real-world height of the objects in centimeters (approximate)
# A standard soda can is ~12cm, a standard water bottle is ~20cm
KNOWN_HEIGHT_CAN = 12.0  
KNOWN_HEIGHT_BOTTLE = 20.0

# Focal length (F) will need to be calibrated for your specific webcam.
# Formula to find F: (Pixels_Height * Real_Distance) / Real_Height
# For now, we estimate a standard laptop webcam focal length value.
FOCAL_LENGTH = 600  

# --- LOAD MODEL ---
# 'yolov8n.pt' is the "nano" version. It is the fastest and runs easily on laptops.
# It will download automatically on the first run.
model = YOLO('yolov8n.pt')

# Target Classes in COCO dataset: 39: 'bottle', 41: 'cup' (often detects cans as cups)
TARGET_CLASSES = [39, 41]

def calculate_distance(bbox_height_px, real_height_cm):
    """
    Estimates distance using the Pinhole Camera Model:
    Distance = (Real_Height * Focal_Length) / Object_Pixel_Height
    """
    if bbox_height_px == 0: return 0
    return (real_height_cm * FOCAL_LENGTH) / bbox_height_px

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0) # 0 is usually the default laptop webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Video Stream... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    # verbose=False keeps the terminal clean
    results = model(frame, verbose=False, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # 1. Class ID
            cls_id = int(box.cls[0])
            
            # Filter: Only process if it is a bottle or cup/can
            if cls_id in TARGET_CLASSES:
                
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 2. Confidence Score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # 3. Object Name
                label_name = model.names[cls_id]
                
                # Determine Real Height based on class for distance calc
                real_height = KNOWN_HEIGHT_BOTTLE if label_name == 'bottle' else KNOWN_HEIGHT_CAN

                # Calculate Dimensions
                width_px = x2 - x1
                height_px = y2 - y1

                # 4. Centroid (Center X, Center Y)
                center_x = int(x1 + width_px / 2)
                center_y = int(y1 + height_px / 2)

                # 5. Estimated Distance (Z-axis info)
                distance_cm = calculate_distance(height_px, real_height)

                # 6. Bounding Box Area (Size info)
                area_px = width_px * height_px

                # --- VISUALIZATION ---
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Centroid
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # Prepare Data Label
                label_text = f"{label_name} {conf}"
                stats_text = f"Dist: {int(distance_cm)}cm | Center:({center_x},{center_y})"
                
                # Display Text
                cv2.putText(frame, label_text, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, stats_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Print the "Odometry" data to terminal
                print(f"Object: {label_name} | Dist: {distance_cm:.2f}cm | "
                      f"Pos: ({center_x}, {center_y}) | Conf: {conf}")

    cv2.imshow('Object Detection & Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()