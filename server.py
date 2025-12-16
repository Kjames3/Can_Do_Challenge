"""
Viam Rover Control Server with Bottle Detection

This WebSocket server connects to a Viam-powered rover and provides:
- Motor control (left/right)
- Camera feed streaming
- Lidar data streaming
- Real-time bottle/can detection using YOLOv8

Usage:
    python server.py
"""

import asyncio
import json
import websockets
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.motor import Motor
from viam.components.camera import Camera
from viam.components.motor import Motor
from viam.components.camera import Camera
from viam.components.sensor import Sensor
from viam.components.power_sensor import PowerSensor
import base64
import io
import struct
import numpy as np
import cv2
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Robot Connection Details
ROBOT_ADDRESS = "yeep-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "d55dcbc4-6c31-4d78-97d9-57792293a0b7"
API_KEY = "3u88u6fsuowp1wv4inpyebnv13k6dkhn"

# Component Names
# Component Names
LEFT_MOTOR_NAME = "left"
RIGHT_MOTOR_NAME = "right"
CAMERA_NAME = "cam"
LIDAR_NAME = None  # No lidar on this robot
BATTERY_NAME = "ina219"

# Detection Configuration
KNOWN_HEIGHT_BOTTLE = 20.0  # Standard water bottle height in cm
KNOWN_HEIGHT_CAN = 12.0     # Standard soda can height in cm
FOCAL_LENGTH = 600          # Webcam focal length (calibrate for accuracy)
TARGET_CLASSES = [39, 41]   # COCO classes: 39=bottle, 41=cup (often detects cans)

# =============================================================================
# GLOBAL STATE
# =============================================================================

robot: RobotClient = None
left_motor: Motor = None
right_motor: Motor = None
camera: Camera = None
camera: Camera = None
lidar: Camera = None
camera: Camera = None
lidar: Camera = None
battery: PowerSensor = None

# Auto-drive state
is_auto_driving: bool = False
target_distance_cm: float = 12.5  # Target distance (between 10-15cm)
dist_threshold_cm: float = 2.5    # +/- 2.5cm tolerance
center_threshold_px: int = 50     # Pixel error tolerance for centering
  
# FPS Optimization
frame_count: int = 0
detection_interval: int = 3  # Run detection every N frames
last_detections: list = []


# Detection state
detection_model: YOLO = None
detection_enabled: bool = False

# Connected WebSocket clients
connected_clients = set()

# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def initialize_detection_model():
    """Load the YOLO model for object detection."""
    global detection_model
    try:
        detection_model = YOLO('yolov8n.pt')
        print("✓ YOLO detection model loaded successfully.")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False


def calculate_distance(bbox_height_px: float, real_height_cm: float) -> float:
    """
    Estimate distance using the Pinhole Camera Model.
    Distance = (Real_Height * Focal_Length) / Object_Pixel_Height
    """
    if bbox_height_px <= 0:
        return 0.0
    return (real_height_cm * FOCAL_LENGTH) / bbox_height_px


def process_detection(image_bytes: bytes) -> tuple:
    """
    Process an image for bottle/can detection.
    
    Args:
        image_bytes: JPEG image as bytes
        
    Returns:
        Tuple of (annotated_image_b64, detections_list)
    """
    global detection_model
    
    if detection_model is None:
        return None, []
    
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, []
        
        # Run YOLO inference
        results = detection_model(frame, verbose=False, stream=True)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # Only process target classes (bottle, cup/can)
                if cls_id not in TARGET_CLASSES:
                    continue
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                
                # Calculate dimensions
                width_px = x2 - x1
                height_px = y2 - y1
                
                # Get label and confidence
                label = detection_model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Calculate center point
                center_x = int(x1 + width_px / 2)
                center_y = int(y1 + height_px / 2)
                
                # Estimate distance
                real_height = KNOWN_HEIGHT_BOTTLE if label == 'bottle' else KNOWN_HEIGHT_CAN
                distance_cm = calculate_distance(height_px, real_height)
                
                # Store detection data
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "distance_cm": round(distance_cm, 1),
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": [x1, y1, x2, y2],
                    "area_px": width_px * height_px
                })
                
                # Draw annotations on frame
                color = (0, 255, 0)  # Green for detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw label text
                label_text = f"{label} {confidence:.0%}"
                dist_text = f"{int(distance_cm)}cm"
                cv2.putText(frame, label_text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, dist_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode annotated frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return annotated_b64, detections
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None, []

# =============================================================================
# LIDAR PARSING
# =============================================================================

def parse_pcd(data: bytes) -> list:
    """
    Parse a PCD (Point Cloud Data) file in binary format.
    Returns a list of [x, y] points.
    """
    try:
        # Find the end of the header
        header_end_index = data.find(b"DATA binary\n")
        if header_end_index == -1:
            return []
        
        header = data[:header_end_index].decode('ascii')
        raw_data = data[header_end_index + 12:]
        
        # Parse header fields
        lines = header.split('\n')
        fields = []
        size = []
        type_ = []
        
        for line in lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                size = [int(s) for s in line.split()[1:]]
            elif line.startswith("TYPE"):
                type_ = line.split()[1:]
        
        # Parse as float32 array if all fields are floats
        if all(s == 4 for s in size) and all(t == 'F' for t in type_):
            num_floats = len(fields)
            arr = np.frombuffer(raw_data, dtype=np.float32)
            if arr.size % num_floats == 0:
                arr = arr.reshape(-1, num_floats)
                return arr[:, :2].tolist()
        
        return []
        
    except Exception as e:
        print(f"PCD parsing error: {e}")
        return []

# =============================================================================
# ROBOT CONNECTION
# =============================================================================

async def connect_to_robot() -> bool:
    """Establish connection to the Viam robot."""
    global robot, left_motor, right_motor, camera, lidar, battery
    
    print("Connecting to robot...")
    
    try:
        options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
        robot = await RobotClient.at_address(ROBOT_ADDRESS, options)
        robot = await RobotClient.at_address(ROBOT_ADDRESS, options)
        print(f"✓ Connected to {ROBOT_ADDRESS}")
        
        print("Available resources:")
        for name in robot.resource_names:
            print(f" - {name.name} ({name.subtype})")
        
        # Initialize motors (optional)
        try:
            left_motor = Motor.from_robot(robot, LEFT_MOTOR_NAME)
            right_motor = Motor.from_robot(robot, RIGHT_MOTOR_NAME)
            print("✓ Motors initialized")
        except Exception:
            print("✗ Motors not found")
            left_motor = None
            right_motor = None
        
        # Initialize camera (optional)
        try:
            camera = Camera.from_robot(robot, CAMERA_NAME)
            print("✓ Camera initialized")
        except Exception:
            print("✗ Camera not found")
        
        # Initialize lidar (optional)
        if LIDAR_NAME:
            try:
                lidar = Camera.from_robot(robot, LIDAR_NAME)
                print("✓ Lidar initialized")
            except Exception:
                print("✗ Lidar not found")
                lidar = None
        else:
            lidar = None
            
        # Initialize battery (optional)
        try:
            battery = PowerSensor.from_robot(robot, BATTERY_NAME)
            print("✓ Battery initialized")
        except Exception:
            print(f"✗ Battery '{BATTERY_NAME}' not found")
            battery = None
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        robot = None
        return False

# =============================================================================
# WEBSOCKET HANDLERS
# =============================================================================

async def producer_task():
    """
    Continuously broadcast sensor data to all connected clients.
    Runs at ~10Hz for smooth video/lidar updates.
    """
    global robot, left_motor, right_motor, camera, lidar, battery
    global connected_clients, detection_enabled, is_auto_driving
    global frame_count, last_detections

    
    while True:
        if robot and connected_clients:
            try:
                # Gather motor data
                left_pos, left_power_data, right_pos, right_power_data = await asyncio.gather(
                    left_motor.get_position(),
                    left_motor.is_powered(),
                    right_motor.get_position(),
                    right_motor.is_powered()
                )
                
                # Build response data
                data = {
                    "type": "readout",
                    "left_pos": left_pos,
                    "left_power": left_power_data[1],
                    "right_pos": right_pos,
                    "right_power": right_power_data[1],
                    "right_pos": right_pos,
                    "right_power": right_power_data[1],
                    "detection_enabled": detection_enabled,
                    "is_auto_driving": is_auto_driving
                }
                
                # Get battery data
                if battery:
                    try:
                        # For PowerSensor, get_voltage() returns (volts, is_ac)
                        volts, is_ac = await battery.get_voltage()
                        # Also could get watts, current, etc.
                        data["battery"] = {"volts": volts}
                    except Exception:
                        pass
                
                if camera:
                    try:
                        # Note: newer Viam SDK returns ViamImage object, use .data for bytes
                        viam_img = await camera.get_image(mime_type="image/jpeg")
                        img_bytes = viam_img.data
                        
                        # Resize for performance (optional, but good practice if camera is 1080p)
                        # We need to decode to resize anyway if we are detecting.
                        # If NOT detecting, we just pass through.
                        
                        if detection_enabled:
                            frame_count += 1
                            
                            # Decode once
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # Resize if huge
                            h, w = frame.shape[:2]
                            if w > 640:
                                frame = cv2.resize(frame, (640, 480))
                                
                            # Run detection only on interval
                            if frame_count % detection_interval == 0:
                                # Run detection on this frame
                                # We need to pass the ARRAY to process_detection now, or refactor process_detection
                                # Let's optimize process_detection to accept an image array instead of bytes if possible, 
                                # or just do it inline here to avoid re-encoding/decoding.
                                
                                # Refactoring process_detection is cleaner but complex edit.
                                # Let's just use the logic here.
                                
                                if detection_model:
                                    results = detection_model(frame, verbose=False, stream=True)
                                    new_detections = []
                                    for r in results:
                                        for box in r.boxes:
                                            cls_id = int(box.cls[0])
                                            if cls_id not in TARGET_CLASSES: continue
                                            
                                            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                                            width_px, height_px = x2-x1, y2-y1
                                            label = detection_model.names[cls_id]
                                            conf = float(box.conf[0])
                                            center_x = int(x1 + width_px/2)
                                            center_y = int(y1 + height_px/2)
                                            real_h = KNOWN_HEIGHT_BOTTLE if label == 'bottle' else KNOWN_HEIGHT_CAN
                                            dist_cm = calculate_distance(height_px, real_h)
                                            
                                            new_detections.append({
                                                "label": label, "confidence": round(conf, 2),
                                                "distance_cm": round(dist_cm, 1),
                                                "center_x": center_x, "center_y": center_y,
                                                "bbox": [x1, y1, x2, y2], "area_px": width_px * height_px
                                            })
                                    last_detections = new_detections
                            
                            # Draw LAST known detections on CURRENT frame (Tracking effect)
                            for d in last_detections:
                                x1, y1, x2, y2 = d['bbox']
                                label = d['label']
                                conf = d['confidence']
                                dist = d['distance_cm']
                                
                                color = (0, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.circle(frame, (d['center_x'], d['center_y']), 4, (0, 0, 255), -1)
                                cv2.putText(frame, f"{label} {dist}cm", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Re-encode to send
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            
                            data["image"] = base64.b64encode(buffer).decode('utf-8')
                            data["detections"] = last_detections
                                
                        else:
                            # Send raw image without detection
                            data["image"] = base64.b64encode(img_bytes).decode('utf-8')
                            data["detections"] = []


                        # === AUTO-DRIVE LOGIC ===
                        if is_auto_driving and detection_enabled and data["detections"]:
                            # Find the closest bottle/can
                            target = max(data["detections"], key=lambda d: d['confidence'])
                            
                            img_center_x = 320 # Assuming 640x480
                            
                            # Error calculation
                            error_x = target['center_x'] - img_center_x
                            dist = target['distance_cm']
                            
                            # Control Logic
                            linear = 0.0
                            angular = 0.0
                            
                            print(f"Auto-drive: Dist={dist}cm, ErrorX={error_x}")
                            
                            if abs(error_x) > center_threshold_px:
                                # Turn to center
                                angular = 0.2 if error_x > 0 else -0.2
                            else:
                                # Centered, check distance
                                if dist > (target_distance_cm + dist_threshold_cm):
                                    linear = 0.3 # Move forward
                                elif dist < (target_distance_cm - dist_threshold_cm):
                                    linear = -0.3 # Move backward
                                else:
                                    # At target
                                    linear = 0.0
                                    angular = 0.0
                                    print("Target reached!")
                                    is_auto_driving = False # Stop auto-driving
                            
                            # Mixing
                            l_pow = linear + angular
                            r_pow = linear - angular
                            
                            # Clamp
                            l_pow = max(min(l_pow, 1.0), -1.0)
                            r_pow = max(min(r_pow, 1.0), -1.0)
                            
                            if left_motor and right_motor:
                                await left_motor.set_power(l_pow)
                                await right_motor.set_power(r_pow)
                                
                        elif is_auto_driving:
                             # No detections or loss of tracking
                             if left_motor and right_motor:
                                 await left_motor.set_power(0)
                                 await right_motor.set_power(0)
                            
                    except Exception as e:
                        print(f"Camera/Auto-drive loop error: {e}")
                        pass
                
                # Process lidar data
                if lidar:
                    try:
                        pc_bytes, _ = await lidar.get_point_cloud()
                        points = parse_pcd(pc_bytes)
                        data["lidar_points"] = points
                    except Exception:
                        pass
                
                # Broadcast to all clients
                message = json.dumps(data)
                send_tasks = [client.send(message) for client in connected_clients]
                if send_tasks:
                    await asyncio.gather(*send_tasks)
                    
            except Exception as e:
                print(f"Producer error: {e}")
        
        # Rate limit: 10Hz
        # Rate limit: ~60Hz (limited by camera speed roughly)
        await asyncio.sleep(0.01)


async def consumer_task(websocket):
    """Handle incoming messages from a WebSocket client."""
    global robot, left_motor, right_motor, detection_enabled, is_auto_driving
    
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if not robot:
                continue
            
            if msg_type == "set_power":
                motor = data.get("motor")
                power = float(data.get("power", 0.0))
                
                if motor == "left":
                    await left_motor.set_power(power)
                elif motor == "right":
                    await right_motor.set_power(power)
            
            elif msg_type == "stop":
                await left_motor.set_power(0)
                await right_motor.set_power(0)
            
            elif msg_type == "toggle_detection":
                detection_enabled = data.get("enabled", False)
                print(f"Detection {'enabled' if detection_enabled else 'disabled'}")

            elif msg_type == "start_auto_drive":
                 print("Starting Auto-Drive...")
                 is_auto_driving = True
                 detection_enabled = True # Force detection on
            
            elif msg_type == "stop_auto_drive":
                 print("Stopping Auto-Drive...")
                 is_auto_driving = False
                 await left_motor.set_power(0)
                 await right_motor.set_power(0)
                
        except Exception as e:
            print(f"Consumer error: {e}")


async def handler(websocket):
    """Handle a new WebSocket connection."""
    global connected_clients
    
    print(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    
    try:
        await consumer_task(websocket)
    finally:
        print(f"Client disconnected: {websocket.remote_address}")
        connected_clients.remove(websocket)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Initialize and run the server."""
    
    # Load detection model
    initialize_detection_model()
    
    # Connect to robot
    if not await connect_to_robot():
        return
    
    # Start producer task
    asyncio.create_task(producer_task())
    
    # Start WebSocket server
    server_address = "localhost"
    server_port = 8081
    
    print(f"\n{'=' * 50}")
    print(f"WebSocket server running on ws://{server_address}:{server_port}")
    print(f"{'=' * 50}\n")
    
    async with websockets.serve(handler, server_address, server_port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())