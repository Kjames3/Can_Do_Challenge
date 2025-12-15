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
ROBOT_ADDRESS = "rover-3-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "164902f7-7737-4675-85d6-151fedb70a82"
API_KEY = "ab5lhwuctyf0t34wt7mu9gq24kgx8azh"

# Component Names
LEFT_MOTOR_NAME = "left"
RIGHT_MOTOR_NAME = "right"
CAMERA_NAME = "cam"
LIDAR_NAME = "rplidar"

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
lidar: Camera = None

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
    global robot, left_motor, right_motor, camera, lidar
    
    print("Connecting to robot...")
    
    try:
        dial_options = DialOptions.with_api_key(API_KEY, API_KEY_ID)
        robot = await RobotClient.at_address(ROBOT_ADDRESS, dial_options)
        print(f"✓ Connected to {robot.name}")
        
        # Initialize motors
        left_motor = Motor.from_robot(robot, LEFT_MOTOR_NAME)
        right_motor = Motor.from_robot(robot, RIGHT_MOTOR_NAME)
        print("✓ Motors initialized")
        
        # Initialize camera (optional)
        try:
            camera = Camera.from_robot(robot, CAMERA_NAME)
            print("✓ Camera initialized")
        except Exception:
            print("✗ Camera not found")
        
        # Initialize lidar (optional)
        try:
            lidar = Camera.from_robot(robot, LIDAR_NAME)
            print("✓ Lidar initialized")
        except Exception:
            print("✗ Lidar not found")
        
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
    global robot, left_motor, right_motor, camera, lidar
    global connected_clients, detection_enabled
    
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
                    "detection_enabled": detection_enabled
                }
                
                # Process camera feed
                if camera:
                    try:
                        img_bytes = await camera.get_image(mime_type="image/jpeg")
                        
                        if detection_enabled:
                            # Run detection and get annotated image
                            annotated_b64, detections = process_detection(img_bytes)
                            if annotated_b64:
                                data["image"] = annotated_b64
                                data["detections"] = detections
                            else:
                                data["image"] = base64.b64encode(img_bytes).decode('utf-8')
                                data["detections"] = []
                        else:
                            # Send raw image without detection
                            data["image"] = base64.b64encode(img_bytes).decode('utf-8')
                            data["detections"] = []
                            
                    except Exception as e:
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
                    await asyncio.wait(send_tasks)
                    
            except Exception as e:
                print(f"Producer error: {e}")
        
        # Rate limit: 10Hz
        await asyncio.sleep(0.1)


async def consumer_task(websocket):
    """Handle incoming messages from a WebSocket client."""
    global robot, left_motor, right_motor, detection_enabled
    
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
                
        except Exception as e:
            print(f"Consumer error: {e}")


async def handler(websocket, path):
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