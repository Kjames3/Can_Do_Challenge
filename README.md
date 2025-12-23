# Viam Rover Control System

A web-based control panel for Viam-powered rovers with real-time camera streaming, LIDAR visualization, and YOLOv8-based object detection.

![Rover Control Panel](https://via.placeholder.com/800x400?text=Viam+Rover+Control+Panel)

## Features

### 🎮 Motor Control
- Dual motor control with vertical sliders
- Emergency stop button
- Gamepad support (Xbox, PlayStation, etc.)
  - Right stick for arcade-style driving
  - X/A button for emergency stop

### 📹 Camera Feed
- Real-time JPEG streaming from rover camera
- Low-latency WebSocket transmission

### 🔍 Object Detection
- YOLOv8-based bottle and can detection
- Real-time bounding box annotations
- Distance estimation using pinhole camera model
- Detection confidence display
- Toggle on/off to save processing power

### 📡 LIDAR Visualization
- Real-time 2D point cloud rendering
- PCD (Point Cloud Data) binary format parsing
- Configurable display scale

### 📊 Telemetry
- Motor position (revolutions)
- Motor power percentage
- Connection status indicators

## Requirements

### Python Dependencies
```bash
pip install websockets viam-sdk ultralytics opencv-python numpy
```

### Hardware
- Viam-compatible rover with:
  - Left/Right motors
  - Camera (optional)
  - RPLIDAR or compatible LIDAR sensor (optional)

## Quick Start

### 1. Configure Robot Connection

Edit `server.py` and update your Viam robot credentials:

```python
ROBOT_ADDRESS = "your-robot.viam.cloud"
API_KEY_ID = "your-api-key-id"
API_KEY = "your-api-key"
```

### 2. Start the Server

```bash
python server.py
```

You should see:
```
✓ YOLO detection model loaded successfully.
Connecting to robot...
✓ Connected to RoverName
✓ Motors initialized
✓ Camera initialized
✓ Lidar initialized

==================================================
WebSocket server running on ws://localhost:8081
==================================================
```

### 3. Open the GUI

Open `GUI.html` in a web browser (Chrome, Firefox, Edge recommended).

### 4. Connect

Click the **Connect** button to establish the WebSocket connection.

## Usage Guide

### Motor Control

**Using Sliders:**
- Drag the LEFT and RIGHT sliders up/down to control each motor
- Values range from -100% to +100%

**Using Gamepad:**
1. Connect a gamepad to your computer
2. The GUI will automatically detect it
3. Use the **right stick** for arcade-style driving:
   - Up/Down: Forward/Reverse
   - Left/Right: Turn
4. Press **X** (PlayStation) or **A** (Xbox) for emergency stop

**Emergency Stop:**
- Click the red STOP button, or
- Press the X/A button on your gamepad

### Object Detection

1. Click the **Detection** toggle in the Camera section header
2. The camera feed will now show:
   - Green bounding boxes around detected bottles/cans
   - Red centroid markers
   - Distance estimates in centimeters
3. View detection details in the panel below the camera feed

**Detected Object Classes:**
- `bottle` - Water bottles, soda bottles (COCO class 39)
- `cup` - Often detects cans and cups (COCO class 41)

### LIDAR

1. Click the **Enable** toggle in the LIDAR section
2. The visualization shows:
   - Green dots: LIDAR scan points
   - Red dot at center: Robot position
   - Grid circles: 1m and 2m range markers

## File Structure

```
viam_projects/
├── server_pi.py           # WebSocket server for Raspberry Pi
├── server_jetson.py       # WebSocket server for Jetson
├── GUI.html               # Web-based control panel
├── yolov8n_cans.pt        # Custom trained can detection model
├── train_yolov8_cans.py   # Training script for custom model
├── navigation_fsm.py      # Autonomous navigation state machine
└── README.md              # This file
```

## Training Custom Model

The `yolov8n_cans.pt` model is a fine-tuned YOLOv8n trained to detect soda cans. To retrain or improve the model:

### 1. Download Datasets

Download the following Roboflow datasets in **YOLOv8 format**:

| Dataset | Link | Description |
|---------|------|-------------|
| Cans Dataset | [Roboflow - Cans](https://universe.roboflow.com/heho/cans-p8c8x/dataset/4/download) | 783 training images with can condition labels |
| Soda Can Dataset | [Roboflow - Soda Cans](https://universe.roboflow.com/soda-can-dataset/my-first-project-qqbah) | 288 training images of soda cans |

Extract them to:
```
datasets/
├── can1_dataset/    # First dataset
└── can2_dataset/    # Second dataset
```

### 2. Train the Model

```bash
# Install dependencies
pip install ultralytics pyyaml

# Train for 100 epochs (GPU recommended)
python train_yolov8_cans.py --epochs 100

# The script will:
# - Merge both datasets into a single "can" class
# - Create train/valid/test splits
# - Save the best model to runs/can_detection/yolov8n_cans/weights/best.pt
```

### 3. Deploy the Model

Copy the trained model to the project root:
```bash
copy runs\can_detection\yolov8n_cans\weights\best.pt yolov8n_cans.pt
```

## Configuration

### Detection Parameters (server.py)

```python
KNOWN_HEIGHT_BOTTLE = 20.0  # Water bottle height in cm
KNOWN_HEIGHT_CAN = 12.0     # Soda can height in cm
FOCAL_LENGTH = 600          # Camera focal length (calibrate for accuracy)
TARGET_CLASSES = [39, 41]   # COCO classes to detect
```

### WebSocket Settings

Default server address: `ws://localhost:8081`

To change, edit both:
- `server.py`: `server_address` and `server_port` in `main()`
- `GUI.html`: `SERVER_ADDRESS` constant

## API Reference

### WebSocket Messages (Client → Server)

**Set Motor Power:**
```json
{
  "type": "set_power",
  "motor": "left" | "right",
  "power": -1.0 to 1.0
}
```

**Stop Motors:**
```json
{
  "type": "stop"
}
```

**Toggle Detection:**
```json
{
  "type": "toggle_detection",
  "enabled": true | false
}
```

### WebSocket Messages (Server → Client)

**Readout (broadcast at 10Hz):**
```json
{
  "type": "readout",
  "left_pos": 0.0,
  "left_power": 0.0,
  "right_pos": 0.0,
  "right_power": 0.0,
  "image": "base64...",
  "detection_enabled": false,
  "detections": [
    {
      "label": "bottle",
      "confidence": 0.95,
      "distance_cm": 45.2,
      "center_x": 320,
      "center_y": 240,
      "bbox": [100, 150, 200, 350],
      "area_px": 20000
    }
  ],
  "lidar_points": [[0.5, 0.3], [0.6, 0.2], ...]
}
```

## Troubleshooting

### Connection Issues

**"WebSocket connection failed"**
- Ensure `server.py` is running
- Check that port 8081 is not blocked by firewall
- Verify no other application is using port 8081

**"Failed to connect to robot"**
- Verify your Viam credentials are correct
- Check your internet connection
- Ensure the robot is online in the Viam app

### Detection Issues

**"Failed to load YOLO model"**
- Ensure `yolov8n.pt` is in the same directory as `server.py`
- Install ultralytics: `pip install ultralytics`

**Inaccurate distance estimates**
- Calibrate `FOCAL_LENGTH` for your specific camera
- Formula: `FOCAL_LENGTH = (measured_pixel_height * actual_distance) / actual_height`

### Performance

**Laggy video feed**
- Reduce camera resolution in Viam app
- Disable detection when not needed
- Close other browser tabs

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Viam Robotics](https://www.viam.com/) - Robot platform SDK
- [Ultralytics](https://ultralytics.com/) - YOLOv8 object detection
- [OpenCV](https://opencv.org/) - Computer vision library
