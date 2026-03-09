# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A high-performance autonomous rover platform running on Raspberry Pi 5. The system bypasses Viam SDK's 100 API calls/sec limit by using direct GPIO control via `gpiozero` + `rpi-lgpio`, achieving <5ms motor command latency and 200+ Hz update rates. The rover uses YOLO object detection to find and navigate to aluminum cans.

## Running the System

**On the Raspberry Pi (server):**
```bash
python server_native.py          # Production mode (requires hardware)
python server_native.py --sim    # Simulation mode (no GPIO, for development)
```

**On your laptop/PC (client):**
```bash
python launcher.py               # Opens GUI in browser + optional image sync via SSH/SCP
python scan_for_robot.py         # Find Pi's IP address on network (scans port 8081)
```

**Install dependencies on Pi 5:**
```bash
./install_native.sh
# Or manually:
pip install websockets gpiozero rpi-lgpio picamera2 smbus2 numpy ultralytics opencv-python
```

## Architecture

### Components

| File | Role |
|------|------|
| `server_native.py` | Main WebSocket server on port 8081. Handles client messages, runs broadcast loop at 20fps, orchestrates detection and navigation. |
| `drivers.py` | Hardware abstraction: `NativeMotor`, `NativeEncoder`, `NativeIMU` (MPU6050 I2C), `Picamera2Driver`, `NativeLidar`, `NativePowerSensor`. |
| `robot_state.py` | Extended Kalman Filter (EKF) for odometry. Fuses encoder deltas (prediction) with IMU heading (measurement correction). |
| `navigation_fsm.py` | Autonomous navigation via FSM: IDLE → SEARCHING → APPROACHING (ACQUIRE/ROTATE/DRIVE phases) → ARRIVED → RETURNING. Uses pure pursuit for arc driving. |
| `web/GUI.html` | Control dashboard: camera canvas, joystick sliders, telemetry, 3D Three.js top-down map. |
| `web/main.js` | Frontend WebSocket client, gamepad polling, UI state management, Three.js rendering. |

### Data Flow

```
Browser ──WebSocket (ws://<pi-ip>:8081)──> server_native.py
                                                │
                    ┌───────────────────────────┤
                    ▼                           ▼
              drivers.py                 navigation_fsm.py
         (NativeMotor/Encoder)           (FSM + pure pursuit)
                    │                           │
                    └──────────> robot_state.py (EKF)
                                      │
                              broadcast_loop
                                      │
                              ──> Browser (telemetry + base64 JPEG at 20fps)
```

### WebSocket Protocol

**Client → Server:**
```json
{"type": "set_power", "motor": "left", "power": 0.5}
{"type": "stop"}
{"type": "toggle_detection", "enabled": true}
{"type": "set_model", "model": "yolo11n_cans"}
{"type": "start_auto_drive"}
{"type": "stop_auto_drive"}
{"type": "start_demo", "interval": 30.0}
{"type": "capture_image"}
```

**Server → Client:**
```json
{
  "type": "readout",
  "robot_pose": {"x": 100, "y": 50, "theta": 0.1},
  "detections": [{"label": "can", "confidence": 0.92, "distance_cm": 45.2, "bbox": [...]}],
  "image": "<base64_jpeg>",
  "fps": {"camera": 20, "yolo": 12},
  "power": {"voltage": 11.8, "current": 2.3}
}
```

## Key Parameters to Know

**`server_native.py`** — main tuning constants:
- `DRIFT_COMPENSATION` — left motor correction factor for straight-line driving
- `CONFIDENCE_THRESHOLD = 0.25` / `DISPLAY_CONFIDENCE_THRESHOLD = 0.70`
- `FOCAL_LENGTH = 1298`, `KNOWN_HEIGHT_CAN = 15.7` — for distance estimation
- `VIDEO_FPS_CAP = 20`, `IMAGE_WIDTH = 1536`, `IMAGE_HEIGHT = 864`
- `_MODEL_MAP` — maps model keys to (ncnn_path, pt_fallback_path, single_class_bool)

**`robot_state.py`** — EKF tuning:
- `WHEEL_CIRCUMFERENCE_MM = 75.0`, `WHEEL_BASE_MM = 600.0`
- `Q = diag([0.5, 0.5, 0.1])` (process noise), `R = 0.05` (IMU noise)

**`navigation_fsm.py`** — navigation tuning:
- `target_distance_cm = 4.0` — stop distance from target
- `drive_speed = 0.40`, `rotate_speed = 0.35`
- `bearing_threshold = 0.20 rad`

## Coordinate System

Standard Right-Hand Rule (ROS-compatible):
- **X+** = Forward, **Y+** = Left, **Theta=0** = facing forward
- Counter-clockwise is positive rotation
- Heading to waypoint: `np.arctan2(delta_y, delta_x)`

## Detection Models

Models live in `models/`. The NCNN format is preferred on Pi 5 for speed (~1.6ms inference).

| Key | Description |
|-----|-------------|
| `yolo11n_cans` | Custom-trained single-class (can), NCNN |
| `yolo11n_standard` | Official YOLO11n, all 80 COCO classes |
| `yolo11n_teacher` / `yolo11n_student` | Knowledge-distilled pair from Colab training |

Switching models at runtime: send `{"type": "set_model", "model": "<key>"}`. Server hot-swaps the YOLO instance and broadcasts `model_changed`.

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `convert_to_ncnn.py` | Convert `.pt` → NCNN for faster Pi 5 inference |
| `monitor_sensor_noise.py` | Real-time matplotlib graph of IMU/encoder data |
| `calibration/calibrate_focal_length.py` | Calibrate `FOCAL_LENGTH` constant |
| `calibration/calibrate_motors.py` | Calibrate `DRIFT_COMPENSATION` |
| `download_model.py` | Download official YOLO models from Ultralytics |
