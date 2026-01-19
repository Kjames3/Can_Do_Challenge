# Viam Rover 2 - Native Control System

A high-performance robotics platform for autonomous object detection and collection, built to bypass Viam SDK limitations.

![Rover Control System](https://via.placeholder.com/800x200?text=Viam+Rover+2+Native+Control)

---

## Project Goal

Build an autonomous rover capable of:
- **Real-time object detection** using YOLO11
- **Autonomous navigation** with FSM-based control
- **Precise motor control** using encoder feedback
- **IMU-assisted heading** for accurate turns

---

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| **SBC** | Raspberry Pi 5 (16GB) | Main compute |
| **Camera** | IMX708 (Pi Camera Module 3) | 1536×864 @ 20fps |
| **Motors** | Viam Rover 2 DC Motors | Differential drive |
| **Encoders** | Hall effect (1000 PPR) | Odometry |
| **IMU** | MPU6050 | Heading & tilt detection |
| **LiDAR** | RPLiDAR A1M8 (optional) | Obstacle detection |
| **Battery** | (4) 12V 18640 batteries | Power supply |
| **Power Monitor** | INA219 | Current & voltage monitoring |
| **Converter** | OKY3502-4 DC-DC converter| Power management |

### GPIO Pin Mapping

```
Left Motor:   In1=31, In2=29, PWM=33, Encoder=38
Right Motor:  In1=22, In2=18, PWM=32, Encoder=40
IMU:          I2C Bus 1, Address 0x68
```

---

## Why Native Implementation?

The standard Viam SDK has a **100 API calls/second limit** which causes:
- Motor command timeouts during high-frequency control loops
- Laggy response when combining motor control + sensor reads
- **Network latency** adds 50-200ms per cloud API call

**Our Solution**: Direct GPIO control using `gpiozero` + `rpi-lgpio`:

| Metric | Viam SDK | Native |
|--------|----------|--------|
| Motor update rate | ~50 Hz (limited) | **200+ Hz** |
| Command latency | 50-200ms | **<5ms** |
| API call limit | 100/sec | **Unlimited** |
| Network dependency | Required | **None** |

---

## Quick Start

### 1. Install Dependencies (on Pi 5)

```bash
# Run the installation script
chmod +x install_native.sh
./install_native.sh
```

Or manually:
```bash
pip install websockets gpiozero rpi-lgpio picamera2 smbus2 numpy ultralytics opencv-python
```

### 2. Start the Server

```bash
# Production mode
python server_native.py

# Simulation mode (no hardware)
python server_native.py --sim
```

### 3. Open the GUI

Open `web/GUI.html` in a web browser on any device on the same network.

### 4. Connect

Enter the Pi's IP address and click **Connect**.

---

## Project Structure

```
viam_projects/
├── server_native.py      # Main server (native GPIO control)
├── drivers.py            # Hardware driver classes
├── robot_state.py        # Odometry & pose tracking
├── navigation_fsm.py     # Autonomous navigation FSM
├── web/                  # Web control interface (GUI)
├── yahboom/              # Yahboom X3 specific drivers & server
├── models/               # YOLO detection models
├── install_native.sh     # Installation script
│
├── viam/                 # Viam SDK implementations
├── calibration/          # Hardware calibration scripts
├── training/             # YOLO model training
├── tests/                # Test & verification scripts
├── datasets/             # Training data
└── runs/                 # Training run history
```

---

## Directory Guide

| Folder | Description | When to Use |
|--------|-------------|-------------|
| [web/](web/README.md) | Web GUI Source | Editing the frontend interface |
| [yahboom/](yahboom/README.md) | X3 Robot Drivers | Running on Jetson Orin Nano |
| [models/](models/README.md) | AI Models | Managing YOLO versions |
| [viam/](viam/README.md) | Viam SDK server implementations | Cloud monitoring, remote access |
| [calibration/](calibration/README.md) | Motor & camera calibration | After hardware changes |
| [training/](training/README.md) | YOLO model training | Improving detection accuracy |
| [tests/](tests/README.md) | Hardware verification | Debugging, validation |

---

## Core Features

### Motor Control
- **Arcade-style gamepad** control (right stick)
- **Keyboard controls** via GUI
- **Drift compensation** for straight-line driving

### Object Detection
### Object Detection
- **YOLOv26n (Nano)**: Optimized for edge inference.
- **Performance Metrics**:
  - Inference: ~1.6ms (GPU) / Extremely fast on Pi 5 (NCNN)
  - mAP50: 84.2%
  - mAP50-95: 64.5%
  - Precision: 82.8%
- **Distance estimation** via pinhole camera model
- Real-time bounding box overlay

### Autonomous Navigation
- **Behavior Tree Architecture**: Modular `Selector` and `Sequence` nodes replace rigid FSM logic.
- **Prioritized Behaviors**: Avoid Obstacle → Return Home → Approach Target → Search.
- **Pure Pursuit Controller**: Uses a lookahead point and constant curvature blending to drive smooth arcs towards the target.
- **Robust Return-to-Home**: Wait → Backup (20cm) → Navigate → Align.
- **Smart Final Alignment**: Visual Lock (3° precision) and Blind Fallback (15° precision).
- **Obstacle avoidance** with backup behavior.
- **3D Trajectory visualization** in GUI.

### State Estimation (EKF)
- **Extended Kalman Filter**: Probabilistically fuses wheel odometry (prediction) with IMU heading (correction).
- **Process Noise**: Models encoder slip and drift.
- **Measurement Noise**: Models IMU sensor uncertainty.
- **Result**: Robust position tracking even when wheels slip or IMU drifts.

### Telemetry
- Encoder positions (revolutions)
- Robot pose (X, Y, θ)
- IMU heading
- Motor power levels
- FPS & detection stats

---

## Coordinate System & Navigation Math

The robot now uses a **Standard Right-Hand (X-Forward)** coordinate system (ROS compatible):

- **X+** = Forward (North)
- **Y+** = Left (West)
- **Theta=0°** = Facing X+ (Forward)
- **Theta=+90°** = Facing Y+ (Left) - *Counter-Clockwise is Positive*

### Navigation Heading
To calculate the heading from Point A (Start) to Point B (Goal):

```python
# Standard atan2 for Right-Hand Rule
heading = np.arctan2(delta_y, delta_x)
```

## 🔄 Recent Updates (Jan 2026)
- **Navigation Engine (Global-Only)**: Removed Hybrid Navigation logic. Robot now navigates using exclusively Global Map Coordinates (Pure Pursuit Gain=1.5) to prevent "random turns" and "wide circles".
- **NCNN Upgrade**: Switched model to `yolo11n_cans_ncnn_model` for high-performance detection on Raspberry Pi 5.
- **Coordinate System Overhaul**: Switched to Standard Right-Hand Rule. Fixed "Mirror World" & 70° turn bugs.
- **Improved Alignment**: Added timeout (5s) and relaxed thresholds (8°).
- **Sensor Polarity**: Fixed encoder delta calculation to match IMU.

### Auto-Return Logic
1.  **Goal Persistence**: The robot remembers the (X, Y) of the target even after driving back.
2.  **Singularity Handling**: If facing 180° away from target, it forces a turn to break mathematical deadlock.
3.  **Pulse Alignment**: When error < 20°, motors "pulse" (0.15s ON) to nudge alignment without oscillation.

## Configuration

Key parameters in `server_native.py`:

```python
# Camera Settings
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 864
VIDEO_FPS_CAP = 20

# Detection
FOCAL_LENGTH = 1298           # Calibrate with calibration/calibrate_focal_length.py
KNOWN_HEIGHT_CAN = 15.7       # 16oz Coke Can Height (cm)
CONFIDENCE_THRESHOLD = 0.25

# Motor Tuning
DRIFT_COMPENSATION = -0.10    # Calibrate with calibration/calibrate_motors.py
```

---

## WebSocket API

Default: `ws://<pi-ip>:8081`

### Client → Server

```json
// Motor control
{"type": "set_power", "motor": "left", "power": 0.5}
{"type": "stop"}

// Toggle detection
{"type": "toggle_detection", "enabled": true}

// Autonomous mode
{"type": "start_auto"}
{"type": "stop_auto"}
```

### Server → Client

```json
{
  "type": "readout",
  "left_pos": 12.5,
  "right_pos": 12.3,
  "robot_pose": {"x": 100, "y": 50, "theta": 0.1},
  "image": "base64...",
  "detections": [{"label": "can", "distance_cm": 45, ...}]
}
```

---

## See Also

- **[viam/README.md](viam/README.md)** - Using Viam SDK instead
- **[calibration/README.md](calibration/README.md)** - Calibrating motors & camera
- **[training/README.md](training/README.md)** - Training custom YOLO models
- **[tests/README.md](tests/README.md)** - Running hardware tests

- **[tests/README.md](tests/README.md)** - Running hardware tests

---

## Future Improvements

While the current system uses robust deterministic logic (dead reckoning), the project roadmap includes upgrading to probabilistic state refinenent:

### 1. Probabilistic Perception (Bayesian/Fisher)
Moving from binary "trust" to probabilistic updates.
- **Current**: If YOLO says "Can is at 45cm", we believe it 100%.
- **Future**: Use **Fisher Information** to quantify how much information a frame provides. Use a **Bayesian Estimator** to update the belief map, trusting "sharp/close" frames more than "blurry/far" frames.

### 2. Parameter Estimation (Least Squares)
Calibrating physical constants mathematically rather than manually.
- **Concept**: The "Calibration Problem" involves solving for the *true* physical parameters that minimize error over time.
- **Application**: Driving the robot in a closed loop and using **Least Squares Estimation (LSE)** on the start/end drift to calculate the *exact* effective wheel diameter (to 0.01mm) and track width. This would significantly reduce systematic odometry errors.

---

## License

MIT License - See LICENSE file for details.
