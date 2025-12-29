# Viam Rover 2 - Native Control System

A high-performance robotics platform for autonomous object detection and collection, built to bypass Viam SDK limitations.

![Rover Control System](https://via.placeholder.com/800x200?text=Viam+Rover+2+Native+Control)

---

## 🎯 Project Goal

Build an autonomous rover capable of:
- **Real-time object detection** using YOLO11
- **Autonomous navigation** with FSM-based control
- **Precise motor control** using encoder feedback
- **IMU-assisted heading** for accurate turns

---

## 🔧 Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| **SBC** | Raspberry Pi 5 (8GB) | Main compute |
| **Camera** | IMX708 (Pi Camera Module 3) | 1536×864 @ 20fps |
| **Motors** | Viam Rover 2 DC Motors | Differential drive |
| **Encoders** | Hall effect (1000 PPR) | Odometry |
| **IMU** | MPU6050 | Heading & tilt detection |
| **LiDAR** | RPLiDAR A1M8 (optional) | Obstacle detection |

### GPIO Pin Mapping

```
Left Motor:   In1=31, In2=29, PWM=33, Encoder=38
Right Motor:  In1=22, In2=18, PWM=32, Encoder=40
IMU:          I2C Bus 1, Address 0x68
```

---

## 🚀 Why Native Implementation?

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

## 📦 Quick Start

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

Open `GUI.html` in a web browser on any device on the same network.

### 4. Connect

Enter the Pi's IP address and click **Connect**.

---

## 📁 Project Structure

```
viam_projects/
├── server_native.py      # Main server (native GPIO control)
├── GUI.html              # Web control interface
├── drivers.py            # Hardware driver classes
├── robot_state.py        # Odometry & pose tracking
├── navigation_fsm.py     # Autonomous navigation FSM
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

## 🗂️ Directory Guide

| Folder | Description | When to Use |
|--------|-------------|-------------|
| [viam/](viam/) | Viam SDK server implementations | Cloud monitoring, remote access |
| [calibration/](calibration/) | Motor & camera calibration | After hardware changes |
| [training/](training/) | YOLO model training | Improving detection accuracy |
| [tests/](tests/) | Hardware verification | Debugging, validation |

---

## 🎮 Core Features

### Motor Control
- **Arcade-style gamepad** control (right stick)
- **Keyboard controls** via GUI
- **Drift compensation** for straight-line driving

### Object Detection
- **YOLO11n** trained on soda can dataset
- **Distance estimation** via pinhole camera model
- Real-time bounding box overlay

### Autonomous Navigation
- **FSM States**: IDLE → SEARCHING → APPROACHING → ARRIVED
- **IMU-precision turns** using gyroscope
- **Obstacle avoidance** with backup behavior
- **Auto-return** to starting position

### Telemetry
- Encoder positions (revolutions)
- Robot pose (X, Y, θ)
- IMU heading
- Motor power levels
- FPS & detection stats

---

## ⚙️ Configuration

Key parameters in `server_native.py`:

```python
# Camera Settings
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 864
VIDEO_FPS_CAP = 20

# Detection
FOCAL_LENGTH = 1298           # Calibrate with calibration/calibrate_focal_length.py
CONFIDENCE_THRESHOLD = 0.25

# Motor Tuning
DRIFT_COMPENSATION = -0.10    # Calibrate with calibration/calibrate_motors.py
```

---

## 🔌 WebSocket API

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

## 📚 See Also

- **[viam/README.md](viam/README.md)** - Using Viam SDK instead
- **[calibration/README.md](calibration/README.md)** - Calibrating motors & camera
- **[training/README.md](training/README.md)** - Training custom YOLO models
- **[tests/README.md](tests/README.md)** - Running hardware tests

---

## 📄 License

MIT License - See LICENSE file for details.
