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
- **FSM States**: IDLE → SEARCHING → APPROACHING → ARRIVED → RETURNING
- **Pure Pursuit Controller**: Uses a lookahead point and constant curvature blending to drive smooth arcs towards the target, rather than sharp "rotate-then-drive" movements.
- **Map-Based Navigation** using Y-Forward Cartesian coordinates
- **Robust Return-to-Home**: Wait → Backup (20cm) → Navigate → Align
- **Smart Final Alignment**:
    - **Visual Lock**: Uses camera to center target (3° precision)
    - **Blind Fallback**: Uses goal coordinates if target lost (15° precision)
    - **Pulse Logic**: Micro-pulses motors to fix oscillation close to target
- **Obstacle avoidance** with backup behavior
- **3D Trajectory visualization** in GUI

### Telemetry
- Encoder positions (revolutions)
- Robot pose (X, Y, θ)
- IMU heading
- Motor power levels
- FPS & detection stats

---

## 🧭 Coordinate System & Navigation Math

The robot uses a **Y-Forward / Right-Handed** coordinate system:

- **X+** = Right (East)
- **Y+** = Forward (North)
- **Theta=0°** = Facing Y+ (North)
- **Theta=+90°** = Facing Left (West) - *Standard mathematical convention*

### Navigation Heading
To calculate the heading from Point A (Start) to Point B (Goal):

```python
# Standard atan2(y, x) is for X-Forward systems.
# For Y-Forward, we rotate the inputs:
heading = np.arctan2(-delta_x, delta_y)
```

### Auto-Return Logic
1.  **Goal Persistence**: The robot remembers the (X, Y) of the target even after driving back.
2.  **Singularity Handling**: If facing 180° away from target, it forces a turn to break mathematical deadlock.
3.  **Pulse Alignment**: When error < 20°, motors "pulse" (0.15s ON) to nudge alignment without oscillation.

## ⚙️ Configuration

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

- **[tests/README.md](tests/README.md)** - Running hardware tests

---

## 🔮 Future Improvements

While the current system uses robust deterministic logic (dead reckoning), the project roadmap includes upgrading to probabilistic state refinenent:

### 1. Advanced State Estimation (KF/EKF)
Replacing the current dead reckoning model with **Kalman Filters (KF)** or **Extended Kalman Filters (EKF)**.
- **Why**: Dead reckoning assumes zero wheel slip. An EKF would fuse encoder data (proprioception) with IMU and Camera data (exteroception) to estimate the robot's *true* position and covariance (uncertainty) dynamically.

### 2. Probabilistic Perception (Bayesian/Fisher)
Moving from binary "trust" to probabilistic updates.
- **Current**: If YOLO says "Can is at 45cm", we believe it 100%.
- **Future**: Use **Fisher Information** to quantify how much information a frame provides. Use a **Bayesian Estimator** to update the belief map, trusting "sharp/close" frames more than "blurry/far" frames.

### 3. Parameter Estimation (Least Squares)
Calibrating physical constants mathematically rather than manually.
- **Concept**: The "Calibration Problem" involves solving for the *true* physical parameters that minimize error over time.
- **Application**: Driving the robot in a closed loop and using **Least Squares Estimation (LSE)** on the start/end drift to calculate the *exact* effective wheel diameter (to 0.01mm) and track width. This would significantly reduce systematic odometry errors.

---

## 📄 License

MIT License - See LICENSE file for details.
