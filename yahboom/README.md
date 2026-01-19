# Yahboom X3 Drivers

This directory contains drivers and server implementations specific to the **Yahboom X3** robot platform (Jetson Orin Nano).

## Files

- **`server_x3.py`**: The main WebSocket server for the X3 robot. It handles motor control (mecanum drive), sensor readings (Lidar, Camera), and communicates with the web interface.
  - usage: `python yahboom/server_x3.py`
  
- **`drivers_x3.py`**: Hardware drivers for the X3.
  - `Rosmaster`: Handles serial communication with the robot's MCU (Motor Control Unit).
  - `MecanumDrive`: Implements holonomic movement logic (strafe, rotate, drive).
  - `YDLidarDriver`: Interface for the 2D Lidar sensor.

- **`test_x3_motors.py`**: A simple script to verify motor movement and direction without running the full server.

## Dependencies

- These scripts rely on `drivers.py` and `robot_state.py` from the root directory.
- Requires `pyserial` for MCU communication.
