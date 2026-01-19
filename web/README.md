# Web Interface

This directory contains the frontend code for the robot's control interface.

## Files

- **`GUI.html`**: The main dashboard for controlling the robot.
  - Features: Camera feed, Joystick control, Telemetry display, Map visualization, Settings.
  - Open this file in a web browser to connect to the robot.

- **`main.js`**: The core JavaScript logic for the GUI.
  - Handles WebSocket connection, gamepad input, UI updates, and 3D visualization (Three.js).

- **`style.css`**: Styling for the GUI interface.

- **`debug_gui.html`**: A stripped-down version of the interface used for debugging connection or low-level issues.

## Usage

1. Start the robot server (`server_native.py` or `yahboom/server_x3.py`).
2. Open `GUI.html` in your browser.
3. Enter the Robot's IP address and click Connect.
