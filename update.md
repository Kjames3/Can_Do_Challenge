# Viam Rover Project - Update Log

## 2024-12-24 13:45 PST

### Changes Made
1. **MPU6050 IMU Integration**
   - Added `NativeIMU` class for direct I2C access to MPU6050
   - Gyroscope integration for accurate heading tracking
   - Auto-calibration on startup
   - Tilt detection from accelerometer

2. **Drift Compensation (Encoder + IMU)**
   - IMU heading used to maintain straight line during auto-drive
   - Proportional controller adjusts motor power to correct drift
   - Config: `DRIFT_CORRECTION_GAIN = 0.02`

3. **Heading-Based Odometry**
   - 3D viewer now uses IMU heading instead of encoder-derived heading
   - Much more accurate position tracking
   - New method: `robot_state.update_with_imu()`

4. **Tilt Safety**
   - Emergency stop if pitch or roll > 30°
   - Prevents damage from falling off edges
   - Config: `MAX_TILT_DEGREES = 30.0`

5. **Stuck Detection**
   - Detects when motors running but no movement for 1.5s
   - Uses both encoder velocity and IMU acceleration
   - Logs warning (can optionally auto-stop)

### Dependencies to Install on Pi
```bash
pip install smbus2
```

---

## 2023-12-23 21:42 PST

### Changes Made
1. **Added Image Capture Button to GUI**
   - New "📸 Capture" button in the camera feed card
   - Displays count of saved images
   - Images saved to `captured_images/` folder on Pi

2. **Added Capture Handler to server_native.py**
   - Handles `capture_image` WebSocket message
   - Saves JPEG with timestamp (e.g., `can_20231223_214200_123456.jpg`)
   - Sends back capture count to GUI

3. **Updated Training Script to YOLO11**
   - Changed from YOLOv8n to YOLO11n (lighter, faster)
   - Model output: `yolo11n_cans.pt`
   - Better performance on Raspberry Pi hardware

4. **Removed STL Loader from GUI**
   - Fixed CORS error when loading 3D robot model
   - Now uses simple box geometry instead

### Why
- User requested image capture feature to collect ~50 training images from rover's perspective
- YOLO11 recommended for better performance on Pi hardware
- STL loader was causing CORS errors when opening GUI from file://

### To Retrieve Captured Images
```bash
scp -r besto@<PI_IP>:~/Can_Do_Challenge/captured_images ./captured_images
```

---

## Previous Updates

*(Add older updates below as needed)*
