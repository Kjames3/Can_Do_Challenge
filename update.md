# Viam Rover Project - Update Log

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
