# AI Models

This directory contains the YOLO object detection models used by the robot.

## Files

- **`yolo11n_cans.pt`**: PyTorch model trained for detecting beverage cans (Nano v11).
- **`yolo11n_cans_ncnn_model/`**: Optimized NCNN format model folder for faster inference on Raspberry Pi 5 CPU.
- **`yolov8n_cans.pt`**: Previous YOLOv8 model iteration.
- **`yolo26n.pt`**: YOLOv26 model (experimental/latest).
- **`*.onnx`**: ONNX export versions for interoperability.

## Usage

- The `server_native.py` script is configured to look for models in this directory.
- To convert a `.pt` model to NCNN, use the `../convert_to_ncnn.py` script.
