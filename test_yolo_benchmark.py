"""
YOLOv8 Performance Benchmark for Jetson Orin Nano vs Raspberry Pi 4

This script benchmarks YOLO inference performance with:
- Multiple model sizes (yolov8n, yolov8s)
- Multiple resolutions (320, 480, 640)
- FPS measurement and GPU detection

Usage:
    python test_yolo_benchmark.py
"""

import cv2
import time
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to test (will download automatically if not present)
MODELS_TO_TEST = ['yolov8n.pt', 'yolov8s.pt']

# Resolutions to test
RESOLUTIONS = [320, 480, 640]

# Number of warmup frames (not counted in benchmark)
WARMUP_FRAMES = 10

# Number of frames to benchmark
BENCHMARK_FRAMES = 100

# Target classes for detection (bottle, cup)
TARGET_CLASSES = [39, 41]

# Known object heights for distance estimation
KNOWN_HEIGHT_BOTTLE = 20.0  # cm
KNOWN_HEIGHT_CAN = 12.0     # cm
FOCAL_LENGTH = 600


# =============================================================================
# SYSTEM INFO & GPU DIAGNOSTICS
# =============================================================================

def print_system_info():
    """Print GPU and system information with detailed diagnostics."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION & GPU DIAGNOSTICS")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU with a simple tensor operation
            try:
                test_tensor = torch.zeros(1).cuda()
                print(f"GPU Test: ✓ Tensor successfully moved to CUDA")
                device = "GPU (CUDA)"
            except Exception as e:
                print(f"GPU Test: ✗ Failed - {e}")
                device = "CPU (CUDA available but failed)"
        else:
            print("\n⚠️  CUDA NOT AVAILABLE!")
            print("   This means PyTorch was NOT installed with GPU support.")
            print("   You need to install the Jetson-specific PyTorch wheel.")
            print("\n   Fix: Run these commands:")
            print("   pip uninstall torch")
            print("   pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl")
            device = "CPU (CUDA not available)"
    else:
        print("PyTorch: Not installed")
        device = "CPU (PyTorch not available)"
    
    print(f"\nOpenCV Version: {cv2.__version__}")
    print(f"Inference Device: {device}")
    print("=" * 60 + "\n")
    return device


def load_model_with_gpu(model_name):
    """
    Load YOLO model and explicitly move to GPU if available.
    Returns (model, device_name)
    """
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Force model to GPU
        model.to('cuda')
        
        # Verify model is on GPU
        try:
            device = next(model.model.parameters()).device
            print(f"✓ Model loaded on: {device}")
            if 'cuda' in str(device):
                return model, "GPU"
            else:
                print("⚠️  Model is NOT on GPU!")
        except Exception as e:
            print(f"Could not verify device: {e}")
    else:
        print("⚠️  Running on CPU - this will be slow!")
    
    return model, "CPU"


# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================

def benchmark_model(model, cap, resolution, num_frames=100):
    """
    Benchmark a YOLO model at a specific resolution.
    
    Returns:
        dict with fps, avg_inference_ms, min_ms, max_ms
    """
    inference_times = []
    detections_count = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Reset camera if we run out of frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        # Time the inference
        start = time.perf_counter()
        results = model(frame, imgsz=resolution, verbose=False)
        end = time.perf_counter()
        
        inference_times.append((end - start) * 1000)  # Convert to ms
        
        # Count detections
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in TARGET_CLASSES:
                    detections_count += 1
    
    if not inference_times:
        return None
    
    avg_ms = np.mean(inference_times)
    return {
        'fps': 1000 / avg_ms,
        'avg_ms': avg_ms,
        'min_ms': np.min(inference_times),
        'max_ms': np.max(inference_times),
        'std_ms': np.std(inference_times),
        'detections': detections_count
    }


# =============================================================================
# LIVE DEMO MODE
# =============================================================================

def run_live_demo(model_name='yolov8s.pt', resolution=640):
    """Run live detection with FPS display (like test_camera_detection.py)."""
    print(f"\n📹 Starting Live Demo: {model_name} @ {resolution}px")
    print("Press 'q' to quit, 'n' for yolov8n, 's' for yolov8s")
    print("Press '1' for 320px, '2' for 480px, '3' for 640px\n")
    
    # Load model with GPU if available
    model, device_type = load_model_with_gpu(model_name)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    current_model_name = model_name
    current_res = resolution
    fps_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time inference
        start = time.perf_counter()
        results = model(frame, imgsz=current_res, verbose=False)
        inference_time = (time.perf_counter() - start) * 1000
        
        # Calculate FPS
        fps = 1000 / inference_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        # Draw detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in TARGET_CLASSES:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    
                    # Distance estimation
                    height_px = y2 - y1
                    real_height = KNOWN_HEIGHT_BOTTLE if label == 'bottle' else KNOWN_HEIGHT_CAN
                    if height_px > 0:
                        distance_cm = (real_height * FOCAL_LENGTH) / height_px
                        distance_in = distance_cm / 2.54
                    else:
                        distance_cm = 0
                        distance_in = 0
                    
                    # Draw box and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"{distance_cm:.0f}cm ({distance_in:.1f}in)", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw FPS and model info (with device type)
        info_text = f"{current_model_name} @ {current_res}px | Device: {device_type}"
        fps_text = f"FPS: {avg_fps:.1f} | Inference: {inference_time:.1f}ms"
        
        # Color code based on device (green for GPU, red for CPU)
        fps_color = (0, 255, 0) if device_type == "GPU" else (0, 0, 255)
        
        cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        cv2.imshow('YOLO Benchmark - Press Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            model, device_type = load_model_with_gpu('yolov8n.pt')
            current_model_name = 'yolov8n.pt'
            fps_history.clear()
        elif key == ord('s'):
            model, device_type = load_model_with_gpu('yolov8s.pt')
            current_model_name = 'yolov8s.pt'
            fps_history.clear()
        elif key == ord('1'):
            current_res = 320
            fps_history.clear()
        elif key == ord('2'):
            current_res = 480
            fps_history.clear()
        elif key == ord('3'):
            current_res = 640
            fps_history.clear()
    
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = print_system_info()
    
    print("Select mode:")
    print("  1. Run full benchmark (all models & resolutions)")
    print("  2. Live demo with FPS display")
    print("  3. Quick benchmark (yolov8s @ 640 only)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '2':
        run_live_demo()
        return
    
    # Open camera for benchmark
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\n📊 Starting Benchmark...")
    print("-" * 70)
    
    results_table = []
    
    if choice == '3':
        # Quick benchmark
        models = ['yolov8s.pt']
        resolutions = [640]
    else:
        # Full benchmark
        models = MODELS_TO_TEST
        resolutions = RESOLUTIONS
    
    for model_name in models:
        print(f"\n🔄 Loading {model_name}...")
        model = YOLO(model_name)
        
        # Warmup
        print(f"   Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            ret, frame = cap.read()
            if ret:
                model(frame, imgsz=320, verbose=False)
        
        for res in resolutions:
            print(f"   Testing {model_name} @ {res}px...", end=" ", flush=True)
            
            result = benchmark_model(model, cap, res, BENCHMARK_FRAMES)
            
            if result:
                print(f"✓ {result['fps']:.1f} FPS ({result['avg_ms']:.1f}ms)")
                results_table.append({
                    'model': model_name,
                    'resolution': res,
                    **result
                })
            else:
                print("✗ Failed")
    
    cap.release()
    
    # Print results table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'Resolution':<12} {'FPS':<10} {'Avg (ms)':<12} {'Min-Max (ms)':<15}")
    print("-" * 70)
    
    for r in results_table:
        print(f"{r['model']:<15} {r['resolution']:<12} {r['fps']:<10.1f} "
              f"{r['avg_ms']:<12.1f} {r['min_ms']:.1f}-{r['max_ms']:.1f}")
    
    print("-" * 70)
    print(f"Device: {device}")
    print(f"Frames per test: {BENCHMARK_FRAMES}")
    print("=" * 70)
    
    # Performance summary
    if len(results_table) >= 2:
        best = max(results_table, key=lambda x: x['fps'])
        print(f"\n🏆 Best: {best['model']} @ {best['resolution']}px = {best['fps']:.1f} FPS")


if __name__ == "__main__":
    main()
