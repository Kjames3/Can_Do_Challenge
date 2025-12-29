# Test Scripts

This folder contains test and verification scripts for various hardware components.

## Files

| File | Description |
|------|-------------|
| `test.py` | General hardware test script |
| `test_astra.py` | Orbbec Astra camera testing |
| `test_camera_detection.py` | Camera + YOLO detection testing |
| `test_yolo_benchmark.py` | YOLO model performance benchmarking |
| `verify_astra_depth_local.py` | Astra depth sensor verification |
| `verify_astra_local.py` | Local Astra camera verification |

---

## Running Tests

### General Hardware Test

```bash
python test.py
```

### Camera Detection Test

```bash
python test_camera_detection.py
```

Tests the camera feed with YOLO detection overlay.

### YOLO Benchmark

```bash
python test_yolo_benchmark.py
```

Measures inference speed across different model sizes and configurations.

### Astra Camera Tests

```bash
# Basic verification
python verify_astra_local.py

# Depth sensor verification
python verify_astra_depth_local.py

# Full Astra test
python test_astra.py
```

---

## See Also

- [Main README](../README.md) - Project overview
- [calibration/](../calibration/) - Hardware calibration scripts
