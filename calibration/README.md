# Calibration Scripts

This folder contains utility scripts for calibrating rover hardware.

## Files

| File | Description |
|------|-------------|
| `calibrate_motors.py` | Find minimum motor power to overcome friction |
| `calibrate_focal_length.py` | Calculate camera focal length for distance estimation |

---

## Motor Calibration

**Purpose**: Find the minimum power needed for each motor to actually move the rover. Below this threshold, power is wasted without movement.

**When to run**: After hardware changes, wheel replacements, or if the rover struggles to start moving.

### Usage

```bash
# Default calibration
python calibrate_motors.py

# Custom parameters
python calibrate_motors.py --step 0.02 --max 0.6 --duration 1.0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | 0.01 | Power increment per test step |
| `--max` | 0.5 | Maximum power to test |
| `--duration` | 0.5 | Seconds to hold each power level |

### Output

The script outputs the minimum power for each motor direction (forward/backward). Update `DRIFT_COMPENSATION` in `server_native.py` if motors have significantly different thresholds.

---

## Focal Length Calibration

**Purpose**: Calculate the correct `FOCAL_LENGTH` value for accurate distance estimation using the pinhole camera model.

**When to run**: When changing cameras, camera resolution, or if distance estimates are inaccurate.

### Prerequisites

1. A standard 12oz soda can (12.0 cm height)
2. Measuring tape

### Usage

```bash
python calibrate_focal_length.py
```

### Process

1. Place a soda can at exactly 50cm from the camera
2. Run the script
3. The script detects the can, measures its pixel height, and calculates focal length
4. Update `FOCAL_LENGTH` in `server_native.py` with the calculated value

### Formula

```
FOCAL_LENGTH = (pixel_height × actual_distance) / actual_height
```

---

## See Also

- [Main README](../README.md) - Project overview
- [server_native.py](../server_native.py) - Uses calibrated values
