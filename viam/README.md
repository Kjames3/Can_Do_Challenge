# Viam SDK Server Implementations

This folder contains server implementations that use the **Viam SDK** to control the rover through Viam's cloud platform.

## Files

| File | Description |
|------|-------------|
| `server_pi.py` | Raspberry Pi 5 optimized server with Viam SDK |
| `server_jetson.py` | Jetson Orin Nano optimized server with Viam SDK (GPU acceleration) |

## Why Viam SDK?

The Viam SDK provides:
- Cloud-based robot configuration and monitoring
- Easy component setup through the Viam web app
- Remote access and fleet management

## Limitations

The Viam SDK has a **100 API calls/second limit**, which can cause timeouts during high-frequency operations like real-time motor control with sensor feedback. This is why the **native implementation** (`server_native.py` in the root folder) was developed.

## When to Use These

Use the Viam SDK implementations when:
- You need cloud-based monitoring/logging
- You want easy remote access through Viam's platform
- You're testing with the Viam simulator
- API call rate isn't a bottleneck for your use case

## Configuration

Edit the following in each script before running:

```python
ROBOT_ADDRESS = "your-robot.viam.cloud"
API_KEY_ID = "your-api-key-id"
API_KEY = "your-api-key"
```

## Usage

```bash
# Raspberry Pi
python server_pi.py

# Jetson (with GPU)
python server_jetson.py

# Simulation mode (no hardware)
python server_pi.py --sim
```

## See Also

- [Main README](../README.md) - Project overview and native implementation
- [server_native.py](../server_native.py) - Native implementation without Viam SDK limits
