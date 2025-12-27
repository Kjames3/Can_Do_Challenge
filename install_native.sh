#!/bin/bash
# =============================================================================
# install_native.sh - Install dependencies for server_native.py
# 
# For Raspberry Pi 5 (or other Pi models with GPIO)
# Run with: sudo bash install_native.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Native Pi Server - Dependency Installer"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root: sudo bash install_native.sh"
    exit 1
fi

# Get the non-root user who called sudo
REAL_USER=${SUDO_USER:-$USER}
echo "Installing for user: $REAL_USER"
echo ""

# =============================================================================
# System Dependencies
# =============================================================================
echo "[1/5] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    i2c-tools \
    libi2c-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp-dev \
    libtiff5-dev \
    libjasper-dev \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libcap-dev \
    python3-picamera2

echo "  ✓ System dependencies installed"
echo ""

# =============================================================================
# Enable I2C (for IMU)
# =============================================================================
echo "[2/5] Enabling I2C interface..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null && \
   ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    # Try both locations (Pi 4 vs Pi 5)
    if [ -f /boot/firmware/config.txt ]; then
        echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
    else
        echo "dtparam=i2c_arm=on" >> /boot/config.txt
    fi
    echo "  ✓ I2C enabled (reboot required)"
else
    echo "  ✓ I2C already enabled"
fi

# Load I2C kernel module now
modprobe i2c-dev 2>/dev/null || true
echo ""

# =============================================================================
# Python Virtual Environment
# =============================================================================
VENV_PATH="/home/$REAL_USER/Can_Do_Challenge/.venv"

echo "[3/5] Setting up Python virtual environment..."
if [ ! -d "$VENV_PATH" ]; then
    sudo -u $REAL_USER python3 -m venv "$VENV_PATH"
    echo "  ✓ Virtual environment created at $VENV_PATH"
else
    echo "  ✓ Virtual environment already exists"
fi
echo ""

# =============================================================================
# Python Packages
# =============================================================================
echo "[4/5] Installing Python packages..."

# Activate venv and install packages
sudo -u $REAL_USER bash << EOF
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Core dependencies
pip install numpy
pip install opencv-python-headless  # Headless for server (no GUI)
pip install websockets
pip install ultralytics  # YOLO11/YOLOv8

# GPIO libraries (try multiple for Pi 5 compatibility)
pip install gpiozero
pip install rpi-lgpio || pip install lgpio || echo "Note: lgpio install failed, will fallback to pigpio"
pip install pigpio || echo "Note: pigpio install optional"

# I2C for IMU
pip install smbus2

# LIDAR (if using rplidar)
pip install rplidar-roboticia || echo "Note: rplidar optional"

# Camera (New)
pip install picamera2

echo "  ✓ Python packages installed"
EOF

echo ""

# =============================================================================
# Permissions
# =============================================================================
echo "[5/5] Setting up permissions..."

# Add user to required groups
usermod -a -G gpio $REAL_USER 2>/dev/null || true
usermod -a -G i2c $REAL_USER 2>/dev/null || true
usermod -a -G video $REAL_USER 2>/dev/null || true
usermod -a -G dialout $REAL_USER 2>/dev/null || true  # For LIDAR serial

echo "  ✓ User added to gpio, i2c, video, dialout groups"
echo ""

# =============================================================================
# Verify Installation
# =============================================================================
echo "=============================================="
echo "  Verifying Installation"
echo "=============================================="

# Check I2C
if [ -e /dev/i2c-1 ]; then
    echo "✓ I2C bus available at /dev/i2c-1"
    echo "  Scanning for devices..."
    i2cdetect -y 1 2>/dev/null | head -10 || echo "  (i2cdetect not available)"
else
    echo "⚠ I2C bus not found - reboot may be required"
fi

# Check camera
if [ -e /dev/video0 ]; then
    echo "✓ Camera available at /dev/video0"
else
    echo "⚠ Camera not found at /dev/video0"
fi

# Check LIDAR
if [ -e /dev/ttyUSB0 ]; then
    echo "✓ LIDAR likely at /dev/ttyUSB0"
else
    echo "⚠ LIDAR not found at /dev/ttyUSB0"
fi

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Reboot: sudo reboot"
echo "  2. Activate venv: source $VENV_PATH/bin/activate"
echo "  3. Run server: python server_native.py"
echo ""
echo "Copy model to Pi:"
echo "  scp yolo11n_cans.pt $REAL_USER@<PI_IP>:~/Can_Do_Challenge/"
echo ""
