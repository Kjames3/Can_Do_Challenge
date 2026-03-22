#!/bin/bash
# =============================================================================
# install_x3_jetson.sh - Install dependencies for Yahboom X3 on Jetson Orin Nano
#
# Hardware:
#   - Yahboom X3 rover chassis
#   - NVIDIA Jetson Orin Nano (JetPack 5.x or 6.x)
#   - Orbbec Astra Pro SC depth camera
#   - YDLidar 4ROS lidar
#   - Rosmaster controller board (motors/servos via serial)
#
# Run with: sudo bash install_x3_jetson.sh
# =============================================================================

set -e

echo "=============================================="
echo "  Yahboom X3 Jetson Orin Nano - Installer"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash install_x3_jetson.sh"
    exit 1
fi

# Get the non-root user who called sudo
REAL_USER=${SUDO_USER:-$USER}
echo "Installing for user: $REAL_USER"

# Detect JetPack version for awareness (non-blocking)
JETPACK_VERSION="unknown"
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(head -1 /etc/nv_tegra_release)
fi
echo "JetPack: $JETPACK_VERSION"
echo ""

# =============================================================================
# [1/7] System Dependencies
# =============================================================================
echo "[1/7] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopencv-dev \
    python3-opencv \
    i2c-tools \
    libi2c-dev \
    libopenblas-dev \
    libhdf5-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libusb-1.0-0-dev \
    udev \
    python3-serial \
    minicom

echo "  OK System dependencies installed"
echo ""

# =============================================================================
# [2/7] I2C Setup (Jetson style - no dtparam)
# =============================================================================
echo "[2/7] Enabling I2C..."
modprobe i2c-dev 2>/dev/null || true

# Persist i2c-dev across reboots
if ! grep -q "^i2c-dev" /etc/modules 2>/dev/null; then
    echo "i2c-dev" >> /etc/modules
    echo "  OK i2c-dev added to /etc/modules"
else
    echo "  OK i2c-dev already in /etc/modules"
fi

# Set permissions on I2C buses
for i2c_bus in /dev/i2c-*; do
    chmod 666 "$i2c_bus" 2>/dev/null || true
done
echo ""

# =============================================================================
# [3/7] Orbbec Astra Pro SC Camera (OpenNI2 + pyorbbecsdk)
# =============================================================================
echo "[3/7] Setting up Orbbec Astra Pro SC camera..."

# Install libopenni2 for depth camera access
apt-get install -y libopenni2-dev openni2-utils 2>/dev/null || \
    echo "  NOTE: openni2-utils not in repo - install manually from Orbbec SDK"

# Install udev rules for Orbbec cameras
ORBBEC_UDEV_RULES="/etc/udev/rules.d/99-orbbec.rules"
if [ ! -f "$ORBBEC_UDEV_RULES" ]; then
    cat > "$ORBBEC_UDEV_RULES" << 'UDEV'
# Orbbec Astra Pro SC
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="1d27", MODE="0666", GROUP="plugdev"
UDEV
    echo "  OK Orbbec udev rules written to $ORBBEC_UDEV_RULES"
else
    echo "  OK Orbbec udev rules already exist"
fi

udevadm control --reload-rules && udevadm trigger 2>/dev/null || true
echo ""

# =============================================================================
# [4/7] YDLidar 4ROS - udev rule + SDK
# =============================================================================
echo "[4/7] Setting up YDLidar 4ROS..."

YDLIDAR_UDEV="/etc/udev/rules.d/60-ydlidar.rules"
if [ ! -f "$YDLIDAR_UDEV" ]; then
    cat > "$YDLIDAR_UDEV" << 'UDEV'
# YDLidar - create stable symlink at /dev/ydlidar
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
    SYMLINK+="ydlidar", MODE="0666", GROUP="dialout"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", \
    SYMLINK+="ydlidar", MODE="0666", GROUP="dialout"
# Fallback: any ttyUSB gets dialout-accessible
KERNEL=="ttyUSB*", MODE="0666", GROUP="dialout"
UDEV
    echo "  OK YDLidar udev rules written to $YDLIDAR_UDEV"
else
    echo "  OK YDLidar udev rules already exist"
fi

udevadm control --reload-rules && udevadm trigger 2>/dev/null || true
echo ""

# =============================================================================
# [5/7] Rosmaster Controller Board - udev rule
# =============================================================================
echo "[5/7] Setting up Rosmaster controller board..."

ROSMASTER_UDEV="/etc/udev/rules.d/61-rosmaster.rules"
if [ ! -f "$ROSMASTER_UDEV" ]; then
    cat > "$ROSMASTER_UDEV" << 'UDEV'
# Yahboom Rosmaster controller board - CH340/CP210x USB-serial
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", \
    SYMLINK+="rosmaster", MODE="0666", GROUP="dialout"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
    SYMLINK+="rosmaster", MODE="0666", GROUP="dialout"
# Rosmaster may also enumerate on ttyACM
KERNEL=="ttyACM*", MODE="0666", GROUP="dialout"
UDEV
    echo "  OK Rosmaster udev rules written to $ROSMASTER_UDEV"
else
    echo "  OK Rosmaster udev rules already exist"
fi

udevadm control --reload-rules && udevadm trigger 2>/dev/null || true
echo ""

# =============================================================================
# [6/7] Python Virtual Environment + Packages
# =============================================================================
VENV_PATH="/home/$REAL_USER/viam_projects/.venv"
echo "[6/7] Setting up Python venv at $VENV_PATH..."

if [ ! -d "$VENV_PATH" ]; then
    sudo -u "$REAL_USER" python3 -m venv --system-site-packages "$VENV_PATH"
    echo "  OK Virtual environment created"
else
    echo "  OK Virtual environment already exists"
fi

sudo -u "$REAL_USER" bash << PYEOF
source "$VENV_PATH/bin/activate"
pip install --upgrade pip

echo ""
echo "  -- Core packages --"
pip install numpy
pip install websockets
pip install pyserial           # Rosmaster + YDLidar serial comms

echo ""
echo "  -- Computer vision --"
# opencv-python-headless: use system libopencv on Jetson for CUDA support
pip install opencv-python-headless || echo "  NOTE: falling back to system opencv"

echo ""
echo "  -- YOLO inference --"
# On Jetson, prefer TensorRT backend. ultralytics handles this automatically.
pip install ultralytics

echo ""
echo "  -- Depth camera (Orbbec Astra Pro SC) --"
# pyorbbecsdk wraps the Orbbec SDK; open3d is a heavier alternative
pip install pyorbbecsdk || echo "  NOTE: pyorbbecsdk not on PyPI - install from Orbbec GitHub (see notes below)"
# open3d provides OrbbecSDK support on Jetson JetPack 5+
pip install open3d || echo "  NOTE: open3d optional (heavy), pyorbbecsdk preferred"

echo ""
echo "  -- YDLidar --"
pip install ydlidar || echo "  NOTE: ydlidar PyPI package unavailable - use ydlidar-sdk C++ bindings (see notes)"

echo ""
echo "  -- Rosmaster controller board --"
# Yahboom provides a Python library for the Rosmaster board
pip install pyserial
# Yahboom SDK is typically installed from their repo, not PyPI:
pip install Rosmaster-Lib 2>/dev/null || echo "  NOTE: Rosmaster-Lib not on PyPI - clone from Yahboom GitHub (see notes)"

echo ""
echo "  -- IMU / I2C --"
pip install smbus2
pip install mpu6050-raspberrypi || true  # works on any Linux I2C

echo ""
echo "  -- Jetson GPIO (replaces gpiozero/rpi-lgpio) --"
pip install Jetson.GPIO || echo "  NOTE: Jetson.GPIO may need manual setup (see JetPack docs)"

echo ""
echo "  -- INA219 power sensor --"
pip install pi-ina219

echo "  OK Python packages installed"
PYEOF

echo ""

# =============================================================================
# [7/7] User Groups & Permissions
# =============================================================================
echo "[7/7] Setting up user permissions..."

usermod -a -G dialout  "$REAL_USER" 2>/dev/null || true   # serial ports (lidar, rosmaster)
usermod -a -G i2c      "$REAL_USER" 2>/dev/null || true   # I2C (IMU, misc)
usermod -a -G video    "$REAL_USER" 2>/dev/null || true   # camera
usermod -a -G plugdev  "$REAL_USER" 2>/dev/null || true   # USB devices (Orbbec)
usermod -a -G gpio     "$REAL_USER" 2>/dev/null || true   # Jetson GPIO

# Jetson.GPIO needs a udev rule so normal users can access GPIO
JETSON_GPIO_UDEV="/etc/udev/rules.d/99-gpio.rules"
if [ ! -f "$JETSON_GPIO_UDEV" ]; then
    cat > "$JETSON_GPIO_UDEV" << 'UDEV'
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", \
    PROGRAM="/bin/sh -c 'chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport; chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport'"
SUBSYSTEM=="gpio", KERNEL=="gpio*", ACTION=="add", \
    PROGRAM="/bin/sh -c 'chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value; chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value'"
UDEV
    echo "  OK Jetson GPIO udev rules written"
fi

udevadm control --reload-rules && udevadm trigger 2>/dev/null || true
echo "  OK User '$REAL_USER' added to: dialout, i2c, video, plugdev, gpio"
echo ""

# =============================================================================
# Verification
# =============================================================================
echo "=============================================="
echo "  Verification"
echo "=============================================="

# I2C
if ls /dev/i2c-* &>/dev/null; then
    echo "OK I2C buses: $(ls /dev/i2c-*)"
    echo "   Scanning /dev/i2c-7 (Jetson Orin Nano main bus)..."
    i2cdetect -y 7 2>/dev/null | head -12 || \
    i2cdetect -y 1 2>/dev/null | head -12 || \
    echo "   (bus scan unavailable)"
else
    echo "!! I2C bus not found"
fi

# Camera / Orbbec (USB)
echo ""
echo "USB devices:"
lsusb 2>/dev/null | grep -i -E "orbbec|astra|2bc5|1d27" && \
    echo "   OK Orbbec camera detected" || \
    echo "   -- Orbbec camera not currently connected"

# YDLidar
if [ -e /dev/ydlidar ]; then
    echo "OK YDLidar at /dev/ydlidar"
elif ls /dev/ttyUSB* &>/dev/null 2>&1; then
    echo "OK Serial devices: $(ls /dev/ttyUSB*) (ydlidar udev symlink pending reconnect)"
else
    echo "-- YDLidar not connected (plug in to verify)"
fi

# Rosmaster
if [ -e /dev/rosmaster ]; then
    echo "OK Rosmaster board at /dev/rosmaster"
else
    echo "-- Rosmaster board symlink not active (plug in to verify)"
fi

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "IMPORTANT - Manual steps still required:"
echo ""
echo "  1. Orbbec Astra Pro SC SDK (if pyorbbecsdk failed):"
echo "       git clone https://github.com/orbbec/pyorbbecsdk"
echo "       cd pyorbbecsdk && pip install ."
echo ""
echo "  2. YDLidar SDK (if ydlidar pip failed):"
echo "       git clone https://github.com/YDLIDAR/YDLidar-SDK"
echo "       cd YDLidar-SDK && mkdir build && cd build"
echo "       cmake .. && make -j4 && sudo make install"
echo "       cd ../python && pip install ."
echo ""
echo "  3. Rosmaster Python library (if Rosmaster-Lib pip failed):"
echo "       git clone https://github.com/YahboomTechnology/Rosmaster-X3"
echo "       # Follow Yahboom's README for the Python SDK install"
echo ""
echo "  4. Reboot to activate group memberships and udev rules:"
echo "       sudo reboot"
echo ""
echo "  5. After reboot, activate venv and run:"
echo "       source $VENV_PATH/bin/activate"
echo "       python server_native.py --sim   # test without hardware first"
echo ""
echo "Lidar port: check 'ls /dev/ttyUSB*' or use /dev/ydlidar symlink"
echo "Rosmaster port: check 'ls /dev/ttyUSB*' or use /dev/rosmaster symlink"
echo ""
