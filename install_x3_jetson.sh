#!/bin/bash
# =============================================================================
# install_x3_jetson.sh
# Yahboom X3 rover on Jetson Orin Nano — ROS2 Humble full setup
#
# Hardware:
#   - NVIDIA Jetson Orin Nano (JetPack 5.x / Ubuntu 22.04)
#   - Orbbec Astra Pro SC depth camera  (ros2_astra_camera)
#   - YDLidar 4ROS lidar                (ydlidar_ros2_driver)
#   - Yahboom Rosmaster controller board (yahboom_bringup)
#
# Run with: sudo bash install_x3_jetson.sh
# =============================================================================

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC} $1"; }
warn() { echo -e "  ${YELLOW}!!${NC} $1"; }
info() { echo -e "  -- $1"; }

echo "=============================================="
echo -e "  ${BOLD}Yahboom X3 — ROS2 Humble Installer${NC}"
echo "=============================================="
echo ""

# ---------------- root check ----------------
if [ "$EUID" -ne 0 ]; then
    echo "Run as root:  sudo bash install_x3_jetson.sh"
    exit 1
fi

REAL_USER=${SUDO_USER:-$USER}
USER_HOME="/home/$REAL_USER"
ROS2_WS="$USER_HOME/ros2_ws"

echo "User      : $REAL_USER"
echo "Home      : $USER_HOME"
echo "Workspace : $ROS2_WS"

if [ -f /etc/nv_tegra_release ]; then
    echo "JetPack   : $(head -1 /etc/nv_tegra_release)"
fi

# Confirm Ubuntu 22.04 (Humble requirement)
UBUNTU_VERSION=$(. /etc/os-release && echo "$VERSION_ID")
if [ "$UBUNTU_VERSION" != "22.04" ]; then
    warn "Ubuntu 22.04 expected for ROS2 Humble — found $UBUNTU_VERSION"
    warn "Proceeding anyway, but package names may differ."
fi
echo ""

# =============================================================================
# [1/9] Locale (required for ROS2 apt repo)
# =============================================================================
echo "[1/9] Configuring locale..."
apt-get install -y locales > /dev/null
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
ok "Locale set to en_US.UTF-8"
echo ""

# =============================================================================
# [2/9] System dependencies
# =============================================================================
echo "[2/9] Installing system dependencies..."
apt-get update -q
apt-get install -y \
    software-properties-common \
    apt-transport-https \
    gnupg2 \
    lsb-release \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-serial \
    i2c-tools \
    libi2c-dev \
    libusb-1.0-0-dev \
    libopenni2-dev \
    openni2-utils \
    libopencv-dev \
    python3-opencv \
    libopenblas-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    udev \
    minicom \
    htop
ok "System dependencies installed"
echo ""

# =============================================================================
# [3/9] I2C (Jetson — no dtparam, just kernel module)
# =============================================================================
echo "[3/9] Enabling I2C..."
modprobe i2c-dev 2>/dev/null || true
grep -q "^i2c-dev" /etc/modules 2>/dev/null || echo "i2c-dev" >> /etc/modules
ok "i2c-dev module active and persistent"
echo ""

# =============================================================================
# [4/9] ROS2 Humble
# =============================================================================
echo "[4/9] Installing ROS2 Humble..."

# Add ROS2 apt repository
add-apt-repository universe -y > /dev/null 2>&1
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    > /etc/apt/sources.list.d/ros2.list

apt-get update -q

# Desktop install includes rviz2, rqt, and all common packages
apt-get install -y \
    ros-humble-desktop \
    ros-humble-ros-base \
    ros-humble-rosbridge-suite \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-nav2-bringup \
    ros-humble-robot-localization \
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    ros-humble-teleop-twist-keyboard \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-vcstool

ok "ROS2 Humble installed"
echo ""

# Initialise rosdep
echo "  Initialising rosdep..."
rosdep init 2>/dev/null || info "rosdep already initialised"
sudo -u "$REAL_USER" rosdep update
ok "rosdep ready"
echo ""

# =============================================================================
# [5/9] YDLidar C++ SDK  (prereq for ydlidar_ros2_driver)
# =============================================================================
echo "[5/9] Building YDLidar SDK..."
YDLIDAR_SDK_DIR="/opt/YDLidar-SDK"

if [ ! -d "$YDLIDAR_SDK_DIR" ]; then
    git clone --depth=1 https://github.com/YDLIDAR/YDLidar-SDK.git "$YDLIDAR_SDK_DIR"
    mkdir -p "$YDLIDAR_SDK_DIR/build"
    cmake -S "$YDLIDAR_SDK_DIR" -B "$YDLIDAR_SDK_DIR/build" -DCMAKE_BUILD_TYPE=Release
    make -C "$YDLIDAR_SDK_DIR/build" -j"$(nproc)"
    make -C "$YDLIDAR_SDK_DIR/build" install
    ldconfig
    ok "YDLidar SDK built and installed to /usr/local"
else
    ok "YDLidar SDK already present at $YDLIDAR_SDK_DIR"
fi
echo ""

# =============================================================================
# [6/9] udev rules  (Orbbec, YDLidar, Rosmaster)
# =============================================================================
echo "[6/9] Writing udev rules..."

# Orbbec Astra Pro SC
cat > /etc/udev/rules.d/99-orbbec.rules << 'UDEV'
# Orbbec Astra / Astra Pro SC
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="1d27", MODE="0666", GROUP="plugdev"
UDEV
ok "Orbbec udev rules → /etc/udev/rules.d/99-orbbec.rules"

# YDLidar 4ROS  (CP2102: 10c4:ea60 | PL2303: 067b:2303 | CH340: 1a86:7523)
cat > /etc/udev/rules.d/60-ydlidar.rules << 'UDEV'
# YDLidar — stable /dev/ydlidar symlink
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="ydlidar",  MODE="0666", GROUP="dialout"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", SYMLINK+="ydlidar",  MODE="0666", GROUP="dialout"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="ydlidar",  MODE="0666", GROUP="dialout"
UDEV
ok "YDLidar udev rules → /etc/udev/rules.d/60-ydlidar.rules"

# Rosmaster board  (CH340: 1a86:7523 — same chip, different unit)
# If both lidar and rosmaster share the same VID:PID you'll need to
# differentiate by KERNELS (physical USB port) — see post-install note.
cat > /etc/udev/rules.d/61-rosmaster.rules << 'UDEV'
# Yahboom Rosmaster controller board
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="rosmaster", MODE="0666", GROUP="dialout"
KERNEL=="ttyACM*", MODE="0666", GROUP="dialout"
UDEV
ok "Rosmaster udev rules → /etc/udev/rules.d/61-rosmaster.rules"

# Jetson GPIO
cat > /etc/udev/rules.d/99-gpio.rules << 'UDEV'
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", \
    PROGRAM="/bin/sh -c 'chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport ; chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport'"
SUBSYSTEM=="gpio", KERNEL=="gpio*", ACTION=="add", \
    PROGRAM="/bin/sh -c 'chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value ; chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value'"
UDEV
ok "Jetson GPIO udev rules → /etc/udev/rules.d/99-gpio.rules"

udevadm control --reload-rules && udevadm trigger 2>/dev/null || true
echo ""

# =============================================================================
# [7/9] ROS2 workspace — clone + build
# =============================================================================
echo "[7/9] Setting up ROS2 workspace at $ROS2_WS..."

sudo -u "$REAL_USER" mkdir -p "$ROS2_WS/src"

# Source ROS2 for subsequent commands
ROS_SETUP="/opt/ros/humble/setup.bash"
source "$ROS_SETUP"

# ── Clone driver packages ────────────────────────────────────────────────────

echo ""
echo "  Cloning ROS2 driver packages..."

# 1. YDLidar ROS2 driver
YDLIDAR_ROS_DIR="$ROS2_WS/src/ydlidar_ros2_driver"
if [ ! -d "$YDLIDAR_ROS_DIR" ]; then
    sudo -u "$REAL_USER" git clone --depth=1 \
        https://github.com/YDLIDAR/ydlidar_ros2_driver.git \
        "$YDLIDAR_ROS_DIR"
    ok "ydlidar_ros2_driver cloned"
else
    ok "ydlidar_ros2_driver already present"
fi

# 2. Orbbec ros2_astra_camera  (OpenNI2-based — correct driver for Astra Pro SC)
ASTRA_ROS_DIR="$ROS2_WS/src/ros2_astra_camera"
if [ ! -d "$ASTRA_ROS_DIR" ]; then
    sudo -u "$REAL_USER" git clone --depth=1 \
        https://github.com/orbbec/ros2_astra_camera.git \
        "$ASTRA_ROS_DIR"
    ok "ros2_astra_camera cloned"
else
    ok "ros2_astra_camera already present"
fi

# 3. Yahboom X3 ROS2 bringup / Rosmaster driver
#    The official Yahboom repo contains ROS2 packages under the ros2 branch.
#    URL verified at: https://github.com/YahboomTechnology/Rosmaster-X3
YAHBOOM_ROS_DIR="$ROS2_WS/src/Rosmaster-X3"
if [ ! -d "$YAHBOOM_ROS_DIR" ]; then
    sudo -u "$REAL_USER" git clone --depth=1 \
        --branch ros2 \
        https://github.com/YahboomTechnology/Rosmaster-X3.git \
        "$YAHBOOM_ROS_DIR" 2>/dev/null || \
    sudo -u "$REAL_USER" git clone --depth=1 \
        https://github.com/YahboomTechnology/Rosmaster-X3.git \
        "$YAHBOOM_ROS_DIR" && \
    warn "Could not find 'ros2' branch — cloned default branch. Check Yahboom's repo for the correct ROS2 branch."
    ok "Rosmaster-X3 cloned"
else
    ok "Rosmaster-X3 already present"
fi

# ── rosdep install ───────────────────────────────────────────────────────────
echo ""
echo "  Running rosdep install..."
sudo -u "$REAL_USER" bash -c "
    source $ROS_SETUP
    cd $ROS2_WS
    rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
" && ok "rosdep dependencies installed" || warn "rosdep had some failures — check output above"

# ── colcon build ─────────────────────────────────────────────────────────────
echo ""
echo "  Building workspace with colcon (this may take 5–10 minutes)..."
sudo -u "$REAL_USER" bash -c "
    source $ROS_SETUP
    cd $ROS2_WS
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1
" && ok "Workspace built successfully" || warn "Build had errors — see output above"

echo ""

# =============================================================================
# [8/9] Python extras  (YOLO/ultralytics for custom ROS2 detection node)
# =============================================================================
echo "[8/9] Installing Python extras (YOLO)..."

# Install into system Python so ROS2 Python nodes can import without a venv
pip3 install --upgrade pip
pip3 install ultralytics numpy websockets smbus2 pyserial
ok "ultralytics (YOLO), numpy, websockets, smbus2, pyserial installed system-wide"
echo ""

# =============================================================================
# [9/9] User groups + .bashrc
# =============================================================================
echo "[9/9] Setting up groups and shell environment..."

usermod -a -G dialout "$REAL_USER" 2>/dev/null || true
usermod -a -G i2c     "$REAL_USER" 2>/dev/null || true
usermod -a -G video   "$REAL_USER" 2>/dev/null || true
usermod -a -G plugdev "$REAL_USER" 2>/dev/null || true
usermod -a -G gpio    "$REAL_USER" 2>/dev/null || true
ok "User '$REAL_USER' added to: dialout, i2c, video, plugdev, gpio"

# Add ROS2 + workspace sourcing to .bashrc if not already present
BASHRC="$USER_HOME/.bashrc"
if ! grep -q "ros/humble/setup.bash" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << BASHLINES

# === ROS2 Humble (added by install_x3_jetson.sh) ===
source /opt/ros/humble/setup.bash
source $ROS2_WS/install/local_setup.bash 2>/dev/null || true
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Convenient aliases
alias cb='cd $ROS2_WS && colcon build --symlink-install'
alias cs='source $ROS2_WS/install/local_setup.bash'
BASHLINES
    ok ".bashrc updated with ROS2 and workspace sourcing"
else
    ok ".bashrc already has ROS2 sourcing"
fi
echo ""

# =============================================================================
# Verification
# =============================================================================
echo "=============================================="
echo "  Verification"
echo "=============================================="

# ROS2
if [ -f /opt/ros/humble/setup.bash ]; then
    ok "ROS2 Humble installed at /opt/ros/humble"
else
    warn "ROS2 Humble not found at /opt/ros/humble"
fi

# Workspace build artifacts
if [ -d "$ROS2_WS/install" ]; then
    ok "ROS2 workspace built: $ROS2_WS/install"
else
    warn "Workspace install/ not found — build may have failed"
fi

# YDLidar SDK
if ldconfig -p | grep -q libydlidar; then
    ok "YDLidar SDK library found in ldconfig"
else
    warn "YDLidar SDK library not found — driver build may fail"
fi

# I2C
if ls /dev/i2c-* &>/dev/null 2>&1; then
    ok "I2C buses: $(ls /dev/i2c-* | tr '\n' ' ')"
else
    warn "No I2C buses found (reboot may be needed)"
fi

# USB hardware
echo ""
echo "  USB device scan:"
lsusb 2>/dev/null | grep -i -E "orbbec|astra|2bc5|1d27" \
    && ok "Orbbec camera detected" \
    || info "Orbbec camera not currently connected"
[ -e /dev/ydlidar ]   && ok "YDLidar at /dev/ydlidar"    || info "YDLidar not connected (symlink created on plug-in)"
[ -e /dev/rosmaster ] && ok "Rosmaster at /dev/rosmaster" || info "Rosmaster not connected (symlink created on plug-in)"

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Reboot to activate udev rules and group memberships:"
echo "       sudo reboot"
echo ""
echo "  2. After reboot — launch the full sensor stack:"
echo "       ros2 launch yahboom_bringup x3_bringup.launch.py"
echo ""
echo "  3. Launch individual drivers:"
echo "       # YDLidar"
echo "       ros2 launch ydlidar_ros2_driver ydlidar_launch.py"
echo ""
echo "       # Orbbec Astra Pro SC"
echo "       ros2 launch astra_camera astra_pro.launch.xml"
echo ""
echo "       # Check topics"
echo "       ros2 topic list"
echo "       ros2 topic echo /scan               # lidar"
echo "       ros2 topic echo /camera/depth/image  # depth"
echo ""
echo "  4. Visualise in rviz2:"
echo "       rviz2"
echo ""
echo -e "${YELLOW}NOTE:${NC} If YDLidar and Rosmaster share the same USB VID:PID (1a86:7523)"
echo "  both udev symlinks will race. Run 'udevadm info /dev/ttyUSB0' with each"
echo "  device plugged in separately to get their KERNELS (physical port path)"
echo "  and add KERNELS==\"...\" to differentiate them in the udev rules."
echo ""
echo "  Workspace: $ROS2_WS"
echo "  Build alias: cb   (colcon build --symlink-install)"
echo "  Source alias: cs  (source install/local_setup.bash)"
echo ""
