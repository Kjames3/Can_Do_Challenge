
import numpy as np

# =============================================================================
# ROBOT PARAMETERS (from Viam config)
# =============================================================================
# CALIBRATION NOTE: If robot stops short of target:
#   - Increase WHEEL_CIRCUMFERENCE_MM (odometry undercounts)
# If robot goes past target:
#   - Decrease WHEEL_CIRCUMFERENCE_MM (odometry overcounts)
# Wheel diameter: 68mm -> Actual circumference: 213.6mm
# Encoder appears to give ~12 ticks per wheel rotation
# So: 213.6 / 12 ≈ 18mm per encoder "revolution"
WHEEL_CIRCUMFERENCE_MM = 213.6     # mm per encoder revolution (68mm wheel / ~12 ticks)
WHEEL_BASE_MM = 356             # mm (width between wheels)
WHEEL_DIAMETER_CM = WHEEL_CIRCUMFERENCE_MM / (np.pi * 10)  # Convert to cm
WHEEL_BASE_CM = WHEEL_BASE_MM / 10

class RobotState:
    """Track robot pose using wheel encoder odometry with optional IMU heading fusion."""
    
    def __init__(self):
        self.x = 0.0  # cm
        self.y = 0.0  # cm
        self.theta = 0.0  # radians
        self.last_left_pos = 0.0
        self.last_right_pos = 0.0
        self.initialized = False
    
    def update(self, left_pos, right_pos):
        """Update pose from encoder positions (in revolutions)."""
        if not self.initialized:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.initialized = True
            return
        
        # Delta in revolutions
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        # Convert to distance (cm)
        left_dist = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        right_dist = right_delta * WHEEL_CIRCUMFERENCE_MM / 10
        
        # Differential drive kinematics
        linear = (left_dist + right_dist) / 2.0
        angular = (right_dist - left_dist) / WHEEL_BASE_CM
        
        # Update pose
        self.x += linear * np.sin(self.theta)
        self.y += linear * np.cos(self.theta)
        self.theta += angular
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos
    
    def update_with_imu(self, left_pos, right_pos, imu_heading):
        """
        Update pose using encoder distance + IMU heading (more accurate).
        
        Args:
            left_pos: Left encoder position in revolutions
            right_pos: Right encoder position in revolutions
            imu_heading: Heading from IMU in radians
        """
        if not self.initialized:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.initialized = True
            return
        
        # Delta in revolutions
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        # Convert to distance (cm)
        left_dist = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        right_dist = right_delta * WHEEL_CIRCUMFERENCE_MM / 10
        
        # Average distance for forward movement
        distance = (left_dist + right_dist) / 2
        
        # Use IMU heading directly (much more accurate than encoder-derived)
        self.theta = imu_heading
        
        # Update position using IMU heading
        # Convention: Y is forward (cos), X is lateral (sin)
        self.x += distance * np.sin(self.theta)
        self.y += distance * np.cos(self.theta)
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos
