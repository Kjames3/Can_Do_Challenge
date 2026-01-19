
import logging
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)

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
# CALIBRATED VALUE (Effective Circumference)
# This represents the distance traveled per 1000 encoder ticks.
# Derived from calibration (1.36mm/12 ticks) -> 113.3mm/1000 ticks.
WHEEL_CIRCUMFERENCE_MM = 113.3
# WHEEL_DIAMETER_CM = 6.8  # Physical diameter (kept for reference, do not use for calc)
WHEEL_BASE_MM = 356             # mm (width between wheels)
WHEEL_BASE_CM = WHEEL_BASE_MM / 10

class RobotState:
    """
    Track robot pose using an Extended Kalman Filter (EKF).
    Fuses wheel encoder odometry (Prediction) with IMU heading (Update).
    """
    
    def __init__(self):
        # State vector [x, y, theta]
        self.x = 0.0  # cm
        self.y = 0.0  # cm
        self.theta = 0.0  # radians
        
        # EKF Covariance Matrix (Uncertainty)
        # Initial high confidence in starting at (0,0,0)
        self.P = np.eye(3) * 0.1
        
        # Process Noise Covariance (Q) - Uncertainty in model/encoders
        # Tunable: How much we trust the physics prediction vs measurement
        self.Q = np.diag([0.5, 0.5, 0.1])  
        
        # Measurement Noise Covariance (R) - Uncertainty in IMU
        # Tunable: Lower = trust IMU more
        self.R = np.array([[0.05]])  # ~3 degrees variance
        
        self.last_left_pos = 0.0
        self.last_right_pos = 0.0
        self.initialized = False
    
    def update(self, left_pos, right_pos):
        """Standard update (Prediction only) if no IMU data available."""
        if not self.initialized:
            self._init_encoders(left_pos, right_pos)
            return
        
        # Calculate Odometry Control Input (u)
        d_left, d_right = self._get_wheel_deltas(left_pos, right_pos)
        
        # EKF Prediction Step
        self._predict(d_left, d_right)
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos

    def update_with_imu(self, left_pos, right_pos, imu_heading):
        """
        EKF Full Cycle:
        1. Prediction Step (using encoders)
        2. Update Step (using IMU heading)
        """
        if not self.initialized:
            self._init_encoders(left_pos, right_pos)
            # Initialize theta to first IMU reading if desired, 
            # but usually we assume start at 0 or match calibration.
            # self.theta = imu_heading 
            return
        
        # 1. Prediction (Odometry)
        d_left, d_right = self._get_wheel_deltas(left_pos, right_pos)
        self._predict(d_left, d_right)
        
        # 2. Update (Correction)
        self._correct(imu_heading)
        
        self.last_left_pos = left_pos
        self.last_right_pos = right_pos

    def _init_encoders(self, left, right):
        self.last_left_pos = left
        self.last_right_pos = right
        self.initialized = True

    def _get_wheel_deltas(self, left_pos, right_pos):
        """Calculate distance moved by each wheel in cm."""
        left_delta = left_pos - self.last_left_pos
        right_delta = right_pos - self.last_right_pos
        
        d_left = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        d_right = right_delta * WHEEL_CIRCUMFERENCE_MM / 10
        return d_left, d_right

    def _predict(self, d_left, d_right):
        """EKF Prediction Step: x = f(x, u)"""
        ds = (d_left + d_right) / 2.0
        
        # Encoder polarity: (Right - Left)
        # If right wheel moves more, we turn LEFT (CCW).
        # Standard: CCW is Positive Theta.
        d_theta = (d_right - d_left) / WHEEL_BASE_CM
        
        # Use half-angle for better integration accuracy (Runge-Kutta 2nd order approx)
        avg_theta = self.theta + d_theta / 2.0
        
        # Motion Model (Standard X-Forward System):
        # Theta=0 -> +X (East)
        # Theta=90 -> +Y (North/Left)
        # x += ds * cos(theta)
        # y += ds * sin(theta)
        self.x += ds * np.cos(avg_theta)
        self.y += ds * np.sin(avg_theta)
        self.theta += d_theta
        
        # Normalize theta
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # DEBUG LOG
        logger.debug(f"Odom: x={self.x:.1f}, y={self.y:.1f}, th={self.theta:.2f}, ds={ds:.1f}")

        # Jacobian F (df/dx, df/dy, df/dtheta)
        F = np.eye(3)
        # df/dtheta derivatives
        F[0, 2] = -ds * np.sin(avg_theta)
        F[1, 2] =  ds * np.cos(avg_theta)
        
        # Covariance Prediction: P = F*P*F.T + Q
        self.P = F @ self.P @ F.T + self.Q

    def _correct(self, imu_z):
        """EKF Correction Step: Incorporate measurement z"""
        # Measurement Vector z = [imu_heading]
        # Measurement Matrix H = [0, 0, 1] (We measure theta directly)
        H = np.array([[0, 0, 1]])
        
        theta_before = self.theta
        
        # Innovation (Residual) y = z - Hx
        # We need to handle angle wrapping for the residual
        y = imu_z - self.theta
        y = np.arctan2(np.sin(y), np.cos(y))  # Normalize -pi to pi
        
        # Innovation Covariance S = H*P*H.T + R
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain K = P*H.T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State x = x + K*y
        change = K @ np.array([[y]]) # Shape (3,1)
        
        self.x += change[0, 0]
        self.y += change[1, 0]
        self.theta += change[2, 0]
        
        # Normalize Theta again
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # DEBUG: EKF Correction logging
        logger.debug(f"EKF - Heading before: {np.degrees(theta_before):.1f}°, IMU: {np.degrees(imu_z):.1f}°, after: {np.degrees(self.theta):.1f}°")
        
        # Update Covariance P = (I - K*H) * P
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
