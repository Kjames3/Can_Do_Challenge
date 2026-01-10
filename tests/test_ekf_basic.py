
import unittest
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_state import RobotState, WHEEL_CIRCUMFERENCE_MM

class TestEKF(unittest.TestCase):
    def test_init(self):
        rs = RobotState()
        self.assertEqual(rs.x, 0.0)
        self.assertEqual(rs.theta, 0.0)
        
    def test_pure_odometry(self):
        """Test standard update without IMU"""
        rs = RobotState()
        rs.update(0, 0) # Init
        
        # Move forward 10 revs
        # 10 revs * 11.33 cm/rev ≈ 113.3 cm
        rs.update(10, 10)
        
        self.assertAlmostEqual(rs.x, 0.0, places=1) # No lateral movement
        self.assertGreater(rs.y, 100.0) # Moved forward Y
        self.assertAlmostEqual(rs.theta, 0.0, places=1)
        
        print(f"Odom only: x={rs.x:.2f}, y={rs.y:.2f}, th={rs.theta:.2f}")

    def test_ekf_correction(self):
        """Test IMU update correcting drift"""
        rs = RobotState()
        rs.update_with_imu(0, 0, 0) # Init
        
        # Simulate moving forward but drifting in theta according to encoders
        # Left=10, Right=9.8 -> Slight left turn? (L > R -> Right turn? No, R > L -> Left turn)
        # Actually in differential drive:
        # If Left > Right, wheel on left moves more -> Robot turns Right (CW).
        # Let's check logic: angular = (R - L) / W.
        # If R=10, L=10 -> 0.
        # If L=10, R=9 -> (9-10)/W = -1/W (Negative).
        # Negative angular velocity. 
        # If +Theta is Left (CCW), then Negative is Right (CW). Correct.
        
        # So L=10, R=9.8 implies turning Right (Theta decreases).
        
        # But we feed IMU heading = 0.0 (Robot actually went straight).
        # EKF should trust IMU and keep Theta near 0.
        
        rs.update_with_imu(10, 9.8, 0.0)
        
        print(f"EKF Result: theta={rs.theta:.4f} (Should be close to 0.0)")
        self.assertAlmostEqual(rs.theta, 0.0, delta=0.1)
        
        # Verify Covariance decreased (confidence increased? or stabilized)
        print(f"Covariance P:\n{rs.P}")

if __name__ == '__main__':
    unittest.main()
