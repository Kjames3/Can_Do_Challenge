import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_state import RobotState

class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.robot = RobotState()
        # Reset state
        self.robot.x = 0.0
        self.robot.y = 0.0
        self.robot.theta = 0.0
        self.robot.initialized = True
        self.robot.last_left_pos = 0.0
        self.robot.last_right_pos = 0.0
        
        # Standardize for testing
        # Circumference = 113.3 mm -> 11.33 cm
        # Wheel Base = 35.6 cm
        self.WHEEL_CIRC_CM = 11.33
        self.WHEEL_BASE_CM = 35.6
        
        # Monkey patch constants if needed, but better to test with actual values
        
    def test_drive_straight_forward(self):
        """Moving both wheels equal amount should increase X (Forward)"""
        # Drive 10 revolutions
        dist = 10 * self.WHEEL_CIRC_CM 
        
        # RobotState uses positions, not deltas directly in update
        # 1000 ticks per revolution (implied by code comments) -> Let's assume input is raw ticks or normalized?
        # looking at code: d_left = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        # If input is "position", let's assume it's normalized "rotations" or similar?
        # Actually checking robot_state.py: 
        # d_left = left_delta * WHEEL_CIRCUMFERENCE_MM / 10
        # Wait, left_pos usually comes from encoder.get_position() which usually returns rotations?
        # Let's assume input is "1.0" = 1 unit which maps to CIRCUMFERENCE.
        
        # If get_position returns ROTATIONS:
        # Input 1.0 -> 11.33 cm
        
        self.robot.update(10.0, 10.0)
        
        print(f"Straight: X={self.robot.x:.2f}, Y={self.robot.y:.2f}, Th={np.degrees(self.robot.theta):.2f}")
        
        # Expectation: X increases, Y stays 0, Theta 0
        self.assertAlmostEqual(self.robot.y, 0.0, places=1)
        self.assertAlmostEqual(self.robot.theta, 0.0, places=1)
        self.assertGreater(self.robot.x, 100.0) # Should be ~113.3 cm

    def test_turn_left(self):
        """Right wheel moving > Left wheel should increase positive Theta (Left Turn)"""
        # Pivot Left: Right moves 1.0, Left moves 0
        self.robot.update(0.0, 2.0)
        
        print(f"Turn Left: X={self.robot.x:.2f}, Y={self.robot.y:.2f}, Th={np.degrees(self.robot.theta):.2f}")
        
        # Expectation: Positive Theta
        self.assertGreater(self.robot.theta, 0.0)
        self.assertTrue(0 < self.robot.theta < np.pi)

    def test_turn_right(self):
        """Left wheel moving > Right wheel should decrease Theta (Right Turn)"""
        # Pivot Right: Left moves 2.0, Right moves 0
        self.robot.update(2.0, 0.0)
        
        print(f"Turn Right: X={self.robot.x:.2f}, Y={self.robot.y:.2f}, Th={np.degrees(self.robot.theta):.2f}")
        
        # Expectation: Negative Theta
        self.assertLess(self.robot.theta, 0.0)
    
    def test_drive_forward_at_90_degrees(self):
        """If facing 90 deg (Left), driving straight should increase Y"""
        self.robot.theta = np.pi / 2 # Face Left (+Y)
        
        self.robot.update(10.0, 10.0)
        
        print(f"Drive @ 90: X={self.robot.x:.2f}, Y={self.robot.y:.2f}, Th={np.degrees(self.robot.theta):.2f}")
        
        # Expectation: X stays 0, Y increases
        self.assertAlmostEqual(self.robot.x, 0.0, places=1)
        self.assertGreater(self.robot.y, 100.0)

if __name__ == '__main__':
    unittest.main()
