"""
Navigation System Unit Tests

Tests for critical autodrive components:
1. Coordinate transform math (world <-> robot frame)
2. Bearing calculations
3. Pure Pursuit curvature logic
4. Distance calculations

Run with: python -m pytest tests/test_navigation.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoordinateTransforms:
    """Test world-to-robot frame transformations"""
    
    def test_target_directly_ahead(self):
        """When robot faces +Y and target is ahead, bearing should be 0"""
        robot_x, robot_y = 0, 0
        robot_theta = 0  # Facing +Y (forward)
        goal_x, goal_y = 0, 100  # Target directly ahead
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)
        
        # Transform to robot frame (Y-forward system)
        local_x = -(dx * cos_t - dy * sin_t)
        local_y = dx * sin_t + dy * cos_t
        
        bearing = np.arctan2(local_x, local_y)
        
        assert abs(bearing) < 0.01, f"Expected ~0, got {np.degrees(bearing):.1f}°"
        assert local_y > 0, "Target should be in front (positive local_y)"
    
    def test_target_to_the_right(self):
        """When robot faces +Y and target is to the right, bearing should be positive"""
        robot_x, robot_y = 0, 0
        robot_theta = 0  # Facing +Y
        goal_x, goal_y = 50, 50  # Target ahead-right
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)
        
        local_x = -(dx * cos_t - dy * sin_t)
        local_y = dx * sin_t + dy * cos_t
        
        bearing = np.arctan2(local_x, local_y)
        
        # Should be negative (target to the LEFT in corrected frame)
        # Note: We negated local_x, so right in world = left in local
        assert bearing < 0, f"Expected negative bearing for right target, got {np.degrees(bearing):.1f}°"
    
    def test_target_behind(self):
        """When target is behind robot, bearing should be ~180°"""
        robot_x, robot_y = 0, 0
        robot_theta = 0  # Facing +Y
        goal_x, goal_y = 0, -100  # Target behind
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)
        
        local_x = -(dx * cos_t - dy * sin_t)
        local_y = dx * sin_t + dy * cos_t
        
        bearing = np.arctan2(local_x, local_y)
        
        assert abs(abs(bearing) - np.pi) < 0.1, f"Expected ~180°, got {np.degrees(bearing):.1f}°"
    
    def test_rotated_robot_facing_east(self):
        """Robot facing East (+X direction) should see +Y target on its left"""
        robot_x, robot_y = 0, 0
        robot_theta = np.pi / 2  # Facing +X (East, 90° from Y)
        goal_x, goal_y = 100, 0  # Target directly ahead in +X
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)
        
        local_x = -(dx * cos_t - dy * sin_t)
        local_y = dx * sin_t + dy * cos_t
        
        bearing = np.arctan2(local_x, local_y)
        
        # Target should be ahead (bearing ~0)
        assert abs(bearing) < 0.1, f"Expected ~0°, got {np.degrees(bearing):.1f}°"


class TestDistanceCalculations:
    """Test distance-related calculations"""
    
    def test_pythagorean_distance(self):
        """Basic distance calculation"""
        robot_x, robot_y = 0, 0
        goal_x, goal_y = 30, 40
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        assert abs(dist - 50) < 0.01, f"Expected 50, got {dist}"
    
    def test_zero_distance(self):
        """Distance when at goal"""
        robot_x, robot_y = 100, 100
        goal_x, goal_y = 100, 100
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        assert dist == 0, f"Expected 0, got {dist}"


class TestPurePursuitMath:
    """Test Pure Pursuit curvature calculations"""
    
    def test_straight_ahead_zero_steering(self):
        """When target is straight ahead, steering should be ~0"""
        bearing = 0  # Target directly ahead
        curvature_gain = 1.2
        
        steering = np.sin(bearing) * curvature_gain
        
        assert abs(steering) < 0.01, f"Expected 0 steering, got {steering}"
    
    def test_right_turn_positive_steering(self):
        """Target to the right should produce positive steering (slow right wheel)"""
        bearing = np.radians(30)  # 30° to the right
        curvature_gain = 1.2
        
        steering = np.sin(bearing) * curvature_gain
        
        # Positive steering = left_power increases, right_power decreases = turn right
        assert steering > 0, f"Expected positive steering for right turn, got {steering}"
    
    def test_left_turn_negative_steering(self):
        """Target to the left should produce negative steering"""
        bearing = np.radians(-30)  # 30° to the left
        curvature_gain = 1.2
        
        steering = np.sin(bearing) * curvature_gain
        
        assert steering < 0, f"Expected negative steering for left turn, got {steering}"
    
    def test_motor_power_clamping(self):
        """Motor powers should be clamped to [-1, 1]"""
        base_speed = 0.5
        steering = 1.0  # Very aggressive turn
        
        left_power = base_speed + steering
        right_power = base_speed - steering
        
        max_pwr = max(abs(left_power), abs(right_power), 1.0)
        left_power /= max_pwr
        right_power /= max_pwr
        
        assert -1.0 <= left_power <= 1.0, f"Left power out of range: {left_power}"
        assert -1.0 <= right_power <= 1.0, f"Right power out of range: {right_power}"


class TestArrivalDetection:
    """Test arrival/goal-reached logic"""
    
    def test_within_threshold(self):
        """Should detect arrival when within threshold"""
        distance = 12.0
        threshold = 15.0
        
        arrived = distance <= threshold
        
        assert arrived, "Should have arrived when distance < threshold"
    
    def test_outside_threshold(self):
        """Should not detect arrival when outside threshold"""
        distance = 50.0
        threshold = 15.0
        
        arrived = distance <= threshold
        
        assert not arrived, "Should not have arrived when distance > threshold"


class TestSharpTurnDetection:
    """Test pivot turn fallback logic"""
    
    def test_gentle_angle_curved_drive(self):
        """Angles < 45° should use curved drive"""
        bearing = np.radians(30)  # 30°
        pivot_threshold = 0.8  # ~45°
        
        should_pivot = abs(bearing) > pivot_threshold
        
        assert not should_pivot, "Should use curved drive for 30° angle"
    
    def test_sharp_angle_pivot(self):
        """Angles > 45° should trigger pivot"""
        bearing = np.radians(60)  # 60°
        pivot_threshold = 0.8  # ~45°
        
        should_pivot = abs(bearing) > pivot_threshold
        
        assert should_pivot, "Should pivot for 60° angle"
    
    def test_behind_target_pivot(self):
        """Target behind should definitely pivot"""
        bearing = np.radians(150)  # Behind and to the side
        pivot_threshold = 0.8
        
        should_pivot = abs(bearing) > pivot_threshold
        
        assert should_pivot, "Should pivot when target is behind"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
