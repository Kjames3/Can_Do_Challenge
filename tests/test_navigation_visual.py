"""
Graphical Navigation Simulator

Visual test that simulates the Pure Pursuit navigation without hardware.
Shows:
- Robot as a square with direction arrow
- Target as a circle
- Projected trajectory arc
- Simulated movement over time

Run with: python tests/test_navigation_visual.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time


class SimulatedRobot:
    """Simulated robot state for testing navigation"""
    
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        
        # Pure Pursuit settings
        self.drive_speed = 0.5
        self.curvature_gain = 1.2
        self.min_drive_speed = 0.25
        self.arrival_threshold = 15.0  # cm
        self.pivot_threshold = 0.8  # radians (~45°)
        
    def calculate_bearing_to_goal(self, goal_x, goal_y):
        """Calculate bearing to goal in robot frame"""
        dx = goal_x - self.x
        dy = goal_y - self.y
        
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        
        # Transform to robot frame (Y-forward, X-right)
        local_x = -(dx * cos_t - dy * sin_t)
        local_y = dx * sin_t + dy * cos_t
        
        bearing = np.arctan2(local_x, local_y)
        distance = np.sqrt(dx*dx + dy*dy)
        
        return bearing, distance
    
    def calculate_motor_powers(self, bearing, distance):
        """Calculate Pure Pursuit motor powers"""
        # Check if we need to pivot
        if abs(bearing) > self.pivot_threshold:
            # Pivot turn
            turn_power = 0.3 * np.sign(bearing)
            return -turn_power, turn_power
        
        # Curvature steering
        steering = np.sin(bearing) * self.curvature_gain
        
        # Base speed (slow down when close)
        base_speed = self.drive_speed
        if distance < 30.0:
            base_speed = max(self.min_drive_speed, base_speed * (distance / 30.0))
        
        left_power = base_speed + steering
        right_power = base_speed - steering
        
        # Clamp
        max_pwr = max(abs(left_power), abs(right_power), 1.0)
        left_power /= max_pwr
        right_power /= max_pwr
        
        return left_power, right_power
    
    def update(self, left_power, right_power, dt=0.1):
        """Update robot position based on motor powers (differential drive)"""
        # Convert motor powers to velocities
        wheel_base = 20.0  # cm
        max_velocity = 30.0  # cm/s
        
        v_left = left_power * max_velocity
        v_right = right_power * max_velocity
        
        # Differential drive kinematics
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / wheel_base
        
        # Update pose
        self.x += v * np.sin(self.theta) * dt
        self.y += v * np.cos(self.theta) * dt
        self.theta += omega * dt


def calculate_trajectory_arc(robot, goal_x, goal_y, num_points=20):
    """Calculate Bezier trajectory arc for visualization"""
    rx, ry = robot.x, robot.y
    gx, gy = goal_x, goal_y
    
    dx = gx - rx
    dy = gy - ry
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist < 5:
        return [], []
    
    # Calculate bearing for arc curvature
    cos_t = np.cos(robot.theta)
    sin_t = np.sin(robot.theta)
    local_x = -(dx * cos_t - dy * sin_t)
    local_y = dx * sin_t + dy * cos_t
    bearing = np.arctan2(local_x, local_y)
    
    # Generate Bezier curve
    ctrl_offset = dist * 0.3 * np.sin(bearing)
    mid_x = (rx + gx) / 2 + ctrl_offset * cos_t
    mid_y = (ry + gy) / 2 + ctrl_offset * sin_t
    
    traj_x = []
    traj_y = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        px = (1-t)**2 * rx + 2*(1-t)*t * mid_x + t**2 * gx
        py = (1-t)**2 * ry + 2*(1-t)*t * mid_y + t**2 * gy
        traj_x.append(px)
        traj_y.append(py)
    
    return traj_x, traj_y


def run_simulation():
    """Run the graphical navigation simulation"""
    
    # Initialize
    robot = SimulatedRobot(x=0, y=0, theta=0)
    goal_x, goal_y = 80, 100  # Target position
    
    # History for trail
    trail_x = [robot.x]
    trail_y = [robot.y]
    
    # Setup plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 150)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Pure Pursuit Navigation Simulation')
    
    # Create plot elements
    robot_patch = patches.Rectangle((-5, -5), 10, 10, angle=0, 
                                     facecolor='blue', edgecolor='black', linewidth=2)
    ax.add_patch(robot_patch)
    
    # Arrow showing robot direction
    arrow = ax.arrow(0, 0, 0, 10, head_width=3, head_length=2, fc='yellow', ec='black')
    
    # Target marker
    target = plt.Circle((goal_x, goal_y), 5, color='red', label='Target')
    ax.add_patch(target)
    
    # Trail line
    trail_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2, label='Trail')
    
    # Trajectory arc
    traj_line, = ax.plot([], [], 'g--', linewidth=2, label='Projected Path')
    
    # Legend
    ax.legend(loc='upper left')
    
    # Status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    arrived = False
    frame_count = 0
    
    def animate(frame):
        nonlocal arrived, frame_count
        frame_count += 1
        
        if arrived:
            return robot_patch, trail_line, traj_line, status_text
        
        # Calculate bearing and distance to goal
        bearing, distance = robot.calculate_bearing_to_goal(goal_x, goal_y)
        
        # Check arrival
        if distance <= robot.arrival_threshold:
            arrived = True
            status_text.set_text(f'✓ ARRIVED!\nDistance: {distance:.1f}cm\nFrames: {frame_count}')
            return robot_patch, trail_line, traj_line, status_text
        
        # Calculate motor powers
        left_power, right_power = robot.calculate_motor_powers(bearing, distance)
        
        # Update robot position
        robot.update(left_power, right_power, dt=0.1)
        
        # Update trail
        trail_x.append(robot.x)
        trail_y.append(robot.y)
        trail_line.set_data(trail_x, trail_y)
        
        # Update trajectory arc
        traj_x, traj_y = calculate_trajectory_arc(robot, goal_x, goal_y)
        traj_line.set_data(traj_x, traj_y)
        
        # Update robot visualization
        robot_patch.set_xy((robot.x - 5, robot.y - 5))
        robot_patch.angle = np.degrees(robot.theta)
        
        # Update arrow (direction indicator)
        ax.patches = [p for p in ax.patches if p != arrow]
        arrow_len = 12
        arrow_dx = arrow_len * np.sin(robot.theta)
        arrow_dy = arrow_len * np.cos(robot.theta)
        new_arrow = ax.arrow(robot.x, robot.y, arrow_dx, arrow_dy, 
                            head_width=4, head_length=3, fc='yellow', ec='black')
        
        # Update status
        mode = "PIVOT" if abs(bearing) > robot.pivot_threshold else "CURVED"
        status_text.set_text(
            f'Robot: ({robot.x:.1f}, {robot.y:.1f})\n'
            f'Heading: {np.degrees(robot.theta):.1f}°\n'
            f'Bearing: {np.degrees(bearing):.1f}°\n'
            f'Distance: {distance:.1f}cm\n'
            f'Mode: {mode}\n'
            f'Motors: L={left_power:.2f} R={right_power:.2f}'
        )
        
        return robot_patch, trail_line, traj_line, status_text
    
    # Run animation
    anim = FuncAnimation(fig, animate, frames=500, interval=50, blit=False)
    plt.show()


def run_static_test():
    """Run a static test showing the path without animation"""
    
    print("=" * 60)
    print("PURE PURSUIT NAVIGATION - STATIC PATH TEST")
    print("=" * 60)
    
    # Initialize
    robot = SimulatedRobot(x=0, y=0, theta=0)
    goal_x, goal_y = 80, 100
    
    # Simulate path
    trail_x = [robot.x]
    trail_y = [robot.y]
    
    max_steps = 500
    step = 0
    
    while step < max_steps:
        bearing, distance = robot.calculate_bearing_to_goal(goal_x, goal_y)
        
        if distance <= robot.arrival_threshold:
            print(f"\n✓ ARRIVED in {step} steps!")
            print(f"  Final position: ({robot.x:.1f}, {robot.y:.1f})")
            print(f"  Final distance: {distance:.1f}cm")
            break
        
        left_power, right_power = robot.calculate_motor_powers(bearing, distance)
        robot.update(left_power, right_power, dt=0.1)
        
        trail_x.append(robot.x)
        trail_y.append(robot.y)
        step += 1
        
        if step % 50 == 0:
            mode = "PIVOT" if abs(bearing) > robot.pivot_threshold else "CURVED"
            print(f"  Step {step}: pos=({robot.x:.1f}, {robot.y:.1f}), "
                  f"dist={distance:.1f}cm, mode={mode}")
    
    # Plot final result
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 140)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Pure Pursuit Navigation - Path Result')
    
    # Plot trail
    ax.plot(trail_x, trail_y, 'b-', linewidth=2, label='Robot Path')
    
    # Start marker
    ax.plot(0, 0, 'go', markersize=15, label='Start', zorder=5)
    
    # Target marker
    target = plt.Circle((goal_x, goal_y), 5, color='red', label='Target', zorder=5)
    ax.add_patch(target)
    
    # Final position
    ax.plot(robot.x, robot.y, 'bs', markersize=12, label='Final Position', zorder=5)
    
    # Draw robot direction arrow at final position
    arrow_len = 10
    ax.arrow(robot.x, robot.y, 
             arrow_len * np.sin(robot.theta), 
             arrow_len * np.cos(robot.theta),
             head_width=4, head_length=3, fc='blue', ec='black', zorder=6)
    
    ax.legend(loc='upper left')
    
    # Add path statistics
    total_distance = sum(np.sqrt((trail_x[i+1]-trail_x[i])**2 + (trail_y[i+1]-trail_y[i])**2) 
                        for i in range(len(trail_x)-1))
    direct_distance = np.sqrt(goal_x**2 + goal_y**2)
    efficiency = direct_distance / total_distance * 100 if total_distance > 0 else 0
    
    stats_text = (f'Path Length: {total_distance:.1f}cm\n'
                  f'Direct Distance: {direct_distance:.1f}cm\n'
                  f'Efficiency: {efficiency:.1f}%')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.show()
    
    return step, total_distance, efficiency


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--static":
        run_static_test()
    else:
        print("Running animated simulation...")
        print("(Use --static for non-animated test)")
        run_simulation()
