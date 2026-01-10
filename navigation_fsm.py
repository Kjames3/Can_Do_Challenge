"""
Navigation System using Behavior Trees (BT)
Refactored from Finite State Machine (FSM) for modularity and extensibility.
"""

import asyncio
import time
import numpy as np
from enum import Enum, auto

# =============================================================================
# BEHAVIOR TREE FRAMEWORK
# =============================================================================

class NodeStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()

class Context:
    """Shared blackboard for BT nodes"""
    def __init__(self):
        self.left_motor = None
        self.right_motor = None
        self.imu = None
        self.camera = None
        self.config = None
        
        # Inputs (Per Frame)
        self.detection = None
        self.target_pose = None
        self.lidar_min_distance = None
        self.current_pose = {'x':0, 'y':0, 'theta':0}
        
        # Persistent State
        self.goal_x = None
        self.goal_y = None
        self.goal_distance = 0.0
        self.start_pose = {'x':0, 'y':0, 'theta':0}
        self.nav_state = "IDLE"  # For external reporting
        self.approach_phase = "SEARCH"
        
        # Internal Logic State
        self.acquire_samples = []
        self.target_imu_rotation = 0.0
        self.avoid_start_time = 0.0
        self.return_phase = "WAITING"
        self.return_start_time = 0.0
        self.backup_start_pos = None
        self.target_lost_time = 0.0
        self.goal_frozen = False
        self.last_focus_val = -1.0
        
        # Output callbacks
        self.on_arrived = None
        self.on_returned = None

class Node:
    """Base Behavior Tree Node"""
    async def tick(self, ctx: Context) -> NodeStatus:
        return NodeStatus.FAILURE

class Composite(Node):
    def __init__(self, children):
        self.children = children

class Selector(Composite):
    """Or / Fallback: Runs children until one SUCCEEDS or RUNNING"""
    async def tick(self, ctx: Context) -> NodeStatus:
        for child in self.children:
            status = await child.tick(ctx)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE

class Sequence(Composite):
    """And: Runs children until one FAILS or RUNNING"""
    async def tick(self, ctx: Context) -> NodeStatus:
        for child in self.children:
            status = await child.tick(ctx)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

# =============================================================================
# LEAF NODES (ACTIONS & CONDITIONS)
# =============================================================================

class CheckObstacle(Node):
    """Returns SUCCESS if obstacle detected within threshold"""
    async def tick(self, ctx: Context) -> NodeStatus:
        if ctx.lidar_min_distance and ctx.lidar_min_distance < ctx.config.obstacle_min_distance_cm:
            # Don't trigger avoidance if we are in RETURN mode (simplified logic)
            if ctx.nav_state != "RETURNING":
                ctx.nav_state = "AVOIDING"
                return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class AvoidObstacle(Node):
    """Back up for a set duration"""
    async def tick(self, ctx: Context) -> NodeStatus:
        if ctx.avoid_start_time == 0:
            ctx.avoid_start_time = time.time()
            print(f"⚠️ Obstacle! Backing up...")
        
        elapsed = time.time() - ctx.avoid_start_time
        if elapsed < ctx.config.backup_duration_sec:
            await _set_motor_power(ctx, -ctx.config.backup_speed, -ctx.config.backup_speed)
            return NodeStatus.RUNNING
        else:
            await _stop_motors(ctx)
            ctx.avoid_start_time = 0  # Reset
            ctx.nav_state = "SEARCHING" # Default back to search
            return NodeStatus.SUCCESS

class CheckReturnTrigger(Node):
    """Checks if we should be returning home"""
    async def tick(self, ctx: Context) -> NodeStatus:
        if ctx.nav_state == "RETURNING":
            return NodeStatus.SUCCESS
        if ctx.nav_state == "ARRIVED" and ctx.config.auto_return:
            ctx.nav_state = "RETURNING"
            ctx.return_start_time = time.time()
            ctx.return_phase = "WAITING"
            print("↩ Auto-return triggered")
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class ReturnHomeSequence(Node):
    """Complex sequence for returning home: Wait -> Backup -> Navigate -> Align"""
    async def tick(self, ctx: Context) -> NodeStatus:
        # Phase 1: WAITING
        if ctx.return_phase == "WAITING":
            elapsed = time.time() - ctx.return_start_time
            if elapsed < 5.0:
                await _stop_motors(ctx)
                if int(elapsed * 10) % 10 == 0: print(f"  ⏳ Returning in {5 - int(elapsed)}s...", end='\r')
                return NodeStatus.RUNNING
            ctx.return_phase = "BACKING"
            ctx.backup_start_pos = (ctx.current_pose['x'], ctx.current_pose['y'])
            ctx.avoid_start_time = time.time()
            print("\n  ◀ Starting Return Backup")
            return NodeStatus.RUNNING

        # Phase 2: BACKING
        if ctx.return_phase == "BACKING":
            dx = ctx.current_pose['x'] - ctx.backup_start_pos[0]
            dy = ctx.current_pose['y'] - ctx.backup_start_pos[1]
            dist_moved = np.sqrt(dx*dx + dy*dy)
            
            if dist_moved < 20.0 and (time.time() - ctx.avoid_start_time < 5.0):
                await _set_motor_power(ctx, -0.35, -0.35)
                return NodeStatus.RUNNING
            
            ctx.return_phase = "NAVIGATING"
            await _stop_motors(ctx)
            return NodeStatus.RUNNING

        # Phase 3: NAVIGATING (Pure Pursuit to Start)
        if ctx.return_phase == "NAVIGATING":
            # Target is the start pose (0,0 typically)
            tx, ty = ctx.start_pose['x'], ctx.start_pose['y']
            dx = tx - ctx.current_pose['x']
            dy = ty - ctx.current_pose['y']
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < ctx.config.return_distance_threshold:
                print(f"✓ Returned to start ({dist:.1f}cm). Aligning...")
                ctx.return_phase = "ALIGNING"
                await _stop_motors(ctx)
                return NodeStatus.RUNNING
            
            # Helper for Pure Pursuit
            await _execute_pure_pursuit(ctx, tx, ty, dist)
            return NodeStatus.RUNNING
            
        # Phase 4: ALIGNING
        if ctx.return_phase == "ALIGNING":
            # Align 180 deg from original goal
            if ctx.goal_x is not None:
                gx = ctx.goal_x - ctx.start_pose['x']
                gy = ctx.goal_y - ctx.start_pose['y']
                target_heading = np.arctan2(-gx, gy) + np.pi
            else:
                target_heading = np.pi
            
            # Normalize
            target_heading = np.arctan2(np.sin(target_heading), np.cos(target_heading))
            
            current_h = ctx.current_pose['theta']
            diff = target_heading - current_h
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            
            if abs(diff) < 0.1: # ~5 degrees
                await _stop_motors(ctx)
                ctx.nav_state = "IDLE"
                if ctx.on_returned: ctx.on_returned()
                return NodeStatus.SUCCESS
            
            # Original Logic:
            # remaining_turn = target + heading.
            # Wait, let's use a simpler Pivot helper.
            await _pivot_towards(ctx, diff)
            return NodeStatus.RUNNING
            
        return NodeStatus.FAILURE

class CheckTargetKnown(Node):
    """Success if we have a goal or see a target"""
    async def tick(self, ctx: Context) -> NodeStatus:
        if ctx.detection and ctx.detection.get('distance_cm'):
            return NodeStatus.SUCCESS
        if ctx.goal_x is not None:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class ApproachTargetSequence(Node):
    """
    Handles Acquisition -> Curved Drive -> Arrival
    This logic mimics the robust 'APPROACHING' state from the FSM.
    """
    async def tick(self, ctx: Context) -> NodeStatus:
        ctx.nav_state = "APPROACHING"
        
        # 1. Update Goal from Vision (if available and far enough)
        if ctx.detection and ctx.detection.get('distance_cm'):
            dist = ctx.detection['distance_cm']
            # Dynamic Focus
            if ctx.camera:
                 focus = 0.0 if dist > 100 else max(0.0, min(14.0, 70.0 / dist))
                 if abs(focus - ctx.last_focus_val) > 0.2:
                     try: ctx.camera.set_focus(focus)
                     except: pass
                     ctx.last_focus_val = focus

            if ctx.target_pose and ctx.target_pose.get('x') is not None:
                # Goal Freezing Logic
                curr_dist_to_goal = 999
                if ctx.goal_x:
                    dx = ctx.goal_x - ctx.current_pose['x']
                    dy = ctx.goal_y - ctx.current_pose['y']
                    curr_dist_to_goal = np.sqrt(dx*dx + dy*dy)
                
                if not ctx.goal_frozen or curr_dist_to_goal > 22.0:
                    ctx.goal_x = ctx.target_pose['x']
                    ctx.goal_y = ctx.target_pose['y']
                    if curr_dist_to_goal < 22.0:
                        ctx.goal_frozen = True
                        print(f"  🔒 Goal frozen at {curr_dist_to_goal:.1f}cm")

        # 2. Acquire Phase (if no goal yet)
        if ctx.goal_x is None:
            ctx.approach_phase = "ACQUIRE"
            if ctx.detection and ctx.detection.get('distance_cm'):
                # Simple average samples logic could go here, 
                # but for now let's just rely on the first good target pose
                await _stop_motors(ctx)
                return NodeStatus.RUNNING
            else:
                return NodeStatus.FAILURE # Lost target before acquiring goal

        # 3. Drive Phase (Map Based)
        ctx.approach_phase = "DRIVE"
        dx = ctx.goal_x - ctx.current_pose['x']
        dy = ctx.goal_y - ctx.current_pose['y']
        map_dist = np.sqrt(dx*dx + dy*dy)
        ctx.goal_distance = map_dist
        
        if map_dist <= 10.0:
            print(f"✓ TARGET REACHED! Dist: {map_dist:.1f}cm")
            ctx.nav_state = "ARRIVED"
            await _stop_motors(ctx)
            if ctx.on_arrived: ctx.on_arrived()
            return NodeStatus.SUCCESS
            
        await _execute_pure_pursuit(ctx, ctx.goal_x, ctx.goal_y, map_dist)
        return NodeStatus.RUNNING

class SpinSearch(Node):
    """Spin slowly to look for target"""
    async def tick(self, ctx: Context) -> NodeStatus:
        ctx.nav_state = "SEARCHING"
        ctx.goal_frozen = False
        await _set_motor_power(ctx, ctx.config.search_speed, -ctx.config.search_speed)
        return NodeStatus.RUNNING

class Idle(Node):
    async def tick(self, ctx: Context) -> NodeStatus:
        await _stop_motors(ctx)
        return NodeStatus.SUCCESS

# =============================================================================
# HELPERS
# =============================================================================

async def _set_motor_power(ctx, left, right):
    if ctx.left_motor and ctx.right_motor:
        # Simple deadband can be added here
        try:
            await ctx.left_motor.set_power(left)
            await ctx.right_motor.set_power(right)
        except Exception:
            pass

async def _stop_motors(ctx):
    await _set_motor_power(ctx, 0, 0)

async def _pivot_towards(ctx, diff_angle):
    # diff_angle: Positive = Left, Negative = Right
    # diff_angle: Positive = Left, Negative = Right
    # Increased minimum power from 0.32 to 0.45 to prevent stalling
    speed = max(0.45, min(0.6, abs(diff_angle) * 1.8))
    if diff_angle > 0:
        # Turn Left (CCW): Right motor forward, Left stopped/back
        await _set_motor_power(ctx, 0, speed) 
    else:
        # Turn Right (CW): Left motor forward
        await _set_motor_power(ctx, speed, 0)

async def _execute_pure_pursuit(ctx, tx, ty, distance):
    # Coordinate Transform to Robot Frame
    dx = tx - ctx.current_pose['x']
    dy = ty - ctx.current_pose['y']
    heading = ctx.current_pose['theta']
    
    # Transform World Vector (dx, dy) into Robot Frame (local_x, local_y)
    # Robot Frame: X=Right, Y=Forward.
    # Heading theta is angle from Y-axis to Robot-Forward?
    # Based on robot_state.py: x+=sin(th), y+=cos(th). 
    # This implies 0 is North (+Y), 90 is East (+X).
    # 
    # To rotate vector D by -theta:
    # x_rob = dx * cos(theta) + dy * sin(theta)  <-- This assumes standard X-X alignment?
    #
    # Let's stick to the code we backed up which was working:
    # local_x = -(dx * cos_t - dy * sin_t)
    # local_y = dx * sin_t + dy * cos_t
    
    local_x = -(dx * np.cos(heading) - dy * np.sin(heading))
    local_y = dx * np.sin(heading) + dy * np.cos(heading)
    bearing = np.arctan2(local_x, local_y)
    
    # Pure Pursuit steering
    curvature = -np.sin(bearing) * ctx.config.curvature_gain * 0.5
    
    # Speeds
    base = ctx.config.drive_speed
    base = ctx.config.drive_speed
    if distance < 30.0:
        base = max(0.40, base * (distance/30.0))
        
    l_pow = base + curvature
    r_pow = base - curvature
    
    max_p = max(abs(l_pow), abs(r_pow), 1.0)
    await _set_motor_power(ctx, l_pow/max_p, r_pow/max_p)


# =============================================================================
# MAIN WRAPPER CLASS
# =============================================================================

class NavigationConfig:
    """Config dataclass"""
    target_distance_cm = 5.0
    dist_threshold_cm = 2.0
    obstacle_min_distance_cm = 20.0
    obstacle_min_distance_cm = 20.0
    backup_duration_sec = 0.8
    search_speed = 0.35  # Increased from 0.20
    drive_speed = 0.60   # Increased from 0.50
    backup_speed = 0.40  # Increased from 0.25
    auto_return = True
    return_distance_threshold = 15.0
    curvature_gain = 1.2

class NavigationFSM:
    """
    Behavior Tree Wrapper.
    Maintains compatibility with external calls (update, start, stop).
    """
    def __init__(self, left_motor, right_motor, camera=None, imu=None, config=None):
        self.ctx = Context()
        self.ctx.left_motor = left_motor
        self.ctx.right_motor = right_motor
        self.ctx.camera = camera
        self.ctx.imu = imu
        self.ctx.config = config or NavigationConfig()
        
        # Build Tree
        self.tree = self._build_tree()
        self.active = False
        
    def _build_tree(self):
        return Selector([
            # 1. High Priority: Obstacle Avoidance
            Sequence([CheckObstacle(), AvoidObstacle()]),
            
            # 2. Return Home Logic (Overrides search if triggered)
            Sequence([CheckReturnTrigger(), ReturnHomeSequence()]),
            
            # 3. Vision/Goal Navigation
            Sequence([CheckTargetKnown(), ApproachTargetSequence()]),
            
            # 4. Default: Search
            SpinSearch()
        ])
        
    @property
    def state(self):
        # Compatibility property
        return self.ctx.nav_state
    
    @property
    def state_summary(self):
        if self.ctx.nav_state == "APPROACHING":
            return f"APPROACHING/{self.ctx.approach_phase}"
        return self.ctx.nav_state
        
    def update_motors(self, left_motor, right_motor, imu=None):
        self.ctx.left_motor = left_motor
        self.ctx.right_motor = right_motor
        if imu: self.ctx.imu = imu
        
    async def start(self, start_pose=None):
        self.active = True
        self.ctx.nav_state = "SEARCHING"
        self.ctx.goal_x = None # Reset goal
        self.ctx.start_pose = start_pose or {'x':0, 'y':0, 'theta':0}
        print("🚀 BT Nav Started")

    async def stop(self):
        self.active = False
        self.ctx.nav_state = "IDLE"
        await _stop_motors(self.ctx)
        print("⏹ BT Nav Stopped")

    async def update(self, detection=None, target_pose=None, lidar_min_distance_cm=None, current_pose=None):
        if not self.active:
            return
            
        # Update Blackboard
        self.ctx.detection = detection
        self.ctx.target_pose = target_pose
        self.ctx.lidar_min_distance = lidar_min_distance_cm
        if current_pose:
            self.ctx.current_pose = current_pose
            
        # Tick Tree
        await self.tree.tick(self.ctx)
