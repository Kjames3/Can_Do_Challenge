"""
Navigation Finite State Machine for Viam Rover

This module implements a clean FSM-based navigation system with states:
- IDLE: Waiting for command
- SEARCHING: Spinning to find target
- APPROACHING: 3-phase navigation (ACQUIRE→ROTATE→DRIVE)
- ARRIVED: At target, stopped
- AVOIDING: Backing up from obstacle
- RETURNING: Navigating back to start position
"""

import asyncio
import time
import numpy as np


class NavigationState:
    """Navigation FSM States"""
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    ARRIVED = "ARRIVED"
    AVOIDING = "AVOIDING"
    RETURNING = "RETURNING"


class ApproachPhase:
    """Sub-phases within APPROACHING state"""
    ACQUIRE = "ACQUIRE"
    ROTATE = "ROTATE"
    DRIVE = "DRIVE"


class NavigationConfig:
    """Configuration for navigation behavior"""
    # Target distance
    target_distance_cm: float = 5.0   # Stop this close to target (gripper range)
    dist_threshold_cm: float = 2.0    # Tolerance (+/- 2cm)
    
    # Bearing thresholds (radians)
    bearing_threshold: float = 0.20       # ~11.5° - relaxed to reduce oscillation
    bearing_hysteresis: float = 0.15      # ~8.5° - prevents rapid switching
    large_turn_threshold: float = 0.35    # ~20° - use tank turn above this
    
    # Motor speeds (higher = fewer small movements = fewer API calls)
    rotate_speed: float = 0.28            # Tank turn speed
    pivot_speed: float = 0.25             # Pivot turn speed
    drive_speed: float = 0.50             # Forward drive speed (increased for faster approach)
    search_speed: float = 0.20            # Search rotation speed (slowed to not miss objects)
    backup_speed: float = 0.25            # Backup speed for avoiding
    
    # Camera (IMX708 - Pi Camera Module 3)
    camera_hfov_deg: float = 66.0  # IMX708 standard FOV (102° for wide version)
    frame_width: int = 1280        # Must match camera resolution in server_native.py
    
    # Acquire samples
    acquire_count: int = 3
    
    # Obstacle avoidance
    obstacle_min_distance_cm: float = 20.0  # Back up if closer than this
    backup_duration_sec: float = 0.8
    
    # Return navigation
    auto_return: bool = True              # Automatically return after reaching target
    return_distance_threshold: float = 15.0  # cm - close enough to start position
    
    # Motor drift compensation (negative = reduce left motor, positive = reduce right motor)
    drift_compensation: float = -0.10     # 10% reduction on LEFT motor (left is faster)
    
    # Pure Pursuit / Curved Drive Settings
    curvature_gain: float = 1.2           # Controls sharpness of turns (higher = sharper)
    min_drive_speed: float = 0.25         # Minimum speed to keep moving while turning


class NavigationFSM:
    """
    Finite State Machine for robot navigation.
    
    Receives detection and lidar data each frame, outputs motor commands.
    """
    
    def __init__(self, left_motor, right_motor, camera=None, imu=None, config: NavigationConfig = None):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.camera = camera
        self.imu = imu  # IMU for precision turns
        self.config = config or NavigationConfig()
        
        # State
        self.state = NavigationState.IDLE
        self.approach_phase = ApproachPhase.ACQUIRE
        
        # Approach phase data
        self.acquire_samples = []
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.target_imu_rotation = 0.0  # IMU: Stores exact angle to turn
        self.last_turn_dir = 0
        
        # Avoiding state data
        self.avoid_start_time = 0.0
        
        # Motor command coalescing (prevents Viam API flooding)
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        self._MOTOR_INTERVAL = 0.10  # 100ms = 10Hz max (was 20Hz, reduced to avoid API overflow)
        self._POWER_DEADBAND = 0.02  # 2% deadband
        self._MOTOR_TIMEOUT = 2.5    # Timeout for motor commands
        
        # Start position tracking for RETURNING state
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.return_phase = "ROTATE"  # ROTATE or DRIVE
        self.return_target_heading = 0.0
        
        # Callbacks for state changes (optional)
        self.on_state_change = None
        self.on_arrived = None
        self.on_returned = None  # Called when returned to start
        
        # Blind drive mode (for when target is lost at close range)
        self.blind_drive_start_pos = None  # (x, y) when blind drive started
        self.blind_drive_target_dist = 0.0  # How far to drive blindly
        self.last_valid_distance = 0.0  # Last known target distance
        self.BLIND_THRESHOLD_CM = 20.0  # If lost within this distance, drive blind
        self.target_lost_time = 0.0  # When we lost the target
        self.COAST_TIME_LIMIT = 0.3  # How long to wait before blind approach
        
        # Map-based navigation: Persistent goal coordinates (world frame)
        self.goal_x = None  # Target X in world coordinates
        self.goal_y = None  # Target Y in world coordinates
        self.goal_distance = 0.0  # Distance to goal (from map, not camera)
        
        # Dynamic focus tracking (prevents flooding camera with identical commands)
        self.last_focus_val = -1.0

        # Return timer / Backup
        self.arrived_time = 0.0
        self.backup_start_pos = None  # (x, y) where backup started
    
    @property
    def state_summary(self) -> str:
        """Human-readable state summary"""
        if self.state == NavigationState.APPROACHING:
            return f"{self.state}/{self.approach_phase}"
        return self.state
    
    def _set_state(self, new_state: str, phase: str = None):
        """Change state with optional callback"""
        old_state = self.state
        self.state = new_state
        if phase:
            self.approach_phase = phase
        
        if old_state != new_state:
            print(f"Nav: {old_state} → {new_state}")
            
            # Start timer when arriving
            if new_state == NavigationState.ARRIVED:
                self.arrived_time = time.time()
                
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    async def start(self, start_pose: dict = None):
        """Start navigation - enters SEARCHING state
        
        Args:
            start_pose: Optional dict with 'x', 'y', 'theta' for return navigation
        """
        self.acquire_samples.clear()
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.last_turn_dir = 0
        self._goal_frozen = False  # Reset goal freeze for new navigation
        
        # Store start position for RETURNING
        if start_pose:
            self.start_x = start_pose.get('x', 0.0)
            self.start_y = start_pose.get('y', 0.0)
            self.start_theta = start_pose.get('theta', 0.0)
        else:
            self.start_x = 0.0
            self.start_y = 0.0
            self.start_theta = 0.0
        
        self._set_state(NavigationState.SEARCHING)
        print(f"🚀 Navigation started - SEARCHING for target (start: x={self.start_x:.1f}, y={self.start_y:.1f})")
    
    async def start_approach(self):
        """Start direct approach - skips SEARCHING (for when target already visible)"""
        self.acquire_samples.clear()
        self._set_state(NavigationState.APPROACHING, ApproachPhase.ACQUIRE)
        print("🎯 Approaching target - ACQUIRING")
    
    async def stop(self):
        """Stop navigation and motors"""
        was_active = self.state != NavigationState.IDLE
        self._set_state(NavigationState.IDLE)
        await self._stop_motors()
        if was_active:
            print("⏹ Navigation stopped")
    
    def update_motors(self, left_motor, right_motor, imu=None):
        """Update motor references (call after reconnection)"""
        self.left_motor = left_motor
        self.right_motor = right_motor
        if imu is not None:
            self.imu = imu
        # Reset coalescing state
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        print("✓ NavigationFSM motors updated")
    
    async def update(self, detection: dict = None, target_pose: dict = None, 
                     lidar_min_distance_cm: float = None, current_pose: dict = None):
        """
        Called each frame to update navigation.
        
        Args:
            detection: Dict with 'distance_cm' and 'center_x' from YOLO
            target_pose: Dict with 'x', 'y', 'distance_cm' - world coordinates of target
            lidar_min_distance_cm: Minimum distance from lidar (for obstacle avoidance)
            current_pose: Dict with 'x', 'y', 'theta' for robot position
        """
        # Update current pose
        if current_pose:
            self.current_x = current_pose.get('x', 0.0)
            self.current_y = current_pose.get('y', 0.0)
            self.current_theta = current_pose.get('theta', 0.0)
        
        # MAP-BASED NAVIGATION: Update persistent goal from target_pose
        # GOAL FREEZING: Only update goal if we're farther than 30cm
        # Close-range depth estimation is noisy and can push goal behind object
        GOAL_FREEZE_THRESHOLD = 22.0  # cm
        
        # Calculate current distance to goal (if we have one)
        current_map_dist = None
        if self.goal_x is not None and self.goal_y is not None:
            dx = self.goal_x - self.current_x
            dy = self.goal_y - self.current_y
            current_map_dist = np.sqrt(dx*dx + dy*dy)
        
        if target_pose and target_pose.get('x') is not None:
            # Only update goal if we're far enough away (or don't have a goal yet)
            if current_map_dist is None or current_map_dist > GOAL_FREEZE_THRESHOLD:
                self.goal_x = target_pose['x']
                self.goal_y = target_pose['y']
                self.last_valid_distance = target_pose.get('distance_cm', 0)
            else:
                # Goal frozen - trust odometry for final approach
                if not hasattr(self, '_goal_frozen') or not self._goal_frozen:
                    print(f"  🔒 Goal frozen at {current_map_dist:.1f}cm - trusting odometry")
                    self._goal_frozen = True
            self.target_lost_time = 0  # Reset lost timer - we see the target
        elif detection:
            # Fallback: if only detection passed (legacy), reset lost timer
            self.target_lost_time = 0
        else:
            # Target lost! Start timer if not already started
            if self.target_lost_time == 0:
                self.target_lost_time = time.time()
        
        if self.state == NavigationState.IDLE:
            return
        
        if self.state == NavigationState.ARRIVED:
            # Check if auto-return is enabled
            if self.config.auto_return:
                # Wait 5 seconds before returning
                time_since_arrival = time.time() - self.arrived_time
                if time_since_arrival > 5.0:
                    self._start_return()
                elif time_since_arrival > 0.5 and int(time_since_arrival) != int(time_since_arrival - 0.1):
                     # Print countdown roughly every second (avoid spam)
                     print(f"  ⏳ Returning in {5.0 - time_since_arrival:.0f}s...")
            return
        
        # Check for obstacles first (safety) - but not during RETURNING
        if self.state != NavigationState.RETURNING:
            if lidar_min_distance_cm is not None and lidar_min_distance_cm < self.config.obstacle_min_distance_cm:
                if self.state != NavigationState.AVOIDING:
                    self._set_state(NavigationState.AVOIDING)
                    self.avoid_start_time = time.time()
                    print(f"⚠️ Obstacle detected at {lidar_min_distance_cm:.1f}cm - backing up")
        
        # Handle current state
        if self.state == NavigationState.SEARCHING:
            await self._handle_searching(detection)
        elif self.state == NavigationState.APPROACHING:
            await self._handle_approaching(detection)
        elif self.state == NavigationState.AVOIDING:
            await self._handle_avoiding()
        elif self.state == NavigationState.RETURNING:
            await self._handle_returning()
    
    # =========================================================================
    # STATE HANDLERS
    # =========================================================================
    
    async def _handle_searching(self, detection: dict):
        """SEARCHING: Spin slowly looking for target"""
        if detection and detection.get('distance_cm'):
            # Found target! Switch to APPROACHING
            self._set_state(NavigationState.APPROACHING, ApproachPhase.ACQUIRE)
            self.acquire_samples.clear()
            print("🎯 Target found! → ACQUIRING")
            await self._stop_motors()
        else:
            # Keep spinning slowly
            await self._set_motor_power(self.config.search_speed, -self.config.search_speed)
    
    async def _handle_approaching(self, detection: dict):
        """
        APPROACHING: Map-Based Navigation with Visual Refinement
        
        Uses persistent goal coordinates (self.goal_x, self.goal_y) instead of
        chasing camera detections. Camera updates the goal when visible, but
        navigation continues using odometry when target is lost.
        """
        
        # 1. Do we have a goal? (Must have seen target at least once)
        if self.goal_x is None or self.goal_y is None:
            # No goal yet - need to acquire target first
            if detection and detection.get('distance_cm'):
                # Wait for target_pose to be set by server
                await self._stop_motors()
                print("  ⏳ Waiting for goal coordinates...")
            else:
                await self._stop_motors()
            return
        
        # 2. Calculate MAP-BASED bearing and distance (Source of Truth)
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        
        # Distance to goal coordinate (from map, not camera)
        map_dist = np.sqrt(dx*dx + dy*dy)
        self.goal_distance = map_dist
        
        # Vector to target (Corrected Inverse Transform for Y-Forward System)
        # We un-rotate the World Vector (dx, dy) by the Robot's Heading (theta)
        # x_local (Right)   = dx * cos(theta) - dy * sin(theta)
        # y_local (Forward) = dx * sin(theta) + dy * cos(theta)
        cos_t = np.cos(self.current_theta)
        sin_t = np.sin(self.current_theta)
        
        # FIXED: Negate local_x to correct turn direction (was spinning away from target)
        local_x = -(dx * cos_t - dy * sin_t)   # Lateral offset (Right, negated for correct steering)
        local_y = dx * sin_t + dy * cos_t      # Forward distance
        
        # In Y-Forward system: local_y is forward, local_x is right
        # Bearing error = atan2(x, y) - angle from forward axis
        map_bearing = np.arctan2(local_x, local_y)
        
        # 3. DYNAMIC FOCUS (if camera available and target visible)
        if detection and detection.get('distance_cm'):
            det_distance = detection['distance_cm']
            if hasattr(self.camera, 'set_focus'):
                if det_distance > 100:
                    new_focus = 0.0
                else:
                    # Calibrated formula based on user data:
                    # 10cm->6.5 (65), 20cm->3.5 (70), 50cm->1.5 (75) => Avg constant ~70
                    new_focus = max(0.0, min(14.0, 70.0 / det_distance))
                if abs(new_focus - self.last_focus_val) > 0.2:
                    self.camera.set_focus(new_focus)
                    self.last_focus_val = new_focus
                    print(f"DEBUG: Focus set to {new_focus:.2f} for dist {det_distance:.1f}cm")
        
        # 4. BLIND SPOT LOGGING (for debugging)
        if not detection or not detection.get('distance_cm'):
            time_since_loss = time.time() - self.target_lost_time if self.target_lost_time > 0 else 0
            if time_since_loss > self.COAST_TIME_LIMIT:
                print(f"  🙈 Blind Nav: MapDist={map_dist:.1f}cm, MapBear={np.degrees(map_bearing):.1f}°")
        
        # 5. CHECK ARRIVAL - Increased threshold to 15cm to prevent 180 spins at close range
        # When very close, small coordinate jitter can flip bearing from "front" to "behind"
        print(f"DEBUG: Check {map_dist:.2f} <= 10.0? {map_dist <= 10.0}")
        if map_dist <= 10.0:
            print(f"✓ TARGET REACHED! Map distance: {map_dist:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            self.goal_x = None  # Clear goal for next target
            self.goal_y = None
            if self.on_arrived:
                self.on_arrived()
            return
        
        # 6. SUB-PHASE LOGIC - Use Pure Pursuit (curved approach)
        if self.approach_phase == ApproachPhase.ACQUIRE:
            # Skip acquire phase - go straight to curved driving
            self.approach_phase = ApproachPhase.DRIVE
            print(f"  ⚡ Switching to Curved Approach: Goal=({self.goal_x:.1f}, {self.goal_y:.1f}), Dist={map_dist:.1f}cm")
        
        # Combine ROTATE and DRIVE into a single "Curved Drive" phase
        if self.approach_phase == ApproachPhase.DRIVE or self.approach_phase == ApproachPhase.ROTATE:
            await self._handle_pure_pursuit(map_dist, map_bearing)
    
    async def _handle_pure_pursuit(self, distance: float, bearing: float):
        """
        Pure Pursuit (Curvature Drive) Logic.
        Calculates a constant curvature arc to the target point.
        """
        
        # 1. Check if we are close enough to stop
        print(f"DEBUG: Check {distance:.2f} <= 10.0? {distance <= 10.0}")
        if distance <= 10.0:  # Increased threshold to prevent 180 spins
            print(f"✓ TARGET REACHED (Curved)! Dist: {distance:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            self.goal_x = None
            self.goal_y = None
            if self.on_arrived:
                self.on_arrived()
            return

        # 2. Check for sharp turns (> 45 degrees)
        # If target is behind or far to the side, pivot first
        if abs(bearing) > 0.8:  # ~45 degrees
            print(f"  ↺ Angle too steep ({np.degrees(bearing):.1f}°) - Pivoting")
            await self._handle_rotate(bearing)
            return

        # 3. Calculate Curvature steering adjustment
        # Pure Pursuit: curvature = 2 * sin(alpha) / L
        # FIXED: Reduced gain and flipped sign for correct turn direction
        steering = -np.sin(bearing) * self.config.curvature_gain * 0.5
        
        # 4. Calculate Differential Speeds
        base_speed = self.config.drive_speed
        
        # Slow down when very close for precision
        if distance < 30.0:
            base_speed = max(self.config.min_drive_speed, base_speed * (distance / 30.0))

        left_power = base_speed + steering
        right_power = base_speed - steering

        # Clamp values to valid motor range (-1.0 to 1.0)
        max_pwr = max(abs(left_power), abs(right_power), 1.0)
        left_power /= max_pwr
        right_power /= max_pwr
        
        # Debug logging
        print(f"  🚗 Curved: dist={distance:.1f}cm, bear={np.degrees(bearing):.1f}°, L={left_power:.2f}, R={right_power:.2f}")

        # 5. Execute
        await self._set_motor_power(left_power, right_power)
    
    async def _handle_acquire(self, distance: float, bearing: float):
        """ACQUIRE sub-phase: Collect samples, average, and LOCK IN IMU TARGET"""
        self.acquire_samples.append((distance, bearing))
        print(f"  Sample {len(self.acquire_samples)}/{self.config.acquire_count}: dist={distance:.1f}cm")
        
        if len(self.acquire_samples) >= self.config.acquire_count:
            # Average the samples
            avg_dist = sum(s[0] for s in self.acquire_samples) / len(self.acquire_samples)
            avg_bearing = sum(s[1] for s in self.acquire_samples) / len(self.acquire_samples)
            self.target_distance = avg_dist
            self.target_bearing = avg_bearing
            
            print(f"✓ Target acquired: dist={avg_dist:.1f}cm, bearing={np.degrees(avg_bearing):.1f}°")
            
            # Decide next phase
            if abs(avg_bearing) > self.config.bearing_threshold:
                self.approach_phase = ApproachPhase.ROTATE
                
                # IMU Pre-calculation: Lock in turn amount
                if self.imu:
                    # Reset IMU "zero" to now. We want to turn exactly 'avg_bearing' amount.
                    # Camera bearing: positive = target is RIGHT
                    # Motor logic: positive remaining_turn = turn RIGHT
                    # IMU heading: positive = counterclockwise (LEFT) rotation
                    # So target_imu_rotation should match bearing sign
                    self.imu.reset_heading()
                    self.target_imu_rotation = avg_bearing
                    print(f"  → ROTATE (IMU Precision Turn: {np.degrees(avg_bearing):.1f}°)")
                else:
                    print(f"  → ROTATE (Camera Reactive Turn: {np.degrees(avg_bearing):.1f}°)")
            else:
                self.approach_phase = ApproachPhase.DRIVE
                print("  → DRIVE")
    
    async def _handle_rotate(self, bearing: float):
        """
        ROTATE sub-phase: SMOOTH PIVOT TURN (1-Wheel Locked)
        """
        
        # 1. Determine how much is left to turn
        # First check for overshoot: if camera bearing flipped sign from original target,
        # we've turned too far and need to reverse direction
        overshot = (np.sign(bearing) != np.sign(self.target_bearing) and 
                    np.sign(self.target_bearing) != 0 and
                    abs(bearing) > self.config.bearing_hysteresis)
        
        if overshot:
            # We overshot! Use current camera bearing to correct
            # This overrides IMU calculation since camera shows actual target position
            remaining_turn = bearing
            threshold = self.config.bearing_hysteresis
            print(f"  ⚠ Overshoot detected! Target now at {np.degrees(bearing):.1f}° - reversing")
        elif self.imu:
            # IMU: Calculate remaining angle based on gyro integration
            # Sign convention analysis:
            # - Camera bearing: positive = target is RIGHT of center
            # - Motor logic: positive remaining_turn = turn RIGHT (left wheel forward)
            # - IMU heading: positive = counterclockwise (LEFT), negative = clockwise (RIGHT)
            # 
            # When target is RIGHT (positive bearing), we turn RIGHT (clockwise).
            # Turning RIGHT makes heading go NEGATIVE.
            # We want remaining_turn to approach 0 as we reach the target.
            # 
            # remaining_turn = target + heading
            # - Start: target = +X, heading = 0, remaining = +X (turn right)
            # - Mid:   target = +X, heading = -X/2, remaining = +X/2 (keep turning)
            # - End:   target = +X, heading = -X, remaining = 0 (done!)
            current_heading = self.imu.get_heading()
            remaining_turn = self.target_imu_rotation + current_heading
            threshold = 0.05  # ~3 degrees precision for IMU
        else:
            # Fallback: Use current camera bearing for reactive turn control
            # The bearing tells us where the target currently is relative to center
            remaining_turn = bearing
            threshold = self.config.bearing_hysteresis

        # 2. Check if we are done (target is centered)
        if abs(bearing) <= threshold:  # Use camera bearing for final check, not IMU
            await self._stop_motors()
            self.approach_phase = ApproachPhase.DRIVE
            self.last_turn_dir = 0
            print("✓ Aligned! → DRIVE")
            # Small pause to let chassis settle before driving
            await asyncio.sleep(0.2)
            return
        
        # 3. Calculate Smooth Speed (Ramp Down)
        # "Deliberate" movement: Start at pivot_speed, slow down as we get closer.
        # MIN_MOVING_POWER ensures we don't stall due to friction.
        
        base_speed = self.config.pivot_speed  # Default 0.25
        MIN_MOVING_POWER = 0.24  # Calibrated value (overcomes friction)
        
        # Simple P-Control: Scale speed based on remaining error 
        dynamic_speed = abs(remaining_turn) * 1.5 
        
        # Clamp speed between stall-threshold and max pivot speed
        target_speed = max(MIN_MOVING_POWER, min(base_speed, dynamic_speed))
        
        # 4. Execute Pivot Turn (One Wheel Locked)
        if remaining_turn > 0:
            # Turn RIGHT -> Lock RIGHT wheel, Drive LEFT wheel forward
            # Pivot point is the right wheel
            l_pow = target_speed
            r_pow = 0.0
        else:
            # Turn LEFT -> Lock LEFT wheel, Drive RIGHT wheel forward
            # Pivot point is the left wheel
            l_pow = 0.0
            r_pow = target_speed
            
        await self._set_motor_power(l_pow, r_pow)
    
    async def _handle_drive(self, distance: float, bearing: float):
        """DRIVE sub-phase: Drive straight to target"""
        # Check if bearing drifted too much
        if abs(bearing) > self.config.bearing_threshold * 2:
            print(f"⚠ Target drifted ({np.degrees(bearing):.1f}°) - re-rotating")
            self.approach_phase = ApproachPhase.ROTATE
            await self._stop_motors()
            return
        
        # Check if arrived
        if distance <= self.config.target_distance_cm + self.config.dist_threshold_cm:
            print(f"✓ TARGET REACHED! Distance: {distance:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            if self.on_arrived:
                self.on_arrived()
            return
        
        # Drive forward
        await self._set_motor_power(self.config.drive_speed, self.config.drive_speed)
    
    async def _execute_blind_drive(self, target_distance_cm: float):
        """
        Blind Drive with ACTIVE Heading Correction.
        Uses IMU to maintain straight-line driving while using encoders for distance.
        This is true sensor fusion - IMU corrects drift, encoders measure progress.
        """
        print(f"  🙈 Blind Drive Start: {target_distance_cm:.1f}cm")
        
        # 1. LOCK THE HEADING: "This is the direction I want to go"
        if self.imu:
            target_heading = self.imu.get_heading()
            print(f"  🧭 Locked heading: {np.degrees(target_heading):.1f}°")
        else:
            target_heading = 0  # Fallback (less accurate)
        
        # 2. Record Start Position (from fused robot_state)
        start_pos = (self.current_x, self.current_y)
        
        # 3. P-Controller gain for heading correction
        kP = 0.5  # Correction strength (tune this if robot oscillates)
        
        # 4. Drive Loop
        while True:
            # --- A. CHECK DISTANCE (Using Encoders via robot_state) ---
            dx = self.current_x - start_pos[0]
            dy = self.current_y - start_pos[1]
            traveled = np.sqrt(dx*dx + dy*dy)
            
            remaining = target_distance_cm - traveled
            if remaining <= 0:
                break  # We arrived!
            
            # --- B. CALCULATE BASE POWER ---
            left_power = self.config.drive_speed
            right_power = self.config.drive_speed
            
            # --- C. ACTIVE HEADING CORRECTION (Using IMU) ---
            if self.imu:
                current_heading = self.imu.get_heading()
                # Calculate Error: Positive = drifted LEFT, Negative = drifted RIGHT
                error = current_heading - target_heading
                
                # Apply P-Controller correction
                # If we drifted LEFT (positive error), slow RIGHT / speed up LEFT
                correction = error * kP
                
                # Clamp correction to prevent wild swings
                correction = max(-0.2, min(0.2, correction))
                
                left_power += correction
                right_power -= correction
            
            # --- D. SEND CORRECTED POWER ---
            await self._set_motor_power(left_power, right_power)
            await asyncio.sleep(0.02)  # 50Hz control loop
        
        # 5. Stop and signal arrival
        await self._stop_motors()
        print("  ✓ Blind Arrival Complete!")
        self._set_state(NavigationState.ARRIVED)
        self.blind_drive_start_pos = None  # Reset for next time
        self.last_valid_distance = 0.0
        if self.on_arrived:
            self.on_arrived()
    
    async def _handle_avoiding(self):
        """AVOIDING: Back up from obstacle"""
        elapsed = time.time() - self.avoid_start_time
        
        if elapsed < self.config.backup_duration_sec:
            # Still backing up
            await self._set_motor_power(-self.config.backup_speed, -self.config.backup_speed)
        else:
            # Done backing up - return to SEARCHING
            await self._stop_motors()
            self._set_state(NavigationState.SEARCHING)
            print("↩ Backup complete → SEARCHING")
    
    def _start_return(self):
        """Initialize return to start position"""
        # Calculate bearing to start position
        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y
        
        # Distance to start
        distance_to_start = np.sqrt(dx**2 + dy**2)
        
        if distance_to_start < self.config.return_distance_threshold:
            # Already close enough to start
            print("✓ Already at start position - navigation complete")
            self._set_state(NavigationState.IDLE)
            if self.on_returned:
                self.on_returned()
            return
        
        # Start with BACKUP to clear the object
        self.return_phase = "BACKUP"
        self.backup_start_pos = (self.current_x, self.current_y)
        
        self._set_state(NavigationState.RETURNING)
        print(f"↩ RETURNING triggered. Starting BACKUP 20cm from ({self.current_x:.1f}, {self.current_y:.1f})")
    
    async def _handle_returning(self):
        """RETURNING: Navigate back to start position with Closed-Loop Control"""
        # Calculate vector to start
        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y
        distance_to_start = np.sqrt(dx**2 + dy**2)
        
        # COORDINATE FIX: Robot is Y-Forward (North=0).
        # atan2(dx, dy) gives correct heading relative to North (+Y)
        target_heading = np.arctan2(dx, dy)
        
        # Check if arrived at start
        if distance_to_start < self.config.return_distance_threshold:
            await self._stop_motors()
            self._set_state(NavigationState.IDLE)
            print(f"✓ RETURNED to start! Distance: {distance_to_start:.1f}cm")
            if self.on_returned:
                self.on_returned()
            return
        
        if self.return_phase == "BACKUP":
            # Backup 20cm to clear the object
            bx, by = self.backup_start_pos
            dist_backed = np.sqrt((self.current_x - bx)**2 + (self.current_y - by)**2)
            
            if dist_backed < 20.0:  # Back up 20cm
                await self._set_motor_power(-0.3, -0.3)
                return
            else:
                # Backup complete
                await self._stop_motors()
                print(f"  ✓ Backup complete ({dist_backed:.1f}cm). Starting return...")
                
                # Check initial heading error to decide mode
                heading_error = target_heading - self.current_theta
                while heading_error > np.pi: heading_error -= 2 * np.pi
                while heading_error < -np.pi: heading_error += 2 * np.pi
                
                # SMART TURN LOGIC:
                # If error is large (> 45 deg), Pivot first (ROTATE)
                # If error is small (< 45 deg), Arc turn (DRIVE)
                if abs(heading_error) > np.radians(45):
                    self.return_phase = "ROTATE"
                    print(f"  ↪ Large Angle ({np.degrees(heading_error):.1f}°) -> Pivoting")
                else:
                    self.return_phase = "DRIVE"
                    print(f"  ↪ Small Angle ({np.degrees(heading_error):.1f}°) -> Arcing")
                
                self.return_target_heading = target_heading
                
                # Pre-lock IMU target if available
                if self.imu:
                    self.target_imu_rotation = target_heading
                return

        if self.return_phase == "ROTATE":
            # Phase 1: Point-and-Shoot (Pivot until roughly facing home)
            
            # Use cached target to avoid oscillation near 180 degrees
            target_heading = self.return_target_heading
            
            # Calculate heading error
            heading_error = target_heading - self.current_theta
            while heading_error > np.pi: heading_error -= 2 * np.pi
            while heading_error < -np.pi: heading_error += 2 * np.pi
            
            # Threshold to switch to driving (10 degrees)
            THRESHOLD = np.radians(10)
            
            if abs(heading_error) <= THRESHOLD:
                await self._stop_motors()
                self.return_phase = "DRIVE"
                print("  ✓ Aligned → DRIVING to start")
                await asyncio.sleep(0.1)
                return
            
            # Execute Turn
            MIN_MOVING_POWER = 0.30
            # Reduced pivot speed for accuracy
            pivot_power = max(MIN_MOVING_POWER, min(self.config.pivot_speed, abs(heading_error) * 1.5))
            
            if heading_error > 0:
                # Target is LEFT -> Turn LEFT (CCW)
                # Left Back, Right Forward
                await self._set_motor_power(-pivot_power, pivot_power) 
            else:
                # Target is RIGHT -> Turn RIGHT (CW)
                # Left Forward, Right Back
                await self._set_motor_power(pivot_power, -pivot_power)
        
        elif self.return_phase == "DRIVE":
            # Phase 2: Drive & Correct (Continuous Heading Correction / Arcing)
            
            # Recalculate Error
            target_heading = np.arctan2(dx, dy)
            heading_error = target_heading - self.current_theta
            while heading_error > np.pi: heading_error -= 2 * np.pi
            while heading_error < -np.pi: heading_error += 2 * np.pi
            
            # If we drift MASSIVELY (> 60 deg), stop and rotate
            # This handles if we get knocked off course
            if abs(heading_error) > np.radians(60):
                 print(f"  ⚠ Excessive Drift ({np.degrees(heading_error):.1f}°) - Re-aligning")
                 self.return_phase = "ROTATE"
                 self.return_target_heading = target_heading
                 await self._stop_motors()
                 return

            # P-Controller for Heading
            # Stronger correction for larger errors to enable arcing
            kP = 1.0 
            correction = heading_error * kP
            
            # Base speed
            speed = self.config.drive_speed
            
            # Apply correction (Turn towards target)
            # If error > 0 (Target is LEFT), we need to turn LEFT.
            # Turn LEFT = Reduce Left Power, Increase Right Power
            left_power = speed - correction
            right_power = speed + correction
            
            # Clamp
            max_p = max(abs(left_power), abs(right_power), 1.0)
            left_power /= max_p
            right_power /= max_p
            
            await self._set_motor_power(left_power, right_power)
    
    # =========================================================================
    # MOTOR HELPERS
    # =========================================================================
    
    async def _set_motor_power(self, left: float, right: float):
        """Set motor power with coalescing and timeout handling"""
        now = time.time()
        
        # Coalescing: Skip if power values haven't changed significantly
        # and minimum interval hasn't elapsed
        left_changed = self._last_left_power is None or abs(left - self._last_left_power) > self._POWER_DEADBAND
        right_changed = self._last_right_power is None or abs(right - self._last_right_power) > self._POWER_DEADBAND
        interval_elapsed = (now - self._last_motor_time) >= self._MOTOR_INTERVAL
        is_stop = (left == 0 and right == 0)
        
        # Only send if something changed OR interval elapsed OR it's a stop command
        if not (left_changed or right_changed or interval_elapsed or is_stop):
            return
        
        try:
            if self.left_motor and self.right_motor:
                # Apply drift compensation (reduce one motor to correct for drift)
                comp = self.config.drift_compensation
                if comp < 0:
                    # Negative = reduce LEFT motor
                    left_adj = left * (1.0 + comp)  # e.g., 0.9 for -0.10
                    right_adj = right
                else:
                    # Positive = reduce RIGHT motor
                    left_adj = left
                    right_adj = right * (1.0 - comp)
                
                # Handle synchronous motors (server_native)
                if not asyncio.iscoroutinefunction(self.left_motor.set_power):
                    self.left_motor.set_power(left_adj)
                    self.right_motor.set_power(right_adj)
                else:
                    # Handle async motors (Viam SDK)
                    await asyncio.wait_for(
                        asyncio.gather(
                            self.left_motor.set_power(left_adj),
                            self.right_motor.set_power(right_adj)
                        ),
                        timeout=self._MOTOR_TIMEOUT
                    )
                
                # Update state on success
                self._last_left_power = left
                self._last_right_power = right
                self._last_motor_time = now
        except asyncio.TimeoutError:
            print(f"⚠ Nav motor timeout (L={left:.2f}, R={right:.2f})")
        except Exception as e:
            print(f"Motor error: {e}")
    
    async def _stop_motors(self):
        """Stop both motors"""
        await self._set_motor_power(0, 0)
