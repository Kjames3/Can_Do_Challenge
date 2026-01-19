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


class ReturnPhase:
    """Sub-phases for RETURNING state"""
    WAITING = "WAITING"      # Wait 5 seconds
    BACKING = "BACKING"      # Back up 20cm
    NAVIGATING = "NAVIGATING" # Pure Pursuit to start
    ALIGNING = "ALIGNING"     # Final rotation to start_theta


class NavigationConfig:
    """Configuration for navigation behavior"""
    # Target distance
    target_distance_cm: float = 4.0   # Stop closer (4cm) as requested
    dist_threshold_cm: float = 2.0    # Tolerance (+/- 2cm)
    
    # Bearing thresholds (radians)
    bearing_threshold: float = 0.20       # ~11.5° - relaxed to reduce oscillation
    bearing_hysteresis: float = 0.15      # ~8.5° - prevents rapid switching
    large_turn_threshold: float = 0.35    # ~20° - use tank turn above this
    
    # Motor speeds (higher = fewer small movements = fewer API calls)
    rotate_speed: float = 0.35            # Reduced for smoother turning
    pivot_speed: float = 0.35             # Reduced for smoother pivoting
    drive_speed: float = 0.40             # Forward drive speed (increased for faster approach)
    search_speed: float = 0.20            # Search rotation speed (slowed to not miss objects)
    backup_speed: float = 0.25            # Backup speed for avoiding
    
    # Camera (IMX708 - Pi Camera Module 3)
    camera_hfov_deg: float = 66.0  # IMX708 standard FOV (102° for wide version)
    frame_width: int = 1280        # Must match camera resolution in server_native.py
    
    # Acquire samples
    acquire_count: int = 5
    
    # Obstacle avoidance
    obstacle_min_distance_cm: float = 20.0  # Back up if closer than this
    backup_duration_sec: float = 0.8
    
    # Return navigation
    auto_return: bool = True              # Automatically return after reaching target
    return_distance_threshold: float = 15.0  # cm - Reduced to prevent early arrival trigger
    
    # Motor drift compensation
    drift_compensation: float = -0.10     # 10% reduction on LEFT motor
    
    # Offsets
    approach_x_offset: float = 40.0       # Pixel offset (Positive = Shifts aim RIGHT)

    # Pure Pursuit / Curved Drive Settings
    curvature_gain: float = 1.0           # Reduced to 1.0 for gentler curves
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
        self._goal_locked = False  # Reset goal lock
        
        # Clear previous goal only when starting a FRESH search
        self.goal_x = None
        self.goal_y = None
        self.goal_distance = 0.0
        
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
        self._goal_locked = False
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
            
        # Store detection for phases that need live valid targets (e.g. Return Alignment)
        self.latest_detection = detection if detection and detection.get('distance_cm') else {}
        
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
        
        if detection and detection.get('distance_cm'):
            # HYBRID NAV: Update goal if we have a new valid detection
            # Filter large jumps (> 30cm) unless it's the first update
            self.target_lost_time = 0
            
            if target_pose and target_pose.get('x') is not None:
                new_x = target_pose['x']
                new_y = target_pose['y']
                
                # Check distance from current goal (if exists)
                if self.goal_x is not None:
                    g_dx = new_x - self.goal_x
                    g_dy = new_y - self.goal_y
                    jump_dist = np.sqrt(g_dx*g_dx + g_dy*g_dy)
                    
                    if jump_dist > 30.0:
                        print(f"  ⚠ Goal Jump {jump_dist:.1f}cm Rejected (Filter)")
                    else:
                        # Smooth update (Low Pass Filter)
                        alpha = 0.3
                        self.goal_x = self.goal_x * (1-alpha) + new_x * alpha
                        self.goal_y = self.goal_y * (1-alpha) + new_y * alpha
                        self.last_valid_distance = target_pose.get('distance_cm', 0)
                else:
                    self.goal_x = new_x
                    self.goal_y = new_y
        
        elif detection:
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
                # Trigger return logic immediately (WAITING phase handles the 5s delay)
                self._start_return()
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
            # Extract nav parameters from detection if available
            dist = 0.0
            bearing = 0.0
            
            if detection:
                dist = detection.get('distance_cm', 0.0)
                # Calculate bearing from center_x if not present
                if 'center_x' in detection:
                    width = self.config.frame_width
                    hfov = self.config.camera_hfov_deg
                    center_x = detection['center_x']
                    # Standard bearing calculation (Positive = Right)
                    bearing = (center_x - (width/2)) * (hfov / width) * (np.pi / 180.0)
            
            await self._handle_approaching(dist, bearing)
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
        
        # 2. HYBRID DRIVE SELECTION
        # Priority: Visual Servoing (Local) > Map Navigation (Global)
        
        use_visual = False
        final_dist = 0.0
        final_bearing = 0.0
        
        # A. Visual Servoing (If Visible)
        if detection and detection.get('distance_cm'):
             det_dist = detection['distance_cm']
             
             # Calculate Visual Bearing directly from center_x
             center_x = detection.get('center_x', self.config.frame_width/2)
             vis_bearing = (center_x - (self.config.frame_width/2)) * (self.config.camera_hfov_deg / self.config.frame_width) * (np.pi / 180.0)
             
             final_dist = det_dist
             final_bearing = vis_bearing
             use_visual = True
             
             # Dynamic Focus
             if hasattr(self.camera, 'set_focus'):
                if det_dist > 100: new_focus = 0.0
                else: new_focus = max(0.0, min(14.0, 70.0 / det_dist))
                if abs(new_focus - self.last_focus_val) > 0.2:
                    self.camera.set_focus(new_focus)
                    self.last_focus_val = new_focus

        # B. Map Navigation (Fallback)
        else:
             time_since_loss = time.time() - self.target_lost_time if self.target_lost_time > 0 else 0
             if time_since_loss > self.COAST_TIME_LIMIT:
                 print(f"  🙈 Blind Nav: MapDist={map_dist:.1f}cm, MapBear={np.degrees(map_bearing):.1f}°")
             
             final_dist = map_dist
             final_bearing = map_bearing  # This comes from the filtered goal_x/y
        
        # 3. CHECK ARRIVAL
        threshold = self.config.target_distance_cm + self.config.dist_threshold_cm
        if final_dist <= threshold:
            print(f"✓ TARGET REACHED! Dist: {final_dist:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            if self.on_arrived: self.on_arrived()
            return

        # 4. EXECUTE PURE PURSUIT (With the selected data)
        if self.approach_phase == ApproachPhase.ACQUIRE:
             await self._handle_acquire(final_dist, final_bearing)
             return
        
        if self.approach_phase == ApproachPhase.DRIVE or self.approach_phase == ApproachPhase.ROTATE:
            await self._handle_pure_pursuit(final_dist, final_bearing)
            return

        # (Skip old logic below)
        return
    
    async def _handle_pure_pursuit(self, distance: float, bearing: float):
        """
        Pure Pursuit (Curvature Drive) Logic.
        Calculates a constant curvature arc to the target point.
        """
        
        # 1. Check if we are close enough to stop
        # 1. Check if we are close enough to stop
        threshold = self.config.target_distance_cm + self.config.dist_threshold_cm
        print(f"DEBUG: Check {distance:.2f} <= {threshold:.1f}? {distance <= threshold}")
        if distance <= threshold:
            print(f"✓ TARGET REACHED (Curved)! Dist: {distance:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            # Goal preserved for return alignment logic
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
            
            # LOCK THE GOAL
            # Calculate World Coordinates for the target
            # Inverse of _handle_approaching logic:
            # lx = dist * sin(bear), ly = dist * cos(bear)
            # dx = lx * cos(theta) - ly * sin(theta)
            # dy = lx * sin(theta) + ly * cos(theta)
            
            lx_biased = avg_dist * np.sin(avg_bearing)
            ly = avg_dist * np.cos(avg_bearing)
            
            # REMOVE OFFSET: The 'avg_bearing' targets the OFFSET point, not the Can.
            # We must subtract the offset to find the Real Can's local X.
            # offset_cm = dist * 1.3 * (offset_px / width)
            offset_px = self.config.approach_x_offset
            width = self.config.frame_width
            offset_cm = avg_dist * 1.3 * (offset_px / width)
            
            lx_real = lx_biased - offset_cm
            
            theta = self.current_theta
            dx = lx_real * np.cos(theta) - ly * np.sin(theta)
            dy = lx_real * np.sin(theta) + ly * np.cos(theta)
            
            self.goal_x = self.current_x + dx
            self.goal_y = self.current_y + dy
            # self._goal_locked = True  <- REMOVED (Allow updates)
            
            print(f"  🎯 Goal Set: ({self.goal_x:.1f}, {self.goal_y:.1f}) [Offset Removed]")
            
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
        
        base_speed = self.config.pivot_speed  # Default 0.40
        MIN_MOVING_POWER = 0.32  # Increased to overcome static friction (was 0.24)
        
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
    
    async def _handle_approaching(self, distance: float, bearing: float):
        """
        APPROACHING Phase: Drive using Global Map Coordinates (Pure Pursuit)
        
        Refactored to rely 100% on Map/Odometry logic.
        - If Detection: Updates the Global Goal (refines the map).
        - If No Detection: Drives to the last known Global Goal.
        """
        
        # 1. Update Global Goal (Fusion)
        # Only if we have a valid detection (passed in as distance/bearing)
        if distance > 0:
            # Calculate new goal candidate from current camera frame
            # Standard X-Forward Geometry
            # LocalX = dist * cos(bear)  (Forward)
            # LocalY = dist * sin(bear)  (Left)
            lx = distance * np.cos(bearing)
            ly = distance * np.sin(bearing)
            
            # Transform to World
            # WorldX = RobotX + lx*cos(th) - ly*sin(th)
            # WorldY = RobotY + lx*sin(th) + ly*cos(th)
            theta = self.current_theta
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            new_goal_x = self.current_x + (lx * cos_t - ly * sin_t)
            new_goal_y = self.current_y + (lx * sin_t + ly * cos_t)
            
            # Low-pass filter the goal to prevent jumpiness
            # Trust new vision 30%, keep old map 70%
            if self.goal_x is None:
                self.goal_x = new_goal_x
                self.goal_y = new_goal_y
            else:
                ALPHA = 0.3
                self.goal_x = self.goal_x * (1-ALPHA) + new_goal_x * ALPHA
                self.goal_y = self.goal_y * (1-ALPHA) + new_goal_y * ALPHA
                
            print(f"  📍 Goal Update: ({self.goal_x:.1f}, {self.goal_y:.1f}) Dist={distance:.1f}cm")

        # 2. Check Arrival
        # Calculate remaining distance to Global Goal
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        map_dist = np.hypot(dx, dy)
        
        # 3. Stop if Close Enough (using odometry, or camera if very close)
        # Trust camera distance if available and < 30cm, otherwise trust map
        check_dist = distance if (distance > 0 and distance < 30) else map_dist
        
        if check_dist < self.config.target_distance_cm:
            await self._stop_motors()
            print(f"✓ Arrived at Target! (Dist={check_dist:.1f}cm)")
            self._set_state(NavigationState.ARRIVED)
            return

        # 4. Pure Pursuit Control (Global Frame)
        # Calculate error to the Global Goal
        # Heading to goal = atan2(dy, dx) - current_theta
        global_bearing_to_goal = np.arctan2(dy, dx)
        heading_error = global_bearing_to_goal - self.current_theta
        
        # Normalize
        while heading_error > np.pi: heading_error -= 2*np.pi
        while heading_error < -np.pi: heading_error += 2*np.pi
        
        await self._handle_pure_pursuit(map_dist, heading_error)

    async def _handle_pure_pursuit(self, map_dist, heading_error):
        """Standard Pure Pursuit Controller"""
        
        # 1. Calculate Lookahead point
        # Dynamic lookahead: Farther = look ahead more
        # Min 40cm, Max 100cm
        lookahead = max(40.0, min(100.0, map_dist * 0.8))
        
        # 2. Calculate Curvature (Gamma)
        # curvature = 2 * sin(alpha) / L
        # alpha is the heading error to the point at lookahead distance
        
        # Steering Gain: P-Controller
        # Reduce "Wide Circles" by increasing gain (turn sharper)
        # But prevent oscillation.
        GAIN = 1.5 
        
        # Driving Logic
        base_speed = self.config.drive_speed
        
        # If error is huge (>60 deg), Turn in Place first
        if abs(heading_error) > np.radians(60):
             # Pivot
             pivot_pwr = 0.4
             if heading_error > 0:
                 # Target is LEFT (+). We need to turn LEFT.
                 # Previous: (-p, +p) caused Right turn.
                 # New: (+p, -p) should cause Left turn.
                 await self._set_motor_power(pivot_pwr, -pivot_pwr) # Left Turn
             else:
                 await self._set_motor_power(-pivot_pwr, pivot_pwr) # Right Turn
             print(f"  🔄 Pivot Adjust: Err={np.degrees(heading_error):.1f}°")
             return

        # Smooth Drive
        # Previous: left = base - steering (ERROR: Caused Right turn when steering > 0)
        # We want Left Turn when steering > 0.
        # If (+L, -R) = Left Turn, then we need left > right.
        # left = base + steering
        # right = base - steering
        steering = np.sin(heading_error) * GAIN
        
        left_power = base_speed + steering
        right_power = base_speed - steering
        
        # Normalization (Maintain forward speed)
        # If one motor saturated > 1.0, scale both down
        max_pwr = max(abs(left_power), abs(right_power), 1.0)
        left_power /= max_pwr
        right_power /= max_pwr
        
        # Debug
        if int(time.time()*4) % 4 == 0:
             print(f"  🚗 PP: Dist={map_dist:.1f}, Err={np.degrees(heading_error):.1f}°, L={left_power:.2f}, R={right_power:.2f}")

        await self._set_motor_power(left_power, right_power)
    
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
        distance_to_start = np.sqrt(dx**2 + dy**2)
        
        if distance_to_start < self.config.return_distance_threshold:
            print("✓ Already at start position - navigation complete")
            self._set_state(NavigationState.IDLE)
            if self.on_returned:
                self.on_returned()
            return
        
        print(f"↩ RETURNING triggered. Waiting 5s before return...")
        self.return_phase = ReturnPhase.WAITING
        self.return_start_time = time.time()
        self._set_state(NavigationState.RETURNING)
    
    async def _handle_returning(self):
        """RETURNING: Wait -> Backup -> Pure Pursuit to Start -> Align to Target"""
        
        # --- PHASE 1: WAITING (5 Seconds) ---
        if self.return_phase == ReturnPhase.WAITING:
            elapsed = time.time() - self.return_start_time
            if elapsed < 5.0:
                await self._stop_motors()
                if int(elapsed) > int(elapsed - 0.1): 
                    print(f"  ⏳ Returning in {5 - int(elapsed)}s...", end='\r')
                return
            else:
                print("\n  ◀ Starting Backup (20cm)")
                self.return_phase = ReturnPhase.BACKING
                self.backup_start_pos = (self.current_x, self.current_y)
                self.avoid_start_time = time.time() 


        # --- PHASE 2: BACKING UP (20cm) ---
        elif self.return_phase == ReturnPhase.BACKING:
            dx = self.current_x - self.backup_start_pos[0]
            dy = self.current_y - self.backup_start_pos[1]
            dist_moved = np.sqrt(dx*dx + dy*dy)
            
            # Backup Speed
            BACKUP_SPEED = 0.35
            BACKUP_DIST = 20.0 

            if dist_moved < BACKUP_DIST:
                if time.time() - self.avoid_start_time > 5.0:
                    print("  ⚠ Backup timeout - forcing return")
                    self.return_phase = ReturnPhase.NAVIGATING
                else:
                    await self._set_motor_power(-BACKUP_SPEED, -BACKUP_SPEED)
            else:
                print(f"  ✓ Backup complete ({dist_moved:.1f}cm) → Returning Home")
                await self._stop_motors()
                self.return_phase = ReturnPhase.NAVIGATING
                await asyncio.sleep(0.5)

        # --- PHASE 3: PURE PURSUIT TO START ---
        elif self.return_phase == ReturnPhase.NAVIGATING:
            dx = self.start_x - self.current_x
            dy = self.start_y - self.current_y
            map_dist = np.sqrt(dx*dx + dy*dy)
            
            # Check arrival
            if map_dist < self.config.return_distance_threshold:
                print(f"✓ Arrived at start ({map_dist:.1f}cm < {self.config.return_distance_threshold}cm). Starting Final Alignment...")
                await self._stop_motors()
                self.return_phase = ReturnPhase.ALIGNING
                return
            
            # Print distance occasionally to confirm threshold check is happening
            if int(time.time() * 2) % 10 == 0:
                 print(f"  Navigating Return: Dist={map_dist:.1f}cm (Threshold={self.config.return_distance_threshold})", end='\r')

            # 2. Coordinate Transform (World -> Robot Frame)
            # Standard Rotation Matrix (Theta = Angle of X-axis w.r.t Global X)
            # Local X (Forward) =  dx * cos(t) + dy * sin(t)
            # Local Y (Left)    = -dx * sin(t) + dy * cos(t)
            
            cos_t = np.cos(self.current_theta)
            sin_t = np.sin(self.current_theta)
            
            # Standard Rotation Transpose (Project World Vector onto Body Axes)
            # Body X (Forward) axis is at 'theta'
            # Body Y (Left) axis is at 'theta + 90'
            
            # Global Vector
            gx = dx 
            gy = dy
            
            # Project onto Body X (Forward): Dot Product
            # forward_vec = [cos(t), sin(t)]
            local_x = gx * cos_t + gy * sin_t
            
            # Project onto Body Y (Left): Dot Product
            # left_vec = [-sin(t), cos(t)]  (Rotated 90 deg CCW)
            local_y = gx * -sin_t + gy * cos_t
            
            # Calculate Bearing (Angle to goal relative to robot)
            # atan2(y, x) -> Angle from Forward (+X) to Target
            map_bearing = np.arctan2(local_y, local_x)

            await self._handle_pure_pursuit(map_dist, map_bearing)

        # --- PHASE 4: ALIGNING TO OLD TARGET ---
        # --- PHASE 4: ALIGNING TO OLD TARGET ---
        elif self.return_phase == ReturnPhase.ALIGNING:
            
            # Initialize Timeout
            if not hasattr(self, '_align_start_time'):
                self._align_start_time = time.time()
                
            # Timeout Check (12 Seconds - Increased for slower turns)
            if time.time() - self._align_start_time > 12.0:
                 print("\n⚠ ALIGN TIMEOUT - Stopping.")
                 await self._stop_motors()
                 self._set_state(NavigationState.IDLE)
                 if hasattr(self, '_align_target_heading'): del self._align_target_heading
                 if hasattr(self, '_align_start_time'): del self._align_start_time
                 if self.on_returned: self.on_returned()
                 return

            # 1. Calculate Target Heading (With 180 Degree Flip)
            if not hasattr(self, '_align_target_heading'):
                if self.goal_x is not None and self.goal_y is not None:
                    gx = self.goal_x - self.start_x
                    gy = self.goal_y - self.start_y
                    
                    # [FIX 1] 180 DEGREE OFFSET
                    # We calculate the vector, then add PI (180 deg) to face the "Old Target"
                    # as requested (since aligning to 0.0 pointed the wrong way).
                    raw_angle = np.arctan2(-gx, gy) # Standard Y-Forward math
                    self._align_target_heading = raw_angle + np.pi 
                    
                    # Normalize to -pi to +pi
                    while self._align_target_heading > np.pi: self._align_target_heading -= 2*np.pi
                    while self._align_target_heading < -np.pi: self._align_target_heading += 2*np.pi

                    print(f"  👀 Aligning to Target Vector: {np.degrees(self._align_target_heading):.1f}°")
                else:
                    # Default to 180 if no goal recorded
                    self._align_target_heading = np.pi
                    print("  👀 No Goal recorded - Defaulting to 180°")

            target_heading = self._align_target_heading
            
            # 2. Dynamic Thresholds (Relaxed)
            THRESHOLD = np.radians(8) # Relaxed from 4 deg to 8 deg for stability
            
            # Visual Override (Optional)
            if self.latest_detection:
                det = self.latest_detection
                width = self.config.frame_width
                hfov = self.config.camera_hfov_deg
                
                center_x = det.get('center_x', width/2)
                bearing = (center_x - (width/2)) * (hfov / width) * (np.pi / 180.0)
                target_heading = self.current_theta + bearing
                print(f"  🎯 Visual Lock! Bearing: {np.degrees(bearing):.1f}°", end='\r')

            # 3. Calculate Error
            heading_error = target_heading - self.current_theta
            while heading_error > np.pi: heading_error -= 2 * np.pi
            while heading_error < -np.pi: heading_error += 2 * np.pi
            
            # DEBUG: Print error to monitor oscillation
            if int(time.time() * 2) % 2 == 0: 
                print(f"  ...Align Error: {np.degrees(heading_error):.1f}°   ", end='\r')

            # 4. Check Completion
            if abs(heading_error) <= THRESHOLD:
                await self._stop_motors()
                self._set_state(NavigationState.IDLE)
                print(f"\n✓ ALIGN COMPLETE! Final Error: {np.degrees(heading_error):.1f}°")
                if hasattr(self, '_align_target_heading'): del self._align_target_heading
                if hasattr(self, '_align_start_time'): del self._align_start_time
                if self.on_returned: self.on_returned()
                return
            
            # [FIX 3] VISUAL STOP
            if self.latest_detection and abs(heading_error) < np.radians(5):
                 await self._stop_motors()
                 self._set_state(NavigationState.IDLE)
                 print(f"\n✓ VISUAL ALIGN COMPLETE! (<5° Error)")
                 if hasattr(self, '_align_target_heading'): del self._align_target_heading
                 if hasattr(self, '_align_start_time'): del self._align_start_time
                 if self.on_returned: self.on_returned()
                 return
            
            # 5. [FIX 2] GENTLER PULSE LOGIC (Increased Cycle Time)
            # If error is small (< 25 deg), use pulse width modulation
            if abs(heading_error) < np.radians(25):
                # Cycle: 0.1s ON, 0.4s OFF (Total 0.5s) - More power, longer wait
                cycle_time = time.time() % 0.5
                
                if cycle_time > 0.1: # OFF PHASE (0.4s) - Coast to stop
                    await self._stop_motors()
                    return
                
                # ON PHASE (0.1s) - Quick nudge
                pivot_power = 0.50 # Increased power to overcome stiction
            else:
                # Standard Control
                MIN_MOVING_POWER = 0.50  # Needed for skid steer rotation
                gain = 0.8
                pivot_power = max(MIN_MOVING_POWER, min(self.config.pivot_speed, abs(heading_error) * gain))
            
            # Apply Motor Power
            # [FIX] Invert Logic to match Pure Pursuit
            # Error > 0 means Target is LEFT. We must turn LEFT.
            # Left Turn = Left Motor FWD (+), Right Motor BACK (-)
            if heading_error > 0:
                await self._set_motor_power(pivot_power, -pivot_power) 
            else:
                await self._set_motor_power(-pivot_power, pivot_power)

    
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
